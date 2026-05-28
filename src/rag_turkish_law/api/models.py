"""Ollama model allowlist and runtime controls for the demo API."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import load_config


class ModelConfigError(ValueError):
    """Raised when the UI asks for a model outside the configured allowlist."""


class OllamaRuntimeError(RuntimeError):
    """Raised when the local Ollama runtime cannot satisfy a model action."""


def _ollama_root() -> str:
    cfg = load_config()
    base_url = str(cfg.generation.get("base_url") or "http://127.0.0.1:11434/v1").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url.rstrip("/")


def _human_bytes(value: Any) -> str | None:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}" if idx else f"{int(size)} {units[idx]}"


def _json_request(method: str, path: str, payload: dict | None = None, *, timeout: float = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        f"{_ollama_root()}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local Ollama endpoint
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaRuntimeError(detail or str(exc)) from exc
    except URLError as exc:
        raise OllamaRuntimeError(str(exc.reason)) from exc
    return json.loads(raw) if raw else {}


def _get_json(path: str, *, timeout: float = 10) -> dict:
    return _json_request("GET", path, timeout=timeout)


def _post_json(path: str, payload: dict, *, timeout: float) -> dict:
    return _json_request("POST", path, payload, timeout=timeout)


def _model_cfg() -> dict:
    return load_config().get("models", {})


def _configured_options() -> list[dict]:
    cfg = load_config()
    raw_options = _model_cfg().get("available") or [cfg.generation.hf_model]
    options: list[dict] = []
    for raw in raw_options:
        if isinstance(raw, str):
            options.append({"name": raw, "label": raw, "note": ""})
        else:
            name = str(raw.get("name", "")).strip()
            if name:
                options.append(
                    {
                        "name": name,
                        "label": str(raw.get("label") or name),
                        "note": str(raw.get("note") or ""),
                    }
                )
    return options


def allowed_model_names() -> set[str]:
    return {m["name"] for m in _configured_options()}


def default_model_name() -> str:
    cfg = load_config()
    default = str(_model_cfg().get("default") or cfg.generation.hf_model)
    return default if default in allowed_model_names() else cfg.generation.hf_model


def validate_model_name(model: str | None) -> str | None:
    if not model:
        return None
    if model not in allowed_model_names():
        raise ModelConfigError(f"Model is not configured for this demo: {model}")
    return model


def _installed_models() -> dict[str, dict]:
    data = _get_json("/api/tags", timeout=10)
    models = data.get("models") or []
    out: dict[str, dict] = {}
    for item in models:
        name = item.get("name") or item.get("model")
        if name:
            out[str(name)] = item
    return out


def _running_models() -> list[dict]:
    data = _get_json("/api/ps", timeout=10)
    return list(data.get("models") or [])


def _runtime_name(item: dict) -> str:
    return str(item.get("name") or item.get("model") or "")


def _runtime_context(item: dict) -> str | None:
    value = item.get("context") or item.get("context_length")
    return None if value in (None, "") else str(value)


def _load_timeout() -> float:
    return float(_model_cfg().get("load_timeout_s") or 600)


def _unload_timeout() -> float:
    return float(_model_cfg().get("unload_timeout_s") or 30)


def _keep_alive() -> str:
    return str(_model_cfg().get("keep_alive") or "10m")


def get_models_state() -> dict:
    try:
        installed = _installed_models()
        running_items = _running_models()
        status = "online"
        error = None
    except OllamaRuntimeError as exc:
        installed = {}
        running_items = []
        status = "offline"
        error = str(exc)

    running_by_name = {_runtime_name(item): item for item in running_items if _runtime_name(item)}
    options = []
    for option in _configured_options():
        name = option["name"]
        installed_item = installed.get(name, {})
        running_item = running_by_name.get(name, {})
        options.append(
            {
                **option,
                "installed": name in installed,
                "running": name in running_by_name,
                "size": _human_bytes(running_item.get("size") or installed_item.get("size")),
                "processor": running_item.get("processor"),
                "context": _runtime_context(running_item),
                "until": running_item.get("expires_at"),
            }
        )

    return {
        "default": default_model_name(),
        "keep_alive": _keep_alive(),
        "ollama_status": status,
        "error": error,
        "models": options,
        "running": [m for m in running_by_name if m in allowed_model_names()],
    }


def _generate_keepalive(model: str, keep_alive: str | int) -> None:
    _post_json(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive},
        timeout=_load_timeout(),
    )


def load_ollama_model(model: str, keep_alive: str | None = None) -> dict:
    model = validate_model_name(model) or model
    keep_alive = keep_alive or _keep_alive()
    running = {_runtime_name(item) for item in _running_models()}
    for other in sorted(running.intersection(allowed_model_names())):
        if other != model:
            _generate_keepalive(other, 0)
    _generate_keepalive(model, keep_alive)
    return get_models_state()


def unload_ollama_model(model: str) -> dict:
    model = validate_model_name(model) or model
    _generate_keepalive(model, 0)
    deadline = time.monotonic() + _unload_timeout()
    running: set[str] = set()
    while True:
        running = {_runtime_name(item) for item in _running_models()}
        if model not in running:
            return get_models_state()
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if model in running:
        raise OllamaRuntimeError(
            f"{model} is still reported as running after the unload request. "
            "Wait for active generations to finish, then try Eject again."
        )
    return get_models_state()
