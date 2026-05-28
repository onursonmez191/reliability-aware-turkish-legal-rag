"""Hugging Face Inference API client wrapper.

Reads `HF_API_TOKEN` from the environment (or `.env` if loaded by the caller).
Falls back to anonymous access; the caller is responsible for handling
rate limits or 401 errors.
"""

from __future__ import annotations

import logging
import os
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import InferenceClient

from ..config import load_config

log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@lru_cache(maxsize=4)
def get_client(model: str | None = None) -> InferenceClient:
    cfg = load_config()
    provider = (cfg.generation.get("provider") or "hf").lower()
    if provider == "ollama":
        base_url = cfg.generation.get("base_url") or "http://127.0.0.1:11434/v1"
        return InferenceClient(base_url=base_url, api_key="ollama", timeout=cfg.generation.timeout_s)
    token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
    model_name = model or cfg.generation.hf_model
    return InferenceClient(model=model_name, token=token, timeout=cfg.generation.timeout_s)


def _ollama_root() -> str:
    cfg = load_config()
    base_url = str(cfg.generation.get("base_url") or "http://127.0.0.1:11434/v1").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url.rstrip("/")


def chat(
    messages: Iterable[ChatMessage] | Iterable[dict],
    model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 2,
) -> str:
    cfg = load_config()
    msg_dicts = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
    model_name = model or cfg.generation.hf_model
    max_tokens = max_new_tokens or cfg.generation.max_new_tokens
    temp = temperature if temperature is not None else cfg.generation.temperature

    provider = (cfg.generation.get("provider") or "hf").lower()
    if provider == "ollama":
        return _retry_ollama_chat(
            msg_dicts,
            model_name,
            max_new_tokens=max_tokens,
            temperature=temp,
            retries=retries,
        )

    client = get_client(model)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat_completion(
                model=model_name,
                messages=msg_dicts,
                max_tokens=max_tokens,
                temperature=temp,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — retry on transient timeouts / 5xx
            last_exc = exc
            if attempt >= retries:
                break
            backoff = 2 ** attempt
            log.warning("Chat call failed (%s); retrying in %ds (attempt %d/%d)", type(exc).__name__, backoff, attempt + 1, retries)
            time.sleep(backoff)
    raise last_exc  # type: ignore[misc]


def _ollama_chat(
    messages: list[dict],
    model_name: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    cfg = load_config()
    keep_alive = str(cfg.get("models", {}).get("keep_alive") or "10m")
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    req = Request(
        f"{_ollama_root()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=cfg.generation.timeout_s) as resp:  # noqa: S310 - local Ollama endpoint
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail or str(exc)) from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    message = data.get("message") or {}
    return message.get("content") or data.get("response") or ""


def _retry_ollama_chat(
    messages: list[dict],
    model_name: str,
    *,
    max_new_tokens: int,
    temperature: float,
    retries: int,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _ollama_chat(
                messages,
                model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - local runtime can fail while loading
            last_exc = exc
            if attempt >= retries:
                break
            backoff = 2 ** attempt
            log.warning(
                "Ollama chat call failed (%s); retrying in %ds (attempt %d/%d)",
                type(exc).__name__,
                backoff,
                attempt + 1,
                retries,
            )
            time.sleep(backoff)
    raise last_exc  # type: ignore[misc]


def _ollama_chat_stream(
    messages: list[dict],
    model_name: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> Iterator[str]:
    cfg = load_config()
    keep_alive = str(cfg.get("models", {}).get("keep_alive") or "10m")
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    req = Request(
        f"{_ollama_root()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=cfg.generation.timeout_s) as resp:  # noqa: S310 - local Ollama endpoint
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                message = data.get("message") or {}
                token = message.get("content") or data.get("response") or ""
                if token:
                    yield token
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail or str(exc)) from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc


def chat_stream(
    messages: Iterable[ChatMessage] | Iterable[dict],
    model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    cfg = load_config()
    msg_dicts = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
    model_name = model or cfg.generation.hf_model
    max_tokens = max_new_tokens or cfg.generation.max_new_tokens
    temp = temperature if temperature is not None else cfg.generation.temperature

    provider = (cfg.generation.get("provider") or "hf").lower()
    if provider == "ollama":
        yield from _ollama_chat_stream(
            msg_dicts,
            model_name,
            max_new_tokens=max_tokens,
            temperature=temp,
        )
        return

    yield chat(
        msg_dicts,
        model=model_name,
        max_new_tokens=max_tokens,
        temperature=temp,
    )
