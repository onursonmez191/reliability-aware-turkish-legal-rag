"""Central configuration loader.

Reads `configs/default.yaml` (or `$RAG_CONFIG` if set) and returns a nested
dict-like object accessible via attribute or item lookup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class _AttrDict(dict):
    """Dict that allows attribute access for nested config nodes."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, _AttrDict):
            value = _AttrDict(value)
            self[name] = value
        return value


def _resolve_paths(cfg: dict, root: Path) -> dict:
    """Make relative paths absolute against the repo root."""
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str((root / p).resolve())
    return cfg


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> _AttrDict:
    config_path = Path(path or os.getenv("RAG_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _resolve_paths(raw, REPO_ROOT)
    return _AttrDict(raw)


def get_repo_root() -> Path:
    return REPO_ROOT
