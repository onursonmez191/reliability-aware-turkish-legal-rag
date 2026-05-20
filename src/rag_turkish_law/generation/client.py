"""Hugging Face Inference API client wrapper.

Reads `HF_API_TOKEN` from the environment (or `.env` if loaded by the caller).
Falls back to anonymous access; the caller is responsible for handling
rate limits or 401 errors.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

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


def chat(
    messages: Iterable[ChatMessage] | Iterable[dict],
    model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 2,
) -> str:
    cfg = load_config()
    client = get_client(model)
    msg_dicts = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]
    model_name = model or cfg.generation.hf_model

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat_completion(
                model=model_name,
                messages=msg_dicts,
                max_tokens=max_new_tokens or cfg.generation.max_new_tokens,
                temperature=temperature if temperature is not None else cfg.generation.temperature,
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
