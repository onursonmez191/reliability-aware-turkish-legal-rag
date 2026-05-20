"""Hugging Face Inference API client wrapper.

Reads `HF_API_TOKEN` from the environment (or `.env` if loaded by the caller).
Falls back to anonymous access; the caller is responsible for handling
rate limits or 401 errors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from huggingface_hub import InferenceClient

from ..config import load_config


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@lru_cache(maxsize=4)
def get_client(model: str | None = None) -> InferenceClient:
    cfg = load_config()
    model_name = model or cfg.generation.hf_model
    token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
    return InferenceClient(model=model_name, token=token, timeout=cfg.generation.timeout_s)


def chat(
    messages: Iterable[ChatMessage] | Iterable[dict],
    model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    cfg = load_config()
    client = get_client(model)
    msg_dicts = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]

    response = client.chat_completion(
        messages=msg_dicts,
        max_tokens=max_new_tokens or cfg.generation.max_new_tokens,
        temperature=temperature if temperature is not None else cfg.generation.temperature,
    )
    return response.choices[0].message.content or ""
