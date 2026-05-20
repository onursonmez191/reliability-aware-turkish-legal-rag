"""Ablation runners — vary top_k or rerank flag and re-evaluate retrieval."""

from __future__ import annotations

from typing import Sequence

from ..config import load_config
from .retrieval_metrics import evaluate_retrieval


def run_topk_ablation(eval_items: Sequence[dict], ks: Sequence[int] = (3, 5, 8)) -> dict:
    return {
        "ablation": "top_k",
        "results": {str(k): evaluate_retrieval(eval_items, k=k) for k in ks},
    }


def run_rerank_ablation(eval_items: Sequence[dict]) -> dict:
    cfg = load_config()
    original = bool(cfg.retrieval.rerank.enabled)
    out: dict = {"ablation": "rerank", "results": {}}
    try:
        cfg.retrieval.rerank["enabled"] = False
        out["results"]["off"] = evaluate_retrieval(eval_items, k=cfg.retrieval.top_k)
        cfg.retrieval.rerank["enabled"] = True
        out["results"]["on"] = evaluate_retrieval(eval_items, k=cfg.retrieval.top_k)
    finally:
        cfg.retrieval.rerank["enabled"] = original
    return out
