"""Optional cross-encoder reranker.

Loading the model is expensive, so we only import sentence_transformers'
CrossEncoder when reranking is actually enabled.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence

from ..config import load_config
from .search import RetrievedPassage


@lru_cache(maxsize=2)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _blend_weights(rerank_cfg) -> tuple[float, float]:
    retrieval_weight = float(rerank_cfg.get("retrieval_weight", 0.7))
    model_weight = float(rerank_cfg.get("model_weight", 0.3))
    total = retrieval_weight + model_weight
    if total <= 0:
        return 1.0, 0.0
    return retrieval_weight / total, model_weight / total


def _minmax(values: Sequence[float]) -> list[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return [0.0 for _ in values]
    min_score = min(finite_values)
    max_score = max(finite_values)
    if max_score == min_score:
        return [1.0 for _ in values]
    return [
        0.0 if not math.isfinite(value) else (value - min_score) / (max_score - min_score)
        for value in values
    ]


def rerank(
    query: str,
    passages: Sequence[RetrievedPassage],
    keep_top: int | None = None,
    model_name: str | None = None,
    force: bool = False,
) -> list[RetrievedPassage]:
    cfg = load_config()
    if not force and not cfg.retrieval.rerank.enabled:
        return list(passages)
    if not passages:
        return []

    name = model_name or cfg.retrieval.rerank.model
    keep = keep_top or cfg.retrieval.rerank.keep_top

    reranker = _get_reranker(name)
    pairs = [(query, p.text) for p in passages]
    raw_scores = [float(score) for score in reranker.predict(pairs)]
    model_scores = _minmax(raw_scores)
    retrieval_weight, model_weight = _blend_weights(cfg.retrieval.rerank)
    scores = [
        (retrieval_weight * _clamp01(p.score)) + (model_weight * model_score)
        for p, model_score in zip(passages, model_scores, strict=True)
    ]
    preserve_top_score = float(cfg.retrieval.rerank.get("preserve_top_retrieval_score", 0.9))
    if passages and passages[0].score >= preserve_top_score and scores:
        scores[0] = max(scores[0], max(scores))
    order = sorted(
        range(len(passages)),
        key=lambda i: (scores[i], passages[i].score),
        reverse=True,
    )
    out: list[RetrievedPassage] = []
    for rank_idx, orig_idx in enumerate(order[:keep]):
        p = passages[orig_idx]
        out.append(
            RetrievedPassage(
                passage_id=p.passage_id,
                text=p.text,
                snippet=p.snippet,
                title=p.title,
                tag=p.tag,
                source_dataset=p.source_dataset,
                score=float(scores[orig_idx]),
            )
        )
    return out
