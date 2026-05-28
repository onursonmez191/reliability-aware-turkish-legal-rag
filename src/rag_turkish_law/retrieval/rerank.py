"""Optional cross-encoder reranker.

Disabled by default in `configs/default.yaml`. Loading the model is expensive,
so we only import sentence_transformers' CrossEncoder when actually needed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from ..config import load_config
from .search import RetrievedPassage


@lru_cache(maxsize=2)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


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
    scores = reranker.predict(pairs)
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
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
