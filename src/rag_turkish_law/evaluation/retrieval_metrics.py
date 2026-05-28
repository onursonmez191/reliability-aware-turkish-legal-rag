"""Retrieval metrics and diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from ..retrieval.rerank import rerank
from ..retrieval.search import RetrievedPassage, retrieve

Retriever = Callable[[str, int], list[RetrievedPassage]]
Reranker = Callable[[str, Sequence[RetrievedPassage], int], list[RetrievedPassage]]


def _recall_at_k(hits: Sequence[RetrievedPassage], gold_id: str, k: int) -> int:
    return int(any(h.passage_id == gold_id for h in hits[:k]))


def _rank(hits: Sequence[RetrievedPassage], gold_id: str) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.passage_id == gold_id:
            return i
    return None


def _reciprocal_rank(hits: Sequence[RetrievedPassage], gold_id: str) -> float:
    rank = _rank(hits, gold_id)
    return 1.0 / rank if rank else 0.0


def _default_reranker(
    query: str,
    hits: Sequence[RetrievedPassage],
    keep_top: int,
) -> list[RetrievedPassage]:
    return rerank(query, hits, keep_top=keep_top, force=True)


def _get_hits(
    question: str,
    *,
    k: int,
    use_rerank: bool,
    candidate_k: int | None,
    retriever: Retriever,
    reranker: Reranker,
) -> list[RetrievedPassage]:
    eval_k = max(k, 5)
    if not use_rerank:
        return retriever(question, eval_k)

    initial_k = max(candidate_k or eval_k, eval_k)
    candidates = retriever(question, initial_k)
    return reranker(question, candidates, eval_k)


def evaluate_retrieval(
    eval_items: Sequence[dict],
    k: int = 5,
    *,
    use_rerank: bool = False,
    candidate_k: int | None = None,
    retriever: Retriever = retrieve,
    reranker: Reranker = _default_reranker,
) -> dict:
    """Evaluate scored items and keep diagnostics for unscored manual items."""

    rec3_sum = 0
    rec5_sum = 0
    reck_sum = 0
    rr_sum = 0.0
    n_scored = 0
    per_q: list[dict] = []

    for item in eval_items:
        hits = _get_hits(
            item["question"],
            k=k,
            use_rerank=use_rerank,
            candidate_k=candidate_k,
            retriever=retriever,
            reranker=reranker,
        )
        gold = item.get("gold_passage_id")
        row = {
            "qid": item.get("qid", ""),
            "question": item.get("question", ""),
            "source": item.get("source", ""),
            "type": item.get("type", ""),
            "expected_verdict": item.get("expected_verdict"),
            "gold_passage_id": gold,
            "top_ids": [h.passage_id for h in hits[:k]],
            "top_scores": [round(float(h.score), 4) for h in hits[:k]],
        }

        if gold:
            n_scored += 1
            rank = _rank(hits, gold)
            r3 = _recall_at_k(hits, gold, 3)
            r5 = _recall_at_k(hits, gold, 5)
            rk = _recall_at_k(hits, gold, k)
            rr = _reciprocal_rank(hits, gold)
            rec3_sum += r3
            rec5_sum += r5
            reck_sum += rk
            rr_sum += rr
            row.update({"rank": rank, "recall@3": r3, "recall@5": r5, f"recall@{k}": rk, "rr": rr})

        per_q.append(row)

    n_total = len(eval_items)
    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_unscored": n_total - n_scored,
        "k": k,
        "use_rerank": use_rerank,
        "candidate_k": candidate_k if use_rerank else None,
        "recall@3": round(rec3_sum / n_scored, 4) if n_scored else None,
        "recall@5": round(rec5_sum / n_scored, 4) if n_scored else None,
        f"recall@{k}": round(reck_sum / n_scored, 4) if n_scored else None,
        "mrr": round(rr_sum / n_scored, 4) if n_scored else None,
        "per_question": per_q,
    }
