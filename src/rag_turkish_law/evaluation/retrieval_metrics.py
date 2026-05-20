"""Retrieval metrics: Recall@k, MRR over questions with gold_passage_id."""

from __future__ import annotations

from typing import Sequence

from ..retrieval.search import retrieve


def _recall_at_k(hits: Sequence, gold_id: str, k: int) -> int:
    return int(any(h.passage_id == gold_id for h in hits[:k]))


def _reciprocal_rank(hits: Sequence, gold_id: str) -> float:
    for i, h in enumerate(hits, start=1):
        if h.passage_id == gold_id:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(eval_items: Sequence[dict], k: int = 5) -> dict:
    scored = [item for item in eval_items if item.get("gold_passage_id")]
    if not scored:
        return {
            "n_scored": 0,
            "note": "No items with gold_passage_id; retrieval metrics skipped.",
        }

    rec3_sum = 0
    rec5_sum = 0
    rr_sum = 0.0
    per_q: list[dict] = []
    top_k = max(k, 5)

    for item in scored:
        hits = retrieve(item["question"], k=top_k)
        gold = item["gold_passage_id"]
        r3 = _recall_at_k(hits, gold, 3)
        r5 = _recall_at_k(hits, gold, 5)
        rr = _reciprocal_rank(hits, gold)
        rec3_sum += r3
        rec5_sum += r5
        rr_sum += rr
        per_q.append({"qid": item.get("qid", ""), "recall@3": r3, "recall@5": r5, "rr": rr})

    n = len(scored)
    return {
        "n_scored": n,
        "k": k,
        "recall@3": round(rec3_sum / n, 4),
        "recall@5": round(rec5_sum / n, 4),
        "mrr": round(rr_sum / n, 4),
        "per_question": per_q,
    }
