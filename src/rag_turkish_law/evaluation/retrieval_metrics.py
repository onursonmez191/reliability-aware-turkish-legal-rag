"""Retrieval metrics and diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from ..retrieval.rerank import rerank
from ..retrieval.search import RetrievedPassage, retrieve

Retriever = Callable[[str, int], list[RetrievedPassage]]
Reranker = Callable[[str, Sequence[RetrievedPassage], int], list[RetrievedPassage]]


def _strict_gold(item: dict) -> list[str]:
    """The single annotated gold passage (strict statute/article ID)."""
    single = item.get("gold_passage_id")
    return [str(single)] if single else []


def _gold_ids(item: dict) -> list[str]:
    """Union of all acceptable gold passage IDs (answer-support judgments).

    Unions `gold_passage_id` and `gold_passage_ids` so a multi-gold annotation
    can never silently drop the primary statute ID. Items where a question is
    correctly answered by either the statute article or a passage that
    explicitly cites it carry several acceptable IDs. An item is "scored" if
    this returns a non-empty set.
    """
    out: list[str] = []
    seen: set[str] = set()
    for source in (item.get("gold_passage_id"), item.get("gold_passage_ids")):
        values = source if isinstance(source, (list, tuple)) else ([source] if source else [])
        for value in values:
            s = str(value)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _recall_at_k(hits: Sequence[RetrievedPassage], gold_ids: Sequence[str], k: int) -> int:
    gold = set(gold_ids)
    return int(any(h.passage_id in gold for h in hits[:k]))


def _rank(hits: Sequence[RetrievedPassage], gold_ids: Sequence[str]) -> int | None:
    """Best (smallest) rank among any acceptable gold id, 1-indexed."""
    gold = set(gold_ids)
    for i, h in enumerate(hits, start=1):
        if h.passage_id in gold:
            return i
    return None


def _reciprocal_rank(hits: Sequence[RetrievedPassage], gold_ids: Sequence[str]) -> float:
    rank = _rank(hits, gold_ids)
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
    """Evaluate scored items, reporting strict and answer-support recall.

    Two scoring modes are reported side by side so result files are never
    ambiguous about what a recall number means:
    - strict: only the single annotated `gold_passage_id` (statute/article ID)
    - answer-support (primary `recall@k`): the union of all acceptable IDs
      (`gold_mode: "answer_support_any"`)
    """

    # Deduplicate the recall cutoffs so k in {3, 5} is not counted twice.
    ks = tuple(dict.fromkeys((3, 5, k)))
    exp = {**{kk: 0 for kk in ks}, "rr": 0.0, "n": 0}
    strict = {**{kk: 0 for kk in ks}, "rr": 0.0, "n": 0}
    per_q: list[dict] = []

    def _accumulate(acc: dict, hits, gold: list[str]) -> dict:
        acc["n"] += 1
        rr = _reciprocal_rank(hits, gold)
        acc["rr"] += rr
        out = {"rank": _rank(hits, gold), "rr": rr}
        for kk in ks:
            r = _recall_at_k(hits, gold, kk)
            acc[kk] += r
            out[f"recall@{kk}"] = r
        return out

    for item in eval_items:
        hits = _get_hits(
            item["question"],
            k=k,
            use_rerank=use_rerank,
            candidate_k=candidate_k,
            retriever=retriever,
            reranker=reranker,
        )
        expanded_gold = _gold_ids(item)
        strict_gold = _strict_gold(item)
        row = {
            "qid": item.get("qid", ""),
            "question": item.get("question", ""),
            "source": item.get("source", ""),
            "type": item.get("type", ""),
            "expected_verdict": item.get("expected_verdict"),
            "gold_passage_id": item.get("gold_passage_id"),
            "gold_passage_ids": expanded_gold or None,
            "top_ids": [h.passage_id for h in hits[:k]],
            "top_scores": [round(float(h.score), 4) for h in hits[:k]],
        }

        if expanded_gold:
            row.update(_accumulate(exp, hits, expanded_gold))
        if strict_gold:
            s = _accumulate(strict, hits, strict_gold)
            row["strict_rank"] = s["rank"]
            row[f"strict_recall@{k}"] = s[f"recall@{k}"]

        per_q.append(row)

    def _summary(acc: dict) -> dict:
        n = acc["n"]
        summary: dict = {"n_scored": n}
        for kk in ks:
            summary[f"recall@{kk}"] = round(acc[kk] / n, 4) if n else None
        summary["mrr"] = round(acc["rr"] / n, 4) if n else None
        return summary

    n_total = len(eval_items)
    exp_summary = _summary(exp)
    return {
        "n_total": n_total,
        "n_scored": exp["n"],
        "n_unscored": n_total - exp["n"],
        "k": k,
        "use_rerank": use_rerank,
        "candidate_k": candidate_k if use_rerank else None,
        # Primary recall@k = answer-support (union of acceptable IDs).
        "gold_mode": "answer_support_any",
        **{key: exp_summary[key] for key in exp_summary if key != "n_scored"},
        # Strict single-gold recall reported alongside to avoid ambiguity.
        "strict": {"description": "single annotated gold_passage_id only", **_summary(strict)},
        "per_question": per_q,
    }
