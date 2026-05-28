"""Simple retrieval confidence checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import load_config
from .search import RetrievedPassage


@dataclass
class RetrievalConfidence:
    label: str
    top_score: float
    avg_top3_score: float
    reason: str


def assess_retrieval_confidence(
    hits: Sequence[RetrievedPassage],
    *,
    min_top_score: float | None = None,
    min_avg_top3_score: float | None = None,
) -> RetrievalConfidence:
    cfg = load_config()
    confidence_cfg = cfg.retrieval.get("confidence", {})
    if not confidence_cfg.get("enabled", True):
        return RetrievalConfidence("high", 1.0, 1.0, "confidence gate disabled")

    top_threshold = float(
        min_top_score
        if min_top_score is not None
        else confidence_cfg.get("min_top_score", 0.75)
    )
    avg_threshold = float(
        min_avg_top3_score
        if min_avg_top3_score is not None
        else confidence_cfg.get("min_avg_top3_score", 0.70)
    )

    if not hits:
        return RetrievalConfidence("low", 0.0, 0.0, "no retrieved passages")

    scores = [float(h.score) for h in hits]
    top_score = scores[0]
    avg_top3 = sum(scores[:3]) / min(3, len(scores))

    top_ok = top_score >= top_threshold
    avg_ok = avg_top3 >= avg_threshold
    if top_ok and avg_ok:
        label = "high"
        reason = "top and average retrieval scores passed thresholds"
    elif top_ok or avg_ok:
        label = "medium"
        reason = "only one retrieval score threshold passed"
    else:
        label = "low"
        reason = "retrieval scores were below thresholds"

    return RetrievalConfidence(label, top_score, avg_top3, reason)
