"""Verifier metrics: precision/recall per label class.

Annotators provide gold labels in the same file format as
`answer_rubric.py` plus a `gold_verdict` field. This module compares each
gold label against the system's output label and reports per-class
precision and recall.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

LABELS = ("supported", "partial", "unsupported", "insufficient", "risk")


def compare(gold: Sequence[str], pred: Sequence[str]) -> dict:
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: gold={len(gold)}, pred={len(pred)}")

    tp = Counter()
    fp = Counter()
    fn = Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    rows = {}
    for label in LABELS:
        precision_denom = tp[label] + fp[label]
        recall_denom = tp[label] + fn[label]
        precision = tp[label] / precision_denom if precision_denom else None
        recall = tp[label] / recall_denom if recall_denom else None
        rows[label] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "support_gold": recall_denom,
        }
    accuracy = sum(tp.values()) / len(gold) if gold else None
    return {
        "n": len(gold),
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
        "per_label": rows,
    }
