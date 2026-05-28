"""Manual verifier evaluation over expected verdict labels."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..api.pipeline import run_pipeline


def evaluate_verifier(eval_items: Sequence[dict], *, k: int = 5) -> dict:
    """Run verified mode for items with `expected_verdict`."""

    rows: list[dict] = []
    counts = Counter()

    for item in eval_items:
        expected = item.get("expected_verdict")
        if not expected:
            continue

        counts["expected"] += 1
        try:
            response = run_pipeline(item["question"], mode="verified", k=k)
            predicted = response.verdict.key if response.verdict else None
            error = None
        except Exception as exc:  # noqa: BLE001 - evaluation should keep row-level failures
            predicted = "error"
            error = repr(exc)

        if predicted == expected:
            counts["correct"] += 1

        rows.append(
            {
                "qid": item.get("qid", ""),
                "question": item.get("question", ""),
                "type": item.get("type", ""),
                "expected_verdict": expected,
                "predicted_verdict": predicted,
                "match": predicted == expected,
                "error": error,
            }
        )

    n = counts["expected"]
    return {
        "n_expected": n,
        "accuracy": round(counts["correct"] / n, 4) if n else None,
        "per_question": rows,
    }
