"""Evaluation-set loading helpers.

The generated held-out set is useful for retrieval smoke tests. The tracked
manual file is intentionally harder: paraphrased, ambiguous, unsupported, and
legal-advice-risk questions that are closer to the demo behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

from ..config import get_repo_root

EvalSetName = Literal["heldout", "manual", "combined"]

DEFAULT_MANUAL_EVAL_PATH = get_repo_root() / "evaluation" / "annotations" / "manual_eval.jsonl"


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    items: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {p} line {line_no}: {exc}") from exc
    return items


def _normalize_item(row: dict, *, idx: int, source: str) -> dict:
    qid = row.get("qid") or row.get("id") or f"{source}-{idx:03d}"
    question = (row.get("question") or row.get("query") or "").strip()
    if not question:
        raise ValueError(f"Evaluation item {qid} from {source} is missing `question`.")

    return {
        "qid": str(qid),
        "question": question,
        "type": row.get("type") or row.get("kind") or source,
        "gold_passage_id": row.get("gold_passage_id") or None,
        "expected_verdict": row.get("expected_verdict") or row.get("gold_verdict"),
        "note": row.get("note", ""),
        "source": source,
    }


def _normalize_many(rows: Iterable[dict], *, source: str) -> list[dict]:
    return [_normalize_item(row, idx=i, source=source) for i, row in enumerate(rows, start=1)]


def load_heldout_eval(heldout_path: str | Path) -> list[dict]:
    """Load generated held-out retrieval items."""

    return _normalize_many(_read_jsonl(heldout_path), source="heldout")


def load_manual_eval(manual_path: str | Path = DEFAULT_MANUAL_EVAL_PATH) -> list[dict]:
    """Load tracked manual/adversarial evaluation items."""

    return _normalize_many(_read_jsonl(manual_path), source="manual")


def build_eval_set(
    heldout_path: str | Path,
    manual_path: str | Path = DEFAULT_MANUAL_EVAL_PATH,
    *,
    eval_set: EvalSetName = "combined",
) -> list[dict]:
    """Return the requested evaluation split."""

    if eval_set == "heldout":
        return load_heldout_eval(heldout_path)
    if eval_set == "manual":
        return load_manual_eval(manual_path)
    if eval_set == "combined":
        return load_heldout_eval(heldout_path) + load_manual_eval(manual_path)
    raise ValueError(f"Unknown eval_set: {eval_set}")
