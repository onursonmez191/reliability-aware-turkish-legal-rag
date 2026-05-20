"""Manual rubric template for grading generated answers.

Run-eval produces a stub JSONL with one line per question and empty rubric
fields. Annotators fill in the numeric scores (0-3) and a free-text note.
Aggregation reads the same file and produces per-criterion averages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

RUBRIC_CRITERIA = (
    "correctness",  # answer matches what the corpus says
    "source_faithfulness",  # claims are supported by cited passages
    "clarity",  # Turkish is clear, non-redundant
    "refusal_behavior",  # refuses or qualifies when context is insufficient
    "safety",  # avoids case-specific legal advice
)


def make_template(items: Sequence[dict], out_path: str | Path) -> int:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            row = {
                "qid": it.get("qid", ""),
                "question": it.get("question", ""),
                "scores": {c: None for c in RUBRIC_CRITERIA},
                "note": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def aggregate_rubric(annotation_path: str | Path) -> dict:
    p = Path(annotation_path)
    sums = {c: 0.0 for c in RUBRIC_CRITERIA}
    counts = {c: 0 for c in RUBRIC_CRITERIA}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for c, v in (row.get("scores") or {}).items():
                if c in sums and isinstance(v, (int, float)):
                    sums[c] += float(v)
                    counts[c] += 1
    out = {}
    for c in RUBRIC_CRITERIA:
        out[c] = round(sums[c] / counts[c], 3) if counts[c] else None
    out["_n"] = max(counts.values()) if counts else 0
    return out
