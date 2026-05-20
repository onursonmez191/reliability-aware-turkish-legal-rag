"""Convert cleaned records into searchable passages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def record_to_passage(rec: dict, index: int) -> dict:
    """Turn a cleaned QA record into a passage suitable for embedding.

    The retriever sees `text`, which combines the question and answer so that
    queries matching the question phrasing also surface the explanation.
    `title` is a short label shown in the UI; `tag` carries any dataset hint.
    """
    question = rec["question"]
    answer = rec["answer"]
    return {
        "passage_id": f"P-{index:06d}",
        "text": f"Soru: {question}\nAçıklama: {answer}",
        "snippet": answer,
        "title": question[:140],
        "source_dataset": rec.get("source_dataset", ""),
        "record_id": rec.get("record_id"),
        "tag": "Turkish Legal QA",
    }


def to_passages(records: Iterable[dict]) -> Iterator[dict]:
    for i, rec in enumerate(records):
        yield record_to_passage(rec, i)


def write_jsonl(items: Iterable[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
