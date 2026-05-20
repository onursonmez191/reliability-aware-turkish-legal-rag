"""Clean and deduplicate raw records before turning them into passages."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from ..config import load_config

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

QUESTION_FIELDS = ("question", "soru", "input", "prompt", "title")
ANSWER_FIELDS = ("answer", "cevap", "explanation", "aciklama", "response", "output")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _hash_question(question: str) -> str:
    return hashlib.sha1(normalize_text(question).lower().encode("utf-8")).hexdigest()


def _pick_field(row: dict, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]
    return ""


@dataclass
class CleanStats:
    seen: int = 0
    kept: int = 0
    dropped_no_question: int = 0
    dropped_no_answer: int = 0
    dropped_too_short: int = 0
    dropped_duplicate: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped_no_question": self.dropped_no_question,
            "dropped_no_answer": self.dropped_no_answer,
            "dropped_too_short": self.dropped_too_short,
            "dropped_duplicate": self.dropped_duplicate,
        }


@dataclass
class CleanResult:
    records: list[dict] = field(default_factory=list)
    stats: CleanStats = field(default_factory=CleanStats)


def clean_records(
    rows: Iterable[dict],
    min_answer_chars: int | None = None,
) -> CleanResult:
    cfg = load_config()
    min_chars = min_answer_chars if min_answer_chars is not None else cfg.data.min_answer_chars

    seen_hashes: set[str] = set()
    result = CleanResult()

    for row in rows:
        result.stats.seen += 1
        question = normalize_text(_pick_field(row, QUESTION_FIELDS))
        answer = normalize_text(_pick_field(row, ANSWER_FIELDS))

        if not question:
            result.stats.dropped_no_question += 1
            continue
        if not answer:
            result.stats.dropped_no_answer += 1
            continue
        if len(answer) < min_chars:
            result.stats.dropped_too_short += 1
            continue

        qhash = _hash_question(question)
        if qhash in seen_hashes:
            result.stats.dropped_duplicate += 1
            continue
        seen_hashes.add(qhash)

        result.records.append(
            {
                "question": question,
                "answer": answer,
                "source_dataset": row.get("_source_dataset", ""),
                "record_id": row.get("id") or row.get("_row_index"),
                "qhash": qhash,
            }
        )

    result.stats.kept = len(result.records)
    return result
