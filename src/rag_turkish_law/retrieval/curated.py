"""Keyword overlay for small curated legal sources.

The FAISS index is still the main retriever. This module only lets us add a
small number of official/legal passages for known corpus gaps without
committing generated FAISS artifacts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from ..config import load_config


def _norm(text: str) -> str:
    return text.casefold()


@lru_cache(maxsize=1)
def load_curated_sources() -> list[dict]:
    cfg = load_config()
    path = Path(cfg.paths.get("curated_sources_file", "data/curated/legal_sources.jsonl"))
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def matching_curated_sources(original_query: str, expanded_queries: Sequence[str]) -> list[tuple[dict, float]]:
    combined = _norm(" ".join([original_query, *expanded_queries]))
    matches: list[tuple[dict, float]] = []
    for row in load_curated_sources():
        terms = row.get("match_terms") or row.get("keywords") or []
        normalized_terms = [_norm(str(term)) for term in terms]
        matched = [term for term in normalized_terms if term and term in combined]
        if not matched:
            continue

        # Curated hits should outrank generic vector matches, but not pretend to
        # be perfect. Exact legal-term matches get a small boost.
        exact_terms = ("hayvan bulunduran", "hayvanın verdiği zarar", "hayvanin verdigi zarar")
        score = 0.9 + min(len(matched), 5) * 0.01
        if any(term in combined for term in exact_terms):
            score += 0.03
        matches.append((row, min(score, 0.98)))
    return matches
