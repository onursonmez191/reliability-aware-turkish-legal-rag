"""Keyword overlay for curated legal sources.

The FAISS index is the main retriever. This module provides two retrieval
paths for curated passages:

1. Keyword matching — instant, no index rebuild required. Used when FAISS
   hasn't been rebuilt after new articles were scraped.
2. FAISS embedding — applies once build_index.py is re-run. All files in
   data/curated/ are picked up automatically.

Performance note: `_load_all` uses an inverted index over the first "anchor"
term per row so large scrape files (1000+ rows) don't cause O(n) scans.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from ..config import load_config


def _norm(text: str) -> str:
    return text.casefold()


def _curated_dir() -> Path:
    cfg = load_config()
    primary = Path(cfg.paths.get("curated_sources_file", "data/curated/legal_sources.jsonl"))
    return primary.parent


@lru_cache(maxsize=1)
def _load_all() -> tuple[list[dict], dict[str, list[int]]]:
    """Load curated sources for keyword overlay.

    Only hand-curated files (legal_sources.jsonl) are used for keyword matching.
    Auto-scraped files (law_articles.jsonl) are large enough that generic term
    matches would override FAISS scores for semantically better results — so
    they are embedded into FAISS only (via build_index.py) and not keyword-matched.

    Returns:
        rows: passage dicts from hand-curated sources
        inv:  inverted index mapping each match_term → list of row indices
    """
    curated_dir = _curated_dir()
    rows: list[dict] = []
    seen_ids: set[str] = set()

    # Only keyword-overlay from the hand-curated file, not the scraped bulk file
    hand_curated_files = {"legal_sources.jsonl"}

    for jsonl_path in sorted(curated_dir.glob("*.jsonl")):
        if jsonl_path.name not in hand_curated_files:
            continue
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = row.get("passage_id", "")
                if pid and pid in seen_ids:
                    continue
                if pid:
                    seen_ids.add(pid)
                # Pre-normalise terms once at load time
                raw_terms = row.get("match_terms") or row.get("keywords") or []
                row["_norm_terms"] = [_norm(str(t)) for t in raw_terms if t]
                rows.append(row)

    # Build inverted index: term → row indices
    inv: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for term in row["_norm_terms"]:
            inv[term].append(i)

    return rows, dict(inv)


def load_curated_sources() -> list[dict]:
    """Return hand-curated sources (keyword overlay set only)."""
    rows, _ = _load_all()
    return rows


def matching_curated_sources(original_query: str, expanded_queries: Sequence[str]) -> list[tuple[dict, float]]:
    combined = _norm(" ".join([original_query, *expanded_queries]))
    rows, inv = _load_all()

    # Word-boundary matching: pad combined with spaces so "tbk m.4" doesn't
    # false-match against queries containing "tbk m.49" or "tbk m.400".
    combined_spaced = f" {combined} "
    candidate_indices: set[int] = set()
    for term, idxs in inv.items():
        if f" {term} " in combined_spaced:
            candidate_indices.update(idxs)

    if not candidate_indices:
        return []

    matches: list[tuple[dict, float]] = []
    for idx in candidate_indices:
        row = rows[idx]
        norm_terms = row["_norm_terms"]
        matched = [t for t in norm_terms if f" {t} " in combined_spaced]
        if not matched:
            continue

        # Curated hits should outrank generic vector matches, not claim perfection.
        # More matched terms → slightly higher confidence.
        score = 0.90 + min(len(matched), 5) * 0.01

        # Boost when query contains an explicit article reference (e.g. "tbk m.49",
        # "6098 m.49"). These are high-precision signals that the user wants exactly
        # this article, so it should outrank generic topic-word matches.
        _art_ref_re = re.compile(r"^\w+(?:\.\w+)? (?:m\.)?\d+$")
        if any(_art_ref_re.match(t) for t in matched):
            score += 0.05

        # Small boost for exact multi-word legal phrases (high-precision signals)
        precise_terms = ("hayvan bulunduran", "hayvanın verdiği zarar", "hayvanin verdigi zarar")
        if any(t in combined for t in precise_terms):
            score += 0.03

        matches.append((row, min(score, 0.98)))

    return matches


def curated_context_filter_terms(matches: Sequence[tuple[dict, float]]) -> list[str]:
    """Return strong terms non-curated hits must contain near curated matches.

    The purpose is to keep generic high-scoring passages, such as unrelated
    traffic-compensation snippets, out of the prompt when a precise curated
    legal source already matches the question.
    """
    terms: list[str] = []
    for row, _score in matches:
        raw_terms = row.get("context_filter_terms") or []
        terms.extend(_norm(str(term)) for term in raw_terms if str(term).strip())
    return list(dict.fromkeys(terms))
