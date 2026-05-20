"""Post-processing helpers for generated answers."""

from __future__ import annotations

import re
from typing import Sequence

CITE_RE = re.compile(r"\[(\d+)\]")


def find_citations(text: str) -> list[int]:
    """Return the unique 1-based citation numbers used in the answer."""
    return sorted({int(m.group(1)) for m in CITE_RE.finditer(text)})


def map_citations_to_ids(text: str, passages: Sequence[dict]) -> dict[int, str]:
    """Return {citation_number: passage_id} for citations actually used."""
    used = find_citations(text)
    out: dict[int, str] = {}
    for n in used:
        if 1 <= n <= len(passages):
            out[n] = passages[n - 1].get("passage_id") or passages[n - 1].get("id", "")
    return out
