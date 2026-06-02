"""BM25Okapi keyword retrieval over passage metadata.

Provides exact-term matching as a complement to FAISS dense retrieval.
Used in hybrid retrieval with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import Sequence

from ..config import load_config
from .index import load_meta


_TR_ASCII = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")

# BM25 parameters (Okapi BM25)
_K1 = 1.5
_B = 0.75


def _normalize(text: str) -> str:
    return text.translate(_TR_ASCII).casefold()


def _tokenize(text: str) -> list[str]:
    """Extract alphanumeric tokens of length ≥ 3 after Turkish normalization.

    Length-3 minimum keeps short legal abbreviations (tbk, tmk, mkn) while
    dropping noise tokens (de, bu, ve, ki).
    """
    return re.findall(r"[a-z0-9]{3,}", _normalize(text))


def _doc_text(row: dict) -> str:
    """Build the searchable text for a passage row."""
    parts = [row.get("title", ""), row.get("text", "")]
    mt = row.get("match_terms") or []
    if mt:
        parts.append(" ".join(str(t) for t in mt))
    return " ".join(p for p in parts if p)


class _BM25Index:
    """Minimal BM25Okapi index with an inverted index for fast scoring."""

    def __init__(self, docs: list[str]) -> None:
        tokenized = [_tokenize(doc) for doc in docs]
        n = len(tokenized)
        self._n = n
        self._doc_lengths = [len(t) for t in tokenized]
        avg_dl = sum(self._doc_lengths) / max(n, 1)
        self._avg_dl = avg_dl

        # inverted index: term → {doc_id: tf}
        inv: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        df: dict[str, int] = defaultdict(int)
        for doc_id, tokens in enumerate(tokenized):
            seen: set[str] = set()
            for tok in tokens:
                inv[tok][doc_id] += 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)

        self._inv: dict[str, dict[int, int]] = {t: dict(v) for t, v in inv.items()}
        self._idf: dict[str, float] = {
            term: math.log((n - cnt + 0.5) / (cnt + 0.5) + 1)
            for term, cnt in df.items()
        }

    def get_top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-k (doc_index, bm25_score) pairs for query."""
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scores: dict[int, float] = {}
        for tok in query_tokens:
            if tok not in self._idf:
                continue
            idf = self._idf[tok]
            for doc_id, tf in self._inv[tok].items():
                dl = self._doc_lengths[doc_id]
                denom = tf + _K1 * (1 - _B + _B * dl / self._avg_dl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (_K1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


@lru_cache(maxsize=1)
def _load_bm25() -> tuple[_BM25Index, list[dict]]:
    """Build BM25 index from passage metadata (cached after first build)."""
    cfg = load_config()
    meta = load_meta(cfg.paths.passage_meta)
    docs = [_doc_text(row) for row in meta]
    return _BM25Index(docs), meta


def bm25_retrieve(queries: Sequence[str], k: int) -> dict[str, float]:
    """Return {passage_id: normalized_bm25_score} for top-k BM25 results.

    Runs BM25 for each query independently, keeps max score per passage,
    then normalizes scores to [0, 1] by dividing by the maximum raw score.

    Args:
        queries: One or more retrieval queries (original + expanded).
        k: Number of top candidates to keep per query.

    Returns:
        Mapping from passage_id to a score in [0, 1].
    """
    idx, meta = _load_bm25()

    raw_scores: dict[int, float] = {}
    for query in queries:
        for doc_id, score in idx.get_top_k(query, k):
            if score > raw_scores.get(doc_id, 0.0):
                raw_scores[doc_id] = score

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    if max_score <= 0:
        return {}

    return {
        meta[doc_id]["passage_id"]: score / max_score
        for doc_id, score in sorted(raw_scores.items(), key=lambda x: -x[1])[:k]
    }


def bm25_retrieve_passages(queries: Sequence[str], k: int) -> list[tuple[dict, float]]:
    """Return list of (row_dict, normalized_score) sorted by score descending."""
    idx, meta = _load_bm25()
    normalized = bm25_retrieve(queries, k)
    meta_by_id = {row["passage_id"]: row for row in meta}
    result = [
        (meta_by_id[pid], score)
        for pid, score in sorted(normalized.items(), key=lambda x: -x[1])
        if pid in meta_by_id
    ]
    return result
