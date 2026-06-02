"""Top-k retrieval over the FAISS index with optional BM25 hybrid search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np

from ..config import load_config
from .curated import curated_context_filter_terms, matching_curated_sources
from . import embed
from . import bm25_search as _bm25_module
from .index import load_index, load_meta
from .query_expansion import expand_retrieval_queries


@dataclass
class RetrievedPassage:
    passage_id: str
    text: str
    snippet: str
    title: str
    tag: str
    source_dataset: str
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.passage_id,
            "text": self.text,
            "snippet": self.snippet,
            "title": self.title,
            "tag": self.tag,
            "source_dataset": self.source_dataset,
            "score": self.score,
        }


@lru_cache(maxsize=1)
def _load_index_and_meta() -> tuple[faiss.Index, list[dict]]:
    cfg = load_config()
    index = load_index(cfg.paths.faiss_index)
    meta = load_meta(cfg.paths.passage_meta)
    if index.ntotal != len(meta):
        raise RuntimeError(
            f"FAISS index size ({index.ntotal}) != metadata rows ({len(meta)}). "
            "Rebuild the index with scripts/build_index.py."
        )
    return index, meta


def _row_to_passage(row: dict, score: float | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=row["passage_id"],
        text=row.get("text", ""),
        snippet=row.get("snippet", row.get("text", ""))[:400],
        title=row.get("title", ""),
        tag=row.get("tag", ""),
        source_dataset=row.get("source_dataset", ""),
        score=float(row.get("score", score if score is not None else 0.0)),
    )


def _retrieve_single(query: str, k: int) -> list[RetrievedPassage]:
    index, meta = _load_index_and_meta()

    qvec = embed.embed_queries([query])
    scores, ids = index.search(qvec.astype(np.float32), k)
    out: list[RetrievedPassage] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        out.append(_row_to_passage(meta[idx], float(score)))
    return out


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Compute RRF scores from multiple ranked lists of passage IDs.

    rrf_k=60 follows Cormack et al. 2009 recommendation.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def retrieve(query: str, k: int | None = None) -> list[RetrievedPassage]:
    cfg = load_config()
    top_k = k or cfg.retrieval.top_k

    # Candidate pool sizes
    retrieval_cfg = cfg.retrieval
    faiss_candidate_k = retrieval_cfg.get("faiss_candidate_k", 30)
    bm25_cfg = retrieval_cfg.get("bm25", {})
    bm25_enabled = bm25_cfg.get("enabled", True)
    bm25_candidate_k = bm25_cfg.get("candidate_k", 30)
    rrf_k = bm25_cfg.get("rrf_k", 60)
    # BM25-only hits are scored in [0, bm25_score_cap]; stays below strong FAISS hits
    # but can exceed the confidence threshold when keyword evidence is strong.
    bm25_score_cap = bm25_cfg.get("score_cap", 0.82)

    expanded_queries = expand_retrieval_queries(query)
    curated_matches = matching_curated_sources(query, expanded_queries)
    filter_terms = curated_context_filter_terms(curated_matches)

    # ── FAISS retrieval ──────────────────────────────────────────────────────
    faiss_by_id: dict[str, RetrievedPassage] = {}
    faiss_hit_counts: dict[str, int] = {}
    for retrieval_query in expanded_queries:
        for hit in _retrieve_single(retrieval_query, faiss_candidate_k):
            faiss_hit_counts[hit.passage_id] = faiss_hit_counts.get(hit.passage_id, 0) + 1
            current = faiss_by_id.get(hit.passage_id)
            if current is None or hit.score > current.score:
                faiss_by_id[hit.passage_id] = hit

    # Multi-hit boost for passages seen across multiple expanded queries
    for pid, count in faiss_hit_counts.items():
        if count > 1 and pid in faiss_by_id:
            faiss_by_id[pid].score = min(
                faiss_by_id[pid].score + min(0.03, 0.01 * (count - 1)), 0.99
            )

    # ── BM25 retrieval ───────────────────────────────────────────────────────
    by_id: dict[str, RetrievedPassage]

    if bm25_enabled:
        bm25_scores = _bm25_module.bm25_retrieve(expanded_queries, bm25_candidate_k)

        if bm25_scores:
            # For passages in BOTH retrievers: score = max(faiss_score, bm25_cap_score).
            # BM25 evidence can lift a passage that FAISS underranked due to vocabulary mismatch.
            for pid, bm25_norm in bm25_scores.items():
                bm25_cap_score = bm25_norm * bm25_score_cap
                if pid in faiss_by_id:
                    if bm25_cap_score > faiss_by_id[pid].score:
                        faiss_by_id[pid].score = bm25_cap_score
                else:
                    # BM25-only hit: load metadata and assign a BM25-derived score.
                    pass  # handled below after meta lookup

            # Load metadata once for BM25-only passages not already in FAISS results
            bm25_only_ids = set(bm25_scores.keys()) - set(faiss_by_id.keys())
            if bm25_only_ids:
                _, meta = _load_index_and_meta()
                meta_by_id: dict[str, dict] = {row["passage_id"]: row for row in meta}
                for pid in bm25_only_ids:
                    if pid in meta_by_id:
                        faiss_by_id[pid] = _row_to_passage(
                            meta_by_id[pid], bm25_scores[pid] * bm25_score_cap
                        )

        by_id = faiss_by_id
    else:
        by_id = faiss_by_id

    # ── Filter and curated overlay (unchanged behaviour) ─────────────────────
    if filter_terms:
        by_id = {
            pid: hit
            for pid, hit in by_id.items()
            if _hit_has_any_term(hit, filter_terms)
        }

    for row, score in curated_matches:
        hit = _row_to_passage(row, score)
        current = by_id.get(hit.passage_id)
        if current is None or hit.score > current.score:
            by_id[hit.passage_id] = hit

    return sorted(by_id.values(), key=lambda h: h.score, reverse=True)[:top_k]


def _hit_has_any_term(hit: RetrievedPassage, terms: Sequence[str]) -> bool:
    blob = f"{hit.title} {hit.snippet} {hit.text}".casefold()
    return any(term in blob for term in terms)


def retrieve_many(queries: Sequence[str], k: int | None = None) -> list[list[RetrievedPassage]]:
    cfg = load_config()
    top_k = k or cfg.retrieval.top_k
    index, meta = _load_index_and_meta()

    qvecs = embed.embed_queries(list(queries))
    scores, ids = index.search(qvecs.astype(np.float32), top_k)
    results: list[list[RetrievedPassage]] = []
    for q_scores, q_ids in zip(scores, ids):
        hits: list[RetrievedPassage] = []
        for score, idx in zip(q_scores, q_ids):
            if idx == -1:
                continue
            m = meta[idx]
            hits.append(
                RetrievedPassage(
                    passage_id=m["passage_id"],
                    text=m.get("text", ""),
                    snippet=m.get("snippet", m.get("text", ""))[:400],
                    title=m.get("title", ""),
                    tag=m.get("tag", ""),
                    source_dataset=m.get("source_dataset", ""),
                    score=float(score),
                )
            )
        results.append(hits)
    return results
