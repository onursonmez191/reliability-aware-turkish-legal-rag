"""Top-k retrieval over the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np

from ..config import load_config
from .curated import matching_curated_sources
from . import embed
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


def retrieve(query: str, k: int | None = None) -> list[RetrievedPassage]:
    cfg = load_config()
    top_k = k or cfg.retrieval.top_k
    expanded_queries = expand_retrieval_queries(query)

    by_id: dict[str, RetrievedPassage] = {}
    for retrieval_query in expanded_queries:
        for hit in _retrieve_single(retrieval_query, top_k):
            current = by_id.get(hit.passage_id)
            if current is None or hit.score > current.score:
                by_id[hit.passage_id] = hit

    for row, score in matching_curated_sources(query, expanded_queries):
        hit = _row_to_passage(row, score)
        current = by_id.get(hit.passage_id)
        if current is None or hit.score > current.score:
            by_id[hit.passage_id] = hit

    return sorted(by_id.values(), key=lambda h: h.score, reverse=True)[:top_k]


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
