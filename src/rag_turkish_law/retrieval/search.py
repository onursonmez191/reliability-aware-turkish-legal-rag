"""Top-k retrieval over the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np

from ..config import load_config
from . import embed
from .index import load_index, load_meta


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


def retrieve(query: str, k: int | None = None) -> list[RetrievedPassage]:
    cfg = load_config()
    top_k = k or cfg.retrieval.top_k
    index, meta = _load_index_and_meta()

    qvec = embed.embed_queries([query])
    scores, ids = index.search(qvec.astype(np.float32), top_k)
    out: list[RetrievedPassage] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = meta[idx]
        out.append(
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
    return out


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
