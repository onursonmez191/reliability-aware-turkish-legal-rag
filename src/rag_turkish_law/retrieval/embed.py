"""Sentence embedding for queries and passages.

`intfloat/multilingual-e5-base` requires the inputs to be prefixed with
"query: " for questions and "passage: " for documents. Mixing them up
degrades retrieval quality, so the prefixing is enforced here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import load_config

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@lru_cache(maxsize=2)
def get_embedder(model_name: str | None = None) -> SentenceTransformer:
    cfg = load_config()
    name = model_name or cfg.retrieval.embedding_model
    return SentenceTransformer(name)


def embed_queries(queries: Sequence[str], model_name: str | None = None) -> np.ndarray:
    model = get_embedder(model_name)
    texts = [QUERY_PREFIX + q for q in queries]
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_passages(
    passages: Iterable[str],
    model_name: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    cfg = load_config()
    bsz = batch_size or cfg.retrieval.batch_size
    model = get_embedder(model_name)
    texts = [PASSAGE_PREFIX + p for p in passages]
    vecs = model.encode(
        texts,
        batch_size=bsz,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype(np.float32)
