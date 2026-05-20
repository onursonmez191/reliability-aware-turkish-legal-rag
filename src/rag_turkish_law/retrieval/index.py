"""FAISS index management.

Vectors are L2-normalized at embed time, so inner product equals cosine.
The metadata file is a parallel JSONL whose line N holds the metadata for
FAISS row N. We never store text in the FAISS index itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape {vectors.shape}")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_meta(records: Iterable[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_meta(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
