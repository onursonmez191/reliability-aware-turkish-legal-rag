"""FAISS build → save → load → search round-trip with synthetic vectors."""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from rag_turkish_law.retrieval.index import build_index, load_index, load_meta, save_index, save_meta


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def test_faiss_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    vectors = _normalize(rng.standard_normal((20, 16)).astype(np.float32))
    meta = [{"passage_id": f"P-{i:03d}", "text": f"t{i}"} for i in range(20)]

    index = build_index(vectors)
    assert index.ntotal == 20

    idx_path = tmp_path / "f.index"
    meta_path = tmp_path / "f.jsonl"
    save_index(index, idx_path)
    save_meta(meta, meta_path)

    loaded = load_index(idx_path)
    loaded_meta = load_meta(meta_path)
    assert loaded.ntotal == 20
    assert loaded_meta == meta

    query = vectors[3:4]
    scores, ids = loaded.search(query, 1)
    assert ids[0][0] == 3
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)
