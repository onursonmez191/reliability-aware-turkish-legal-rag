"""Unit tests for BM25 keyword retrieval and hybrid RRF fusion."""

from __future__ import annotations

import pytest

from rag_turkish_law.retrieval import bm25_search as _bm25_module
from rag_turkish_law.retrieval.bm25_search import _BM25Index, _normalize, _tokenize, bm25_retrieve
from rag_turkish_law.retrieval import search
from rag_turkish_law.retrieval.search import RetrievedPassage, _reciprocal_rank_fusion


# ── Tokenizer ────────────────────────────────────────────────────────────────

def test_tokenize_strips_turkish_diacritics():
    tokens = _tokenize("Türkçe hukuk çözümü")
    assert "turkce" in tokens
    assert "hukuk" in tokens
    assert "cozumu" in tokens


def test_tokenize_ignores_short_tokens():
    # min-length is 3; 1-2 char tokens are dropped
    tokens = _tokenize("a b de ki")
    assert tokens == []
    # 3+ char tokens are kept
    assert "ile" in _tokenize("ile bu ve ama")


def test_tokenize_handles_legal_terms():
    tokens = _tokenize("tapu intikali izale-i şuyu mirasçılık")
    assert "tapu" in tokens
    assert "intikali" in tokens
    assert "izale" in tokens
    assert "mirascilik" in tokens


# ── BM25 Index ───────────────────────────────────────────────────────────────

def _mini_corpus() -> list[str]:
    return [
        "tapu intikali miras mirasçılar tescil",
        "elbirliği mülkiyeti miras ortaklığı",
        "izale-i şuyu ortaklığın giderilmesi dava",
        "kira sözleşmesi kiracı tahliye",
        "hayvan bulunduran tazminat sorumluluk",
        "iş kanunu fesih kıdem tazminatı işçi",
    ]


def test_bm25_index_finds_exact_match():
    idx = _BM25Index(_mini_corpus())
    hits = idx.get_top_k("tapu intikali", k=3)
    assert hits, "Should return at least one hit"
    top_id, top_score = hits[0]
    assert top_id == 0, "First doc (tapu intikali…) should rank #1"
    assert top_score > 0


def test_bm25_index_finds_izale_suyu():
    idx = _BM25Index(_mini_corpus())
    hits = idx.get_top_k("izale-i şuyu ortaklığın giderilmesi", k=3)
    top_ids = [h[0] for h in hits]
    assert 2 in top_ids, "Doc 2 (izale-i şuyu…) should be in top results"


def test_bm25_index_cross_domain_query_does_not_match():
    idx = _BM25Index(_mini_corpus())
    hits = idx.get_top_k("tapu intikali", k=3)
    top_ids = [h[0] for h in hits]
    assert 3 not in top_ids, "Kira doc should not match a tapu query"
    assert 4 not in top_ids, "Hayvan doc should not match a tapu query"


def test_bm25_index_empty_query_returns_empty():
    idx = _BM25Index(_mini_corpus())
    hits = idx.get_top_k("", k=5)
    assert hits == []


def test_bm25_index_unknown_term_returns_empty():
    idx = _BM25Index(_mini_corpus())
    hits = idx.get_top_k("xyznonexistentterm", k=5)
    assert hits == []


# ── bm25_retrieve (normalization) ────────────────────────────────────────────

def _patch_bm25_load(monkeypatch, corpus):
    """Monkeypatch _load_bm25 with an in-memory index and stub metadata."""
    mock_meta = [{"passage_id": f"P-{i:04d}"} for i in range(len(corpus))]
    mock_idx = _BM25Index(corpus)
    monkeypatch.setattr(_bm25_module, "_load_bm25", lambda: (mock_idx, mock_meta))
    return mock_meta


def test_bm25_retrieve_normalizes_scores_to_unit_range(monkeypatch):
    """bm25_retrieve should return scores in [0, 1]."""
    _patch_bm25_load(monkeypatch, _mini_corpus())

    result = bm25_retrieve(["tapu intikali miras"], k=6)
    assert result, "Should return at least one hit"
    assert max(result.values()) <= 1.0
    assert min(result.values()) >= 0.0
    assert max(result.values()) == pytest.approx(1.0)


def test_bm25_retrieve_returns_passage_ids(monkeypatch):
    _patch_bm25_load(monkeypatch, _mini_corpus())

    result = bm25_retrieve(["tapu intikali"], k=3)
    assert "P-0000" in result, "Passage 0 contains 'tapu intikali' and should be returned"


def test_bm25_retrieve_multi_query_takes_max_score(monkeypatch):
    """With two overlapping queries, each passage keeps its maximum BM25 score."""
    _patch_bm25_load(monkeypatch, _mini_corpus())

    result_single = bm25_retrieve(["tapu intikali"], k=6)
    result_multi = bm25_retrieve(["tapu intikali", "tapu intikali miras mirasçılar"], k=6)

    # Multi-query should not produce lower scores than single query
    for pid, score in result_single.items():
        if pid in result_multi:
            assert result_multi[pid] >= score - 1e-9


# ── RRF fusion ───────────────────────────────────────────────────────────────

def test_rrf_scores_passage_in_both_lists_higher():
    """A passage appearing in both ranked lists should score higher than one in only one."""
    faiss_ranked = ["P-001", "P-002", "P-003"]
    bm25_ranked = ["P-003", "P-001", "P-004"]
    rrf = _reciprocal_rank_fusion([faiss_ranked, bm25_ranked])

    # P-001: rank 1 in FAISS, rank 2 in BM25 → high rrf
    # P-004: only in BM25 at rank 4 → lower rrf
    assert rrf["P-001"] > rrf["P-004"]
    # P-003: rank 3 in FAISS, rank 1 in BM25
    assert rrf["P-003"] > rrf["P-004"]


def test_rrf_passage_absent_from_list_gets_no_contribution():
    rrf = _reciprocal_rank_fusion([["P-A", "P-B"], ["P-C", "P-A"]])
    # P-B is only in list 1 at rank 2
    # P-C is only in list 2 at rank 1
    assert "P-B" in rrf
    assert "P-C" in rrf
    # P-A is in both at rank 1 and rank 2 → highest
    assert rrf["P-A"] > rrf["P-B"]
    assert rrf["P-A"] > rrf["P-C"]


def test_rrf_empty_lists_return_empty():
    assert _reciprocal_rank_fusion([]) == {}
    assert _reciprocal_rank_fusion([[], []]) == {}


def test_rrf_rrf_k_parameter_affects_scores():
    rrf_60 = _reciprocal_rank_fusion([["P-001"]], rrf_k=60)
    rrf_10 = _reciprocal_rank_fusion([["P-001"]], rrf_k=10)
    # Lower k → higher score (1/(k+rank) grows as k shrinks)
    assert rrf_10["P-001"] > rrf_60["P-001"]


# ── Hybrid retrieve integration ──────────────────────────────────────────────

def _make_hit(pid: str, score: float = 0.80, text: str = "") -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=pid,
        text=text or f"text {pid}",
        snippet=text or f"snippet {pid}",
        title=f"title {pid}",
        tag="test",
        source_dataset="test",
        score=score,
    )


def test_hybrid_retrieve_surfaces_bm25_only_hit(monkeypatch):
    """A passage found only by BM25 (not FAISS top-k) should appear in results."""

    def fake_retrieve_single(query, _k):
        return [_make_hit("FAISS-HIT", 0.78, "miras mirasçı tereke")]

    # BM25 returns a statute article about tapu intikali not in FAISS results
    def fake_bm25_retrieve(queries, k):
        return {"ART-TMK-599": 1.0, "FAISS-HIT": 0.6}

    meta_rows = [
        {
            "passage_id": "FAISS-HIT",
            "text": "miras mirasçı tereke",
            "snippet": "miras mirasçı tereke",
            "title": "Miras Hukuku",
            "tag": "test",
            "source_dataset": "test",
        },
        {
            "passage_id": "ART-TMK-599",
            "text": "tapu intikali mirasçılar tescil",
            "snippet": "tapu intikali mirasçılar tescil",
            "title": "Tapu İntikali",
            "tag": "Statute",
            "source_dataset": "TMK",
        },
    ]

    import faiss as _faiss
    import numpy as np

    dummy_index = _faiss.IndexFlatIP(4)
    dummy_index.add(np.zeros((2, 4), dtype=np.float32))

    monkeypatch.setattr(search, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(search, "_load_index_and_meta", lambda: (dummy_index, meta_rows))
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", fake_bm25_retrieve)

    hits = search.retrieve(
        "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?",
        k=5,
    )
    pids = [h.passage_id for h in hits]
    assert "ART-TMK-599" in pids, "BM25-only statute article should appear in results"


def test_hybrid_retrieve_bm25_promotes_passage_over_faiss_only(monkeypatch):
    """A passage with strong BM25 evidence should rank above a FAISS-only passage with higher raw FAISS score.

    Uses a query with no expansion triggers so there is a single FAISS call and no
    multi-hit boost to confound the BM25-derived score boost.
    """

    def fake_retrieve_single(query, _k):
        return [
            _make_hit("STRONG-BM25-HIT", 0.72),  # lower raw FAISS score
            _make_hit("FAISS-ONLY", 0.80),         # higher raw FAISS score
        ]

    def fake_bm25_retrieve(queries, k):
        # STRONG-BM25-HIT is the top BM25 result (norm 1.0 → cap 0.82);
        # FAISS-ONLY is absent from BM25 results.
        return {"STRONG-BM25-HIT": 1.0}

    meta_rows = [
        {"passage_id": "STRONG-BM25-HIT", "text": "t", "snippet": "s", "title": "T1", "tag": "test", "source_dataset": "test"},
        {"passage_id": "FAISS-ONLY", "text": "t", "snippet": "s", "title": "T2", "tag": "test", "source_dataset": "test"},
    ]

    import faiss as _faiss
    import numpy as np

    dummy_index = _faiss.IndexFlatIP(4)
    dummy_index.add(np.zeros((2, 4), dtype=np.float32))

    monkeypatch.setattr(search, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(search, "_load_index_and_meta", lambda: (dummy_index, meta_rows))
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", fake_bm25_retrieve)

    # "sicil sorgusu" has no expansion triggers → single FAISS call, no multi-hit boost
    hits = search.retrieve("sicil sorgusu bilgilendirme", k=2)
    pids = [h.passage_id for h in hits]

    # BM25 cap (0.82) > raw FAISS score (0.80), so STRONG-BM25-HIT should rank #1
    assert pids[0] == "STRONG-BM25-HIT", f"Expected STRONG-BM25-HIT first, got: {pids}"
    assert hits[0].score == pytest.approx(0.82)


def test_hybrid_retrieve_uses_rrf_not_raw_display_score(monkeypatch):
    """A BM25-only exact hit can outrank a higher-score FAISS-only fuzzy hit.

    This catches regressions where hybrid retrieval computes BM25 candidates but
    still sorts the final list only by FAISS/capped display score.
    """

    def fake_retrieve_single(query, _k):
        return [_make_hit("FAISS-ONLY", 0.95)]

    def fake_bm25_retrieve(queries, k):
        return {"BM25-ONLY": 1.0}

    meta_rows = [
        {
            "passage_id": "FAISS-ONLY",
            "text": "dense fuzzy hit",
            "snippet": "dense fuzzy hit",
            "title": "Dense Fuzzy",
            "tag": "test",
            "source_dataset": "test",
        },
        {
            "passage_id": "BM25-ONLY",
            "text": "tapu intikali mirasçılar tescil",
            "snippet": "tapu intikali mirasçılar tescil",
            "title": "Tapu İntikali",
            "tag": "Statute",
            "source_dataset": "test",
        },
    ]

    import faiss as _faiss
    import numpy as np

    dummy_index = _faiss.IndexFlatIP(4)
    dummy_index.add(np.zeros((2, 4), dtype=np.float32))

    monkeypatch.setattr(search, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(search, "_load_index_and_meta", lambda: (dummy_index, meta_rows))
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", fake_bm25_retrieve)

    hits = search.retrieve("tapu intikali", k=2)

    assert [h.passage_id for h in hits] == ["BM25-ONLY", "FAISS-ONLY"]
    assert hits[0].score == pytest.approx(0.82)
