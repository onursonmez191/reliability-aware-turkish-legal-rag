from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries
from rag_turkish_law.retrieval import search
from rag_turkish_law.retrieval.search import RetrievedPassage


def hit(pid: str, score: float = 0.8) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=pid,
        text=f"text {pid}",
        snippet=f"snippet {pid}",
        title=f"title {pid}",
        tag="test",
        source_dataset="test",
        score=score,
    )


def test_animal_question_expands_to_legal_terms():
    queries = expand_retrieval_queries("Köpek bana zarar verirse tazminatı kimden isterim?")

    assert queries[0].startswith("Köpek")
    assert "hayvan bulunduranın sorumluluğu" in queries
    assert "hayvanın verdiği zarar tazminat" in queries


def test_retrieve_includes_curated_animal_source(monkeypatch):
    monkeypatch.setattr(search, "_retrieve_single", lambda _query, _k: [hit("GENERIC", 0.83)])

    hits = search.retrieve(
        "Köpeğim birine zarar verirse veya bir köpek bana zarar verirse tazminatı kimden istemeliyim?",
        k=3,
    )

    assert hits[0].passage_id == "CUR-TBK-067"
    assert "Hayvan bulunduran" in hits[0].title
    assert any(h.passage_id == "GENERIC" for h in hits)


def test_non_animal_question_does_not_include_curated_animal_source(monkeypatch):
    monkeypatch.setattr(search, "_retrieve_single", lambda _query, _k: [hit("GENERIC", 0.83)])

    hits = search.retrieve("Kira sözleşmesi bitmeden çıkabilir miyim?", k=3)

    assert [h.passage_id for h in hits] == ["GENERIC"]
