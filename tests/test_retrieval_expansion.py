from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries
from rag_turkish_law.retrieval import search
from rag_turkish_law.retrieval.search import RetrievedPassage


def hit(pid: str, score: float = 0.8, text: str | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=pid,
        text=text or f"text {pid}",
        snippet=text or f"snippet {pid}",
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


def test_inheritance_tapu_question_expands_to_legal_terms():
    queries = expand_retrieval_queries(
        "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?"
    )

    assert queries[0].startswith("Kardeşim")
    assert "mirasçılık belgesi veraset ilamı" in queries
    assert "miras kalan taşınmaz tapu intikali" in queries
    assert "elbirliği mülkiyeti miras ortaklığı" in queries
    assert "miras ortaklığının giderilmesi ortaklığın giderilmesi izale-i şuyu" in queries


def test_standalone_tapu_question_does_not_expand_to_inheritance_terms():
    queries = expand_retrieval_queries("Satın aldığım evin tapu devri yapılmadı.")

    assert queries == ["Satın aldığım evin tapu devri yapılmadı."]


def test_rental_early_exit_question_expands_to_tbk_325_terms():
    queries = expand_retrieval_queries("Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?")

    assert queries[0].startswith("Kira")
    assert (
        "kiracı fesih dönemine uymaksızın kiralananı geri verdiğinde borçları makul süre devam eder"
        in queries
    )
    assert "kiracının kabul edilebilir yeni kiracı bulması kira borçları sona erer" in queries
    assert "Türk Borçlar Kanunu madde 325 erken tahliye makul süre yeni kiracı" in queries


def test_generic_rental_question_does_not_expand_to_early_return_terms():
    queries = expand_retrieval_queries("Kira sözleşmesi yazılı olmak zorunda mı?")

    assert queries == ["Kira sözleşmesi yazılı olmak zorunda mı?"]


def test_retrieve_boosts_hits_seen_across_expanded_queries(monkeypatch):
    def fake_retrieve_single(query, _k):
        if query.startswith("Kira sözleşmesi"):
            return [hit("NOTICE_RULE", 0.91, "kira sözleşmesi bildirim uzama")]
        return [hit("EARLY_RETURN_RULE", 0.90, "kiracı erken tahliye makul süre kira borcu")]

    monkeypatch.setattr(search, "_retrieve_single", fake_retrieve_single)

    hits = search.retrieve("Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?", k=2)

    assert hits[0].passage_id == "EARLY_RETURN_RULE"
    assert hits[0].score > hits[1].score


def test_retrieve_includes_curated_animal_source(monkeypatch):
    monkeypatch.setattr(
        search,
        "_retrieve_single",
        lambda _query, _k: [
            hit("GENERIC", 0.83, "trafik kazası tazminat başvurusu"),
            hit("LAW-6098-M67", 0.82, "hayvan bulunduran hayvanın verdiği zarar için sorumludur"),
        ],
    )

    hits = search.retrieve(
        "Köpeğim birine zarar verirse veya bir köpek bana zarar verirse tazminatı kimden istemeliyim?",
        k=3,
    )

    assert hits[0].passage_id == "CUR-TBK-067"
    assert "Hayvan bulunduran" in hits[0].title
    assert any(h.passage_id == "LAW-6098-M67" for h in hits)
    assert all(h.passage_id != "GENERIC" for h in hits)


def test_non_animal_question_does_not_include_curated_animal_source(monkeypatch):
    monkeypatch.setattr(search, "_retrieve_single", lambda _query, _k: [hit("GENERIC", 0.83)])

    hits = search.retrieve("Kira sözleşmesi bitmeden çıkabilir miyim?", k=3)

    assert [h.passage_id for h in hits] == ["GENERIC"]
