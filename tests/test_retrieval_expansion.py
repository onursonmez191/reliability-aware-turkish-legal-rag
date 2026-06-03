from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries
from rag_turkish_law.retrieval import search, bm25_search as _bm25_module
from rag_turkish_law.retrieval.search import RetrievedPassage


def _no_bm25(queries, k):  # noqa: ARG001
    """Stub that disables BM25 for FAISS-only tests."""
    return {}


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
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", _no_bm25)

    hits = search.retrieve("Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?", k=2)

    assert hits[0].passage_id == "EARLY_RETURN_RULE"
    assert hits[0].score > hits[1].score


def test_retrieve_uses_article_level_animal_source(monkeypatch):
    monkeypatch.setattr(
        search,
        "_retrieve_single",
        lambda _query, _k: [
            hit("GENERIC", 0.83, "trafik kazası tazminat başvurusu"),
            hit("ART-TBK-0067", 0.90, "hayvan bulunduran hayvanın verdiği zarar için sorumludur"),
        ],
    )
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", _no_bm25)

    hits = search.retrieve(
        "Köpeğim birine zarar verirse veya bir köpek bana zarar verirse tazminatı kimden istemeliyim?",
        k=3,
    )

    assert hits[0].passage_id == "ART-TBK-0067"
    assert all(h.passage_id != "CUR-TBK-067" for h in hits)


def test_non_animal_question_does_not_include_curated_animal_source(monkeypatch):
    monkeypatch.setattr(search, "_retrieve_single", lambda _query, _k: [hit("GENERIC", 0.83)])
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", _no_bm25)

    hits = search.retrieve("Kira sözleşmesi bitmeden çıkabilir miyim?", k=3)

    assert [h.passage_id for h in hits] == ["GENERIC"]


# ── Inheritance + property combined expansions ────────────────────────────────

def test_inheritance_tapu_expands_to_property_specific_terms():
    queries = expand_retrieval_queries(
        "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?"
    )
    # Property-specific expansions should fire alongside general inheritance ones
    assert any("tapu intikali mirasçılar tescil" in q for q in queries)
    assert any("ortaklığın giderilmesi davası" in q for q in queries)


def test_standalone_tapu_without_family_does_not_expand_to_inheritance_property():
    queries = expand_retrieval_queries("Satın aldığım evin tapu devri yapılmadı.")
    # No family or inheritance signal → no inheritance-property expansions
    inheritance_terms = ("tapu intikali mirasçılar", "miras intikali tapu sicili")
    assert not any(any(t in q for t in inheritance_terms) for q in queries[1:])


def test_miras_without_property_expands_to_general_not_property_specific():
    queries = expand_retrieval_queries("Miras hukuku nedir?")
    # Has inheritance signal but no property signal → general expansions only
    assert any("mirasçılık belgesi" in q for q in queries)
    # Property-specific terms should NOT appear
    assert not any("tapu intikali mirasçılar tescil" in q for q in queries)


# ── Domain-specific eval cases ────────────────────────────────────────────────

def test_haciz_query_no_inheritance_expansion():
    """Haciz (attachment) queries should not trigger inheritance expansion."""
    queries = expand_retrieval_queries("Banka hesabıma haciz konuldu, ne yapmalıyım?")
    assert not any("miras" in q.lower() for q in queries[1:])
    assert not any("mirasçı" in q.lower() for q in queries[1:])


def test_labor_wrongful_termination_expands():
    queries = expand_retrieval_queries("İşten haksız yere çıkarıldığımı düşünüyorum — ne yapabilirim?")
    assert any("haksız fesih" in q for q in queries)
    assert any("kıdem tazminatı" in q or "iş güvencesi" in q for q in queries)
    assert any("fesih" in q for q in queries)


def test_labor_retrieve_filters_non_labor_contract_hits(monkeypatch):
    def fake_retrieve_single(query, _k):
        return [
            hit("KIRA_FESIH", 0.95, "belirsiz süreli kira sözleşmesi fesih bildirimi"),
            hit("LABOR_FESIH", 0.88, "işçi haksız fesih kıdem tazminatı işe iade"),
        ]

    monkeypatch.setattr(search, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(_bm25_module, "bm25_retrieve", _no_bm25)

    hits = search.retrieve("İşten haksız yere çıkarıldığımı düşünüyorum — ne yapabilirim?", k=5)

    assert [h.passage_id for h in hits] == ["LABOR_FESIH"]


def test_animal_dog_bite_expands_to_liability_terms():
    queries = expand_retrieval_queries("Komşumun köpeği beni ısırdı — kimden tazminat isteyebilirim?")
    assert any("hayvan bulunduranın sorumluluğu" in q for q in queries)
    assert any("kusursuz sorumluluk" in q for q in queries)


def test_ayipli_mal_no_expansion():
    """Defective goods queries don't yet have an expansion rule; should pass through unchanged."""
    queries = expand_retrieval_queries("Sattığım ürün ayıplı çıktı, alıcı ne talep edebilir?")
    # Only original query returned (no domain match yet)
    assert queries[0].startswith("Sattığım")


def test_kvkk_data_deletion_no_expansion():
    queries = expand_retrieval_queries("KVKK kapsamında kişisel verilerimin silinmesini nasıl talep ederim?")
    assert queries[0].startswith("KVKK")


def test_traffic_accident_no_inheritance_expansion():
    queries = expand_retrieval_queries("Trafik kazasında zarar gördüm, sigortadan tazminat alabilir miyim?")
    assert not any("miras" in q.lower() for q in queries[1:])


def test_tebligat_query_no_expansion():
    queries = expand_retrieval_queries("Tebligat adresim değişti, nasıl bildirmeliyim?")
    assert queries[0].startswith("Tebligat")


def test_early_rent_exit_expands_to_tbk_terms():
    queries = expand_retrieval_queries(
        "Kira sözleşmesini süresi bitmeden feshetmek istiyorum, tazminat ödemem gerekir mi?"
    )
    assert any("makul süre" in q for q in queries)
    assert any("Türk Borçlar Kanunu" in q for q in queries)
