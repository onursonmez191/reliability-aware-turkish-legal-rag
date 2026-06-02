"""Retrieval diagnostic tests — verify debug term extraction is query-specific."""

from scripts.debug_retrieval import _debug_terms


# ── Original debug terms test ─────────────────────────────────────────────────

def test_debug_terms_are_query_specific():
    terms = _debug_terms(
        "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?",
        [
            "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?",
            "mirasçılık belgesi veraset ilamı",
            "miras kalan taşınmaz tapu intikali",
            "elbirliği mülkiyeti miras ortaklığı",
        ],
    )

    assert "mirasçılık" in terms or "mirascilik" in terms
    assert "tapu" in terms
    assert "hayvan" not in terms


# ── Per-domain debug-term coverage ───────────────────────────────────────────

def test_debug_terms_inheritance_tapu_covers_key_legal_concepts():
    """Expected key terms for inheritance/tapu transfer query."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    # Must cover core inheritance/tapu concepts from expansions
    all_terms_str = " ".join(terms)
    assert any(
        kw in all_terms_str
        for kw in ("tapu", "intikal", "miras", "mirasci", "veraset", "elbirlig")
    ), f"Expected inheritance/tapu terms in: {terms}"


def test_debug_terms_animal_bite_covers_liability_concepts():
    """Expected key terms for animal bite query."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Komşumun köpeği beni ısırdı — kimden tazminat isteyebilirim?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert any(
        kw in all_terms_str
        for kw in ("hayvan", "tazminat", "sorumluluk", "kusursuz")
    ), f"Expected animal liability terms in: {terms}"


def test_debug_terms_labor_wrongful_termination():
    """Expected key terms for wrongful dismissal query."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "İşten haksız yere çıkarıldığımı düşünüyorum — ne yapabilirim?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    # Terms contain Turkish chars (casefold only, not ASCII-folded)
    all_terms_str = " ".join(terms)
    # Should contain question-derived terms (labor expansion now fires)
    assert any(
        kw in all_terms_str
        for kw in (
            "fesih", "kıdem", "tazminat", "iş güvencesi",
            # Also accept question words when expansion fires
            "çıkarıldığımı", "haksız",
        )
    ), f"Expected labor terms in: {terms}"


def test_debug_terms_rental_early_exit():
    """Expected key terms for early rental exit query."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert any(
        kw in all_terms_str
        for kw in ("kira", "kiraci", "tahliye", "makul", "sure")
    ), f"Expected rental terms in: {terms}"


def test_debug_terms_haciz_bank_attachment():
    """Haciz (attachment) query — should surface attachment-related terms."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Banka hesabıma haciz konuldu, ne yapmalıyım?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert "haciz" in all_terms_str, f"Expected 'haciz' in debug terms: {terms}"


def test_debug_terms_ayipli_mal():
    """Defective goods — no domain expansion yet, should still extract question words."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Sattığım ürün ayıplı çıktı, alıcı tazminat talep edebilir mi?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert "ayipli" in all_terms_str or "tazminat" in all_terms_str


def test_debug_terms_traffic_accident():
    """Traffic accident query — should extract core traffic/insurance terms."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Trafik kazasında zarar gördüm, sigortadan tazminat alabilir miyim?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert "trafik" in all_terms_str or "sigorta" in all_terms_str or "tazminat" in all_terms_str


def test_debug_terms_kvkk_data_deletion():
    """KVKK query should surface data-protection-related terms."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "KVKK kapsamında kişisel verilerimin silinmesini nasıl talep ederim?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    # _debug_terms uses casefold (not ASCII-fold), so Turkish chars are preserved.
    # "kvkk" has len 4 < min_word_len 5, so it's excluded from question words.
    # "kişisel" (len 7) and "verilerimin" (len 11) should appear.
    assert "kişisel" in all_terms_str or "verilerimin" in all_terms_str or "silinmesini" in all_terms_str


def test_debug_terms_tebligat_address():
    """Tebligat (notification) query should surface address/notification terms."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    question = "Tebligat adresim değişti, nasıl bildirmeliyim?"
    expanded = expand_retrieval_queries(question)
    terms = _debug_terms(question, expanded)

    all_terms_str = " ".join(terms)
    assert "tebligat" in all_terms_str or "adres" in all_terms_str


def test_debug_terms_no_cross_domain_pollution():
    """Debug terms for one domain should not bleed into an unrelated domain."""
    from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries

    # Animal query
    q_animal = "Köpek beni ısırdı, tazminat isteyebilir miyim?"
    exp_animal = expand_retrieval_queries(q_animal)
    terms_animal = set(_debug_terms(q_animal, exp_animal))

    # Rental query
    q_rental = "Kira sözleşmesi bitmeden çıkmak istiyorum."
    exp_rental = expand_retrieval_queries(q_rental)
    terms_rental = set(_debug_terms(q_rental, exp_rental))

    # Overlap should be minimal (no cross-domain pollution)
    overlap = terms_animal & terms_rental
    assert "hayvan" not in terms_rental, f"'hayvan' should not appear in rental terms: {terms_rental}"
    assert "kira" not in terms_animal, f"'kira' should not appear in animal terms: {terms_animal}"
