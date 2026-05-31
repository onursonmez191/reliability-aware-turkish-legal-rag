from scripts.debug_retrieval import _debug_terms


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

    assert "mirasçılık" in terms
    assert "tapu" in terms
    assert "hayvan" not in terms
