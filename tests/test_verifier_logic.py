"""Verifier-side logic tests (no LLM calls)."""

from rag_turkish_law.verification.aggregate import aggregate
from rag_turkish_law.verification.claims import split_into_claims
from rag_turkish_law.verification.risk import is_case_specific_advice
from rag_turkish_law.verification.verify import ClaimVerdict, verify_answer


def test_split_into_claims_keeps_citations_on_the_right_sentence():
    text = "İlk cümle desteklenir [1]. İkinci cümle [2] daha geneldir."
    claims = split_into_claims(text)
    assert len(claims) == 2
    assert claims[0].cited == [1]
    assert claims[1].cited == [2]
    assert "[1]" not in claims[0].text_without_citations()


def test_risk_pattern_catches_money_and_imperatives():
    assert is_case_specific_advice("50.000 TL tazminat alabilirsiniz.")
    assert is_case_specific_advice("Hemen dava açın.")
    assert is_case_specific_advice("Kesinlikle haklısınız.")
    assert not is_case_specific_advice("Genel olarak iş kanunu işçiyi korur.")


def test_aggregate_marks_risk_when_any_claim_is_risky():
    claims = [
        ClaimVerdict(text="A", status="supported", source_ids=[1], reason=""),
        ClaimVerdict(text="B", status="risk", source_ids=[], reason=""),
    ]
    out = aggregate(claims)
    assert out.key == "risk"
    assert out.risk == "high"


def test_aggregate_marks_error_when_any_claim_has_verifier_error():
    claims = [
        ClaimVerdict(text="A", status="supported", source_ids=[1], reason=""),
        ClaimVerdict(text="B", status="error", source_ids=[], reason="failed"),
    ]
    out = aggregate(claims)
    assert out.key == "error"
    assert out.score == 0.0
    assert out.risk == "high"


def test_aggregate_supported_when_all_supported():
    claims = [
        ClaimVerdict(text="A", status="supported", source_ids=[1], reason=""),
        ClaimVerdict(text="B", status="supported", source_ids=[2], reason=""),
    ]
    out = aggregate(claims)
    assert out.key == "supported"
    assert out.score == 1.0
    assert out.risk == "low"


def test_aggregate_empty_is_insufficient():
    out = aggregate([])
    assert out.key == "insufficient"
    assert out.score == 0.0


def test_verify_answer_returns_error_on_model_failure(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("rag_turkish_law.verification.verify.client.chat", boom)

    verdicts = verify_answer("Bu iddia desteklenir [1].", [{"text": "kaynak"}])

    assert len(verdicts) == 1
    assert verdicts[0].status == "error"
    assert verdicts[0].cited == [1]


def test_verify_answer_downgrades_when_sources_do_not_match_citations(monkeypatch):
    def fake_chat(*_args, **_kwargs):
        return '{"verdicts":[{"index":1,"status":"supported","source_ids":[2],"reason":"ok"}]}'

    monkeypatch.setattr("rag_turkish_law.verification.verify.client.chat", fake_chat)

    verdicts = verify_answer(
        "Bu iddia desteklenir [1].",
        [{"text": "kaynak 1"}, {"text": "kaynak 2"}],
    )

    assert verdicts[0].status == "partial"
    assert verdicts[0].source_ids == [2]
    assert verdicts[0].cited == [1]
