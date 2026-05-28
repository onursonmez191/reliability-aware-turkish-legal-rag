from rag_turkish_law.api import pipeline
from rag_turkish_law.retrieval.search import RetrievedPassage


def make_hit(score: float) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id="P-1",
        text="weak source",
        snippet="weak source",
        title="Weak",
        tag="",
        source_dataset="test",
        score=score,
    )


def test_low_confidence_rag_refuses_without_generation(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda _question, k: [make_hit(0.1)])
    monkeypatch.setattr(
        pipeline,
        "generate_grounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not generate")),
    )

    response = pipeline.run_pipeline("Soru?", mode="rag", k=5)

    assert "yeterince kapsamıyor" in response.answer
    assert response.verdict is None
    assert response.sources[0].id == "P-1"


def test_low_confidence_verified_returns_insufficient_verdict(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda _question, k: [make_hit(0.1)])

    response = pipeline.run_pipeline("Soru?", mode="verified", k=5)

    assert response.verdict is not None
    assert response.verdict.key == "insufficient"
    assert response.verdict.claims[0].status == "insufficient"
