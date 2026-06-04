from types import SimpleNamespace

from rag_turkish_law.api import pipeline
from rag_turkish_law.generation.generate import GeneratedAnswer
from rag_turkish_law.retrieval.search import RetrievedPassage
from rag_turkish_law.verification.verify import ClaimVerdict


def make_hit(score: float, passage_id: str = "P-1") -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=passage_id,
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


def test_llm_mode_passes_selected_model(monkeypatch):
    seen = {}

    def fake_generate(question: str, model: str | None = None):
        seen["question"] = question
        seen["model"] = model
        return GeneratedAnswer(text="yanıt", citations=[], citation_to_passage_id={}, mode="llm")

    monkeypatch.setattr(pipeline, "generate_llm_only", fake_generate)

    response = pipeline.run_pipeline("Soru?", mode="llm", k=5, model="gemma4:31b")

    assert seen == {"question": "Soru?", "model": "gemma4:31b"}
    assert response.model == "gemma4:31b"


def test_rerank_retrieves_candidate_pool_and_keeps_requested_top_k(monkeypatch):
    seen = {}

    def fake_retrieve(_question, k):
        seen["retrieve_k"] = k
        return [make_hit(0.9, f"P-{i}") for i in range(1, 5)]

    def fake_rerank(_question, hits, keep_top=None):
        seen["rerank_seen"] = len(hits)
        seen["keep_top"] = keep_top
        return list(hits)[:keep_top]

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "rerank", fake_rerank)
    monkeypatch.setattr(
        pipeline,
        "assess_retrieval_confidence",
        lambda _hits: SimpleNamespace(label="high"),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_grounded",
        lambda *_args, **_kwargs: GeneratedAnswer(
            text="yanıt",
            citations=[],
            citation_to_passage_id={},
            mode="rag",
        ),
    )

    response = pipeline.run_pipeline("Soru?", mode="rag", k=2)

    assert seen == {"retrieve_k": 20, "rerank_seen": 4, "keep_top": 2}
    assert [s.id for s in response.sources] == ["P-1", "P-2"]


def test_verified_mode_uses_selected_model_as_verifier(monkeypatch):
    seen = {}

    monkeypatch.setattr(pipeline, "retrieve", lambda _question, k: [make_hit(0.9)])
    monkeypatch.setattr(pipeline, "rerank", lambda _question, hits, keep_top=None: list(hits)[:keep_top])
    monkeypatch.setattr(
        pipeline,
        "assess_retrieval_confidence",
        lambda _hits: SimpleNamespace(label="high"),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_grounded",
        lambda *_args, **_kwargs: GeneratedAnswer(
            text="yanıt [1].",
            citations=[1],
            citation_to_passage_id={1: "P-1"},
            mode="rag",
        ),
    )

    def fake_verify(_answer, _passages, model=None):
        seen["verifier_model"] = model
        return [ClaimVerdict(text="yanıt", status="supported", source_ids=[1], cited=[1])]

    monkeypatch.setattr(pipeline, "verify_answer", fake_verify)

    response = pipeline.run_pipeline("Soru?", mode="verified", k=1, model="qwen2.5:7b-instruct")

    assert response.model == "qwen2.5:7b-instruct"
    assert response.verifier_model == "qwen2.5:7b-instruct"
    assert seen == {"verifier_model": "qwen2.5:7b-instruct"}


def test_llm_stream_emits_chunks_and_final(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "stream_llm_only_text",
        lambda _question, model=None: iter(["Mer", "haba"]),
    )

    events = list(pipeline.run_pipeline_stream("Soru?", mode="llm", k=5, model="gemma4:31b"))

    assert [e["event"] for e in events] == ["step", "chunk", "chunk", "final"]
    assert events[0]["data"]["id"] == "generate"
    assert events[-1]["data"]["answer"] == "Merhaba"
    assert events[-1]["data"]["model"] == "gemma4:31b"


def test_low_confidence_stream_refuses_with_sources(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda _question, k: [make_hit(0.1)])

    events = list(pipeline.run_pipeline_stream("Soru?", mode="rag", k=5))

    assert [e["event"] for e in events] == ["step", "step", "sources", "step", "chunk", "final"]
    assert [e["data"]["id"] for e in events if e["event"] == "step"] == [
        "embed",
        "retrieve",
        "confidence",
    ]
    assert events[2]["data"]["sources"][0]["id"] == "P-1"
    assert "yeterince kapsamıyor" in events[-1]["data"]["answer"]
