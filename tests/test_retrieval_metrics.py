from types import SimpleNamespace

from rag_turkish_law.evaluation import ablation
from rag_turkish_law.evaluation.retrieval_metrics import evaluate_retrieval
from rag_turkish_law.retrieval.confidence import assess_retrieval_confidence
from rag_turkish_law.retrieval.search import RetrievedPassage


def hit(pid: str, score: float = 0.9) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=pid,
        text=f"text {pid}",
        snippet=f"snippet {pid}",
        title=f"title {pid}",
        tag="",
        source_dataset="test",
        score=score,
    )


def test_evaluate_retrieval_applies_reranker():
    items = [{"qid": "Q1", "question": "Soru?", "gold_passage_id": "gold"}]

    def retriever(_question: str, _k: int):
        return [hit("bad"), hit("gold")]

    def reranker(_question: str, hits, keep_top: int):
        return [hits[1], hits[0]][:keep_top]

    off = evaluate_retrieval(items, k=1, retriever=retriever)
    on = evaluate_retrieval(
        items,
        k=1,
        use_rerank=True,
        candidate_k=2,
        retriever=retriever,
        reranker=reranker,
    )

    assert off["recall@1"] == 0.0
    assert on["recall@1"] == 1.0
    assert on["mrr"] == 1.0


def test_rerank_ablation_passes_real_rerank_options(monkeypatch):
    calls = []

    def fake_eval(_items, **kwargs):
        calls.append(kwargs)
        return {"kwargs": kwargs}

    cfg = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_k=5,
            rerank={"candidate_k": 11},
        )
    )
    monkeypatch.setattr(ablation, "load_config", lambda: cfg)
    monkeypatch.setattr(ablation, "evaluate_retrieval", fake_eval)

    result = ablation.run_rerank_ablation([{"question": "Q?", "gold_passage_id": "P"}])

    assert result["results"]["off"]["kwargs"] == {"k": 5}
    assert result["results"]["on"]["kwargs"] == {
        "k": 5,
        "use_rerank": True,
        "candidate_k": 11,
    }


def test_retrieval_confidence_labels_low_scores():
    low = assess_retrieval_confidence(
        [hit("a", 0.2), hit("b", 0.1), hit("c", 0.1)],
        min_top_score=0.75,
        min_avg_top3_score=0.70,
    )
    high = assess_retrieval_confidence(
        [hit("a", 0.9), hit("b", 0.8), hit("c", 0.8)],
        min_top_score=0.75,
        min_avg_top3_score=0.70,
    )

    assert low.label == "low"
    assert high.label == "high"
