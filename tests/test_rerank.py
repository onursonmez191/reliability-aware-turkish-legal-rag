from rag_turkish_law.retrieval import rerank as rerank_module
from rag_turkish_law.retrieval.search import RetrievedPassage


def _hit(pid: str, score: float, title: str) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=pid,
        text=title,
        snippet=title,
        title=title,
        tag="",
        source_dataset="test",
        score=score,
    )


class _FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, _pairs):
        return self._scores


def test_rerank_blends_model_score_with_original_retrieval_score(monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "_get_reranker",
        lambda _name: _FakeReranker([0.0009, 0.22]),
    )
    passages = [
        _hit("ART-TBK-0067", 0.93, "Hayvan bulunduranın sorumluluğu"),
        _hit("ART-TBK-0377", 0.89, "Genel sorumluluk"),
    ]

    out = rerank_module.rerank("Köpek ısırdı, kimden tazminat isterim?", passages, keep_top=2, force=True)

    assert out[0].passage_id == "ART-TBK-0067"
    assert all(0.0 <= hit.score <= 1.0 for hit in out)
    assert out[0].score > 0.7


def test_rerank_can_promote_model_hit_when_top_retrieval_is_not_confident(monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "_get_reranker",
        lambda _name: _FakeReranker([0.0, 1.0]),
    )
    passages = [
        _hit("WEAK-LEXICAL", 0.86, "Lexical match"),
        _hit("BETTER-CONTEXT", 0.85, "Better contextual match"),
    ]

    out = rerank_module.rerank("Soru?", passages, keep_top=2, force=True)

    assert out[0].passage_id == "BETTER-CONTEXT"
