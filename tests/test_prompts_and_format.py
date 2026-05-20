"""Prompt rendering + citation extraction sanity tests."""

from rag_turkish_law.generation.format import find_citations, map_citations_to_ids
from rag_turkish_law.generation.prompts import build_grounded_messages, build_llm_only_messages


def test_grounded_prompt_includes_sources_and_rules():
    passages = [
        {"title": "Kira erken fesih", "snippet": "Kiracı, makul süre vererek tahliye edebilir."},
        {"title": "Tazminat", "snippet": "Yeni kiracı bulunana dek kira sorumluluğu."},
    ]
    msgs = build_grounded_messages("Erken çıkış mümkün mü?", passages)
    assert msgs[0]["role"] == "system"
    assert "köşeli parantez" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1]" in user and "[2]" in user
    assert "Kira erken fesih" in user
    assert "Erken çıkış" in user


def test_llm_only_prompt_has_no_sources_block():
    msgs = build_llm_only_messages("Soru?")
    assert msgs[0]["role"] == "system"
    assert "KAYNAKLAR" not in msgs[1]["content"]


def test_find_citations_dedupes_and_sorts():
    text = "Bu doğrudur [2]. Bu da [1][2]. Şu da [3]."
    assert find_citations(text) == [1, 2, 3]


def test_map_citations_to_ids():
    passages = [{"passage_id": "P-A"}, {"passage_id": "P-B"}, {"passage_id": "P-C"}]
    text = "Açıklama [1] ve [3]."
    mapping = map_citations_to_ids(text, passages)
    assert mapping == {1: "P-A", 3: "P-C"}
