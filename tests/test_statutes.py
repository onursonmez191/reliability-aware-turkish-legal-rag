"""Sanity tests for article-level statute passage building (no network)."""

from rag_turkish_law.data.statutes import (
    _article_id_parts,
    _article_passages_for_law,
    _strip_trailing_heading,
)


def test_strip_trailing_heading_removes_next_article_caption():
    cases = [
        (
            "İrade açıklaması, açık veya örtülü olabilir. 2. İkinci derecedeki noktalar",
            "İrade açıklaması, açık veya örtülü olabilir.",
        ),
        (
            "Sözleşmelerin şekline ilişkin hükümler saklıdır. II. Öneri ve kabul 1. Süreli öneri",
            "Sözleşmelerin şekline ilişkin hükümler saklıdır.",
        ),
        (
            "Öneren, önerisiyle bağlılıktan kurtulur. 2. Süresiz öneri a. Hazır olanlar arasında",
            "Öneren, önerisiyle bağlılıktan kurtulur.",
        ),
    ]
    for raw, expected in cases:
        assert _strip_trailing_heading(raw) == expected


def test_strip_trailing_heading_keeps_clean_text():
    text = "Hayvan bulunduran, hayvanın verdiği zararı gidermekle yükümlüdür."
    assert _strip_trailing_heading(text) == text


def test_article_id_parts_distinguishes_special_articles():
    assert _article_id_parts("MADDE 67") == ("67", "Madde 67")
    assert _article_id_parts("GEÇİCİ MADDE 1") == ("GECICI-1", "Geçici Madde 1")
    assert _article_id_parts("EK MADDE 3") == ("EK-3", "Ek Madde 3")


def test_article_passages_shape_and_filters():
    law = {
        "Kanun Adı": "TÜRK BORÇLAR KANUNU",
        "kanun_numarasi": "6098",
        "url": "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=6098",
        "maddeler": [
            {
                "madde_numarasi": "MADDE 67",
                "text": "Bir hayvanın bakımını üstlenen kişi, onun verdiği zararı gidermekle "
                "yükümlüdür. III. Sorumluluk",
            },
            {"madde_numarasi": "MADDE 68", "text": "kısa"},  # too short → dropped
            {
                "madde_numarasi": "MADDE 69",
                "text": "(Mülga)",  # repealed stub → dropped
            },
        ],
    }
    out = list(_article_passages_for_law(law, min_chars=40))
    assert len(out) == 1
    p = out[0]
    assert p["passage_id"] == "LAW-6098-M67"
    assert p["tag"] == "Kanun Maddesi"
    assert p["law_number"] == "6098"
    assert p["article_number"] == "MADDE 67"
    assert p["title"] == "TÜRK BORÇLAR KANUNU - Madde 67"
    assert p["text"].startswith("Kanun: TÜRK BORÇLAR KANUNU (6098)\nMadde 67:")
    assert "III. Sorumluluk" not in p["text"]
