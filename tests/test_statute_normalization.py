from scripts.normalize_law_articles import normalize_row
from scripts.scrape_mevzuat import _clean_article_text, _clean_article_title, to_ascii


def test_article_text_strips_appended_next_heading():
    raw = (
        "Sözleşme, tarafların iradelerini karşılıklı olarak açıklamalarıyla kurulur.\n"
        "2. İkinci derecedeki noktalar"
    )

    assert _clean_article_text(raw) == (
        "Sözleşme, tarafların iradelerini karşılıklı olarak açıklamalarıyla kurulur."
    )


def test_article_text_strips_structural_heading_block():
    raw = (
        "Diğer hukuki sorumluluk sigortalarına ilişkin kanun hükümleri saklıdır. "
        "ÜÇÜNCÜ BÖLÜM Borçların ve Borç İlişkilerinin Sona Ermesi "
        "BİRİNCİ AYIRIM Sona Erme Hâlleri A. Asıl borca bağlı hak ve borçların sona ermesi"
    )

    assert _clean_article_text(raw) == (
        "Diğer hukuki sorumluluk sigortalarına ilişkin kanun hükümleri saklıdır."
    )


def test_article_text_strips_heading_after_pdf_footnote_marker():
    raw = (
        "Türkiye temsilcilerine nüfus memurluğu yetkisi verilebilir.2 "
        "III. Sorumluluk"
    )

    assert _clean_article_text(raw) == (
        "Türkiye temsilcilerine nüfus memurluğu yetkisi verilebilir."
    )


def test_article_title_strips_pdf_footnote_number():
    assert (
        _clean_article_title("2. Kiralananın sözleşmenin bitiminden önce geri verilmesi2")
        == "2. Kiralananın sözleşmenin bitiminden önce geri verilmesi"
    )


def test_turkish_ascii_normalization_removes_combining_dot():
    assert to_ascii("İrade açıklaması") == "Irade aciklamasi"


def test_normalize_row_rebuilds_title_body_and_match_terms():
    row = {
        "passage_id": "ART-TBK-0325",
        "title": "Türk Borçlar Kanunu m. 325 — 2. Kiralananın sözleşmenin bitiminden önce geri verilmesi2",
        "text": (
            "Türk Borçlar Kanunu m. 325 — 2. Kiralananın sözleşmenin bitiminden önce geri verilmesi2\n\n"
            "Kiracı, sözleşme süresine uymaksızın kiralananı geri verdiği takdirde borçları devam eder. "
            "V. Takastan feragat yasağı"
        ),
        "snippet": "",
        "source_dataset": "6098 sayılı Türk Borçlar Kanunu",
        "source_url": "",
        "law_code": "TBK",
        "law_number": "6098",
        "article_number": 325,
        "article_title": "2. Kiralananın sözleşmenin bitiminden önce geri verilmesi2",
        "match_terms": [],
    }

    normalized = normalize_row(row)

    assert normalized["article_title"].endswith("geri verilmesi")
    assert "verilmesi2" not in normalized["title"]
    assert normalized["text"].endswith("borçları devam eder.")
    assert "i̇" not in " ".join(normalized["match_terms"])
