import json

from scripts.normalize_law_articles import normalize_row
from scripts.scrape_mevzuat import (
    _article_number_label,
    _article_status,
    _candidate_tertips,
    _clean_article_text,
    _clean_article_title,
    _load_text_source_dir,
    _make_passage,
    _select_laws,
    _split_into_articles,
    _source_bytes_to_text,
    _write_passages,
    scrape_law_text_source,
    to_ascii,
)


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
    assert normalized["domain"] == "civil"
    assert normalized["indexable"] is True


def test_normalize_row_preserves_nonempty_text_source_url():
    row = {
        "passage_id": "ART-TAPU-0026",
        "title": "Tapu Kanunu m. 26 — Resmî senet düzenleme",
        "text": "Tapu Kanunu m. 26 — Resmî senet düzenleme\n\nTapu işlemleri resmî senetle yapılır.",
        "snippet": "",
        "source_dataset": "2644 sayılı Tapu Kanunu",
        "source_url": "https://example.test/tapu-kanunu-2644",
        "law_code": "TAPU",
        "law_number": "2644",
        "article_number": 26,
        "article_title": "Resmî senet düzenleme",
        "match_terms": [],
    }

    normalized = normalize_row(row)

    assert normalized["source_url"] == "https://example.test/tapu-kanunu-2644"


def test_domain_selection_includes_enforcement_law():
    laws = _select_laws(None, ["enforcement"])
    numbers = {law["number"] for law in laws}

    assert "2004" in numbers
    assert "6183" in numbers
    assert "6098" not in numbers


def test_unknown_tertip_law_tries_known_candidates():
    law = {"code": "IIK", "number": "2004", "name": "İcra ve İflas Kanunu"}

    assert _candidate_tertips(law) == ["5", "4", "3"]


def test_repealed_article_is_marked_non_indexable():
    law = {
        "code": "ISK",
        "number": "4857",
        "name": "İş Kanunu",
        "domain": "labor",
        "priority": 1,
    }
    passage = _make_passage(law, 95, "Mülga madde", "(Mülga: 20/6/2012-6331/37 md.)")

    assert _article_status(passage["snippet"]) == "repealed"
    assert passage["article_status"] == "repealed"
    assert passage["indexable"] is False


def test_article_number_suffix_keeps_variant_ids_distinct():
    law = {
        "code": "IIK",
        "number": "2004",
        "name": "İcra ve İflas Kanunu",
        "domain": "enforcement",
        "priority": 1,
    }

    plain = _make_passage(law, "Madde 3", "Madde 3", "İcra işleri için genel kural.")
    suffix = _make_passage(law, "MADDE 3 /a", "Madde 3/a", "İcra daireleri başkanlığı düzenlenir.")
    temporary = _make_passage(law, "Geçici Madde 1", "Geçici Madde 1", "Geçiş hükümleri uygulanır.")

    assert plain["passage_id"] == "ART-IIK-0003"
    assert suffix["passage_id"] == "ART-IIK-0003A"
    assert temporary["passage_id"] == "ART-IIK-GECICI-0001"
    assert _article_number_label("EK MADDE 2") == "ek madde 2"


def test_write_passages_keeps_existing_rows_when_no_law_was_successfully_scraped(tmp_path):
    path = tmp_path / "law_articles.jsonl"
    original = {
        "passage_id": "ART-IIK-0085",
        "law_number": "2004",
        "text": "existing",
    }
    path.write_text(json.dumps(original, ensure_ascii=False) + "\n", encoding="utf-8")

    added, duplicates = _write_passages([], path, replace_law_numbers=set())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert added == 0
    assert duplicates == set()
    assert rows == [original]


def test_article_split_accepts_web_marker_without_dash():
    text = (
        "TAPU KANUNU\n"
        "MADDE 1\n"
        "Gayrimenkullerin tescili tapu siciline yapılır.\n"
        "MADDE 2\n"
        "Bu kayıtlar resmî sicil niteliğindedir."
    )

    articles = list(_split_into_articles(text))

    assert [article[0] for article in articles] == [1, 2]
    assert "tapu siciline" in articles[0][2]


def test_html_source_text_preserves_article_markers():
    html = """
    <html><head><style>.x{display:none}</style></head><body>
      <h1>Tapu Kanunu</h1>
      <section><h2>MADDE 1</h2><p>Gayrimenkullerin tescili tapu siciline yapılır.</p></section>
      <section><h2>MADDE 2</h2><p>Bu kayıtlar resmî sicil niteliğindedir.</p></section>
    </body></html>
    """

    text = _source_bytes_to_text(html.encode("utf-8"), source_url="local.html")
    articles = list(_split_into_articles(text))

    assert [article[0] for article in articles] == [1, 2]
    assert "resmî sicil" in articles[1][2]


def test_text_source_dir_discovers_law_number_files(tmp_path):
    source_dir = tmp_path / "law_text"
    source_dir.mkdir()
    (source_dir / "2004.html").write_text("<h2>MADDE 1</h2><p>İcra dairesi.</p>", encoding="utf-8")
    (source_dir / "notes.md").write_text("ignore", encoding="utf-8")

    sources = _load_text_source_dir(source_dir)

    assert sources["2004"]["path"].endswith("2004.html")
    assert "notes" not in sources


def test_scrape_law_text_source_builds_article_passages(tmp_path):
    source = tmp_path / "2644.html"
    source.write_text(
        "<h1>Tapu Kanunu</h1>"
        "<h2>MADDE 26</h2>"
        "<p>Tapu işlemleri resmî senet düzenlenerek yapılır.</p>",
        encoding="utf-8",
    )
    law = {
        "code": "TAPU",
        "number": "2644",
        "name": "Tapu Kanunu",
        "domain": "property",
        "priority": 1,
    }

    result = scrape_law_text_source(
        law,
        {"path": str(source), "source_url": "https://example.test/tapu-kanunu-2644"},
        scraped_at="2026-06-02T00:00:00+00:00",
    )

    assert result.status == "ok"
    assert result.parsed_articles == 1
    assert result.passages[0]["passage_id"] == "ART-TAPU-0026"
    assert result.passages[0]["source_url"] == "https://example.test/tapu-kanunu-2644"
