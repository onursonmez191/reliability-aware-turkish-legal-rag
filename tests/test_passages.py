"""Sanity tests for the data layer."""

from rag_turkish_law.data.clean import clean_records, normalize_text
from rag_turkish_law.data.passages import record_to_passage, read_jsonl, write_jsonl


def test_normalize_strips_html_and_whitespace():
    assert normalize_text("  <p>Merhaba   <b>dünya</b></p>  ") == "Merhaba dünya"


def test_clean_drops_empty_and_dedupes():
    rows = [
        {"question": "Kira sözleşmesi nedir?", "answer": "A" * 100},
        {"question": "Kira sözleşmesi nedir?", "answer": "B" * 100},  # duplicate question
        {"question": "", "answer": "C" * 100},  # no question
        {"question": "Boşanma nedir?", "answer": "x"},  # too short
        {"question": "Miras nedir?", "answer": "D" * 100},
    ]
    result = clean_records(rows, min_answer_chars=60)
    assert result.stats.kept == 2
    assert result.stats.dropped_duplicate == 1
    assert result.stats.dropped_no_question == 1
    assert result.stats.dropped_too_short == 1


def test_record_to_passage_shape():
    rec = {"question": "Soru?", "answer": "Cevap" * 30, "source_dataset": "X", "record_id": 7}
    p = record_to_passage(rec, 3)
    assert p["passage_id"] == "P-000003"
    assert "Soru: Soru?" in p["text"]
    assert "Açıklama:" in p["text"]
    assert p["snippet"].startswith("Cevap")


def test_jsonl_roundtrip(tmp_path):
    items = [{"a": 1, "txt": "İçerik"}, {"a": 2, "txt": "şğüçöı"}]
    p = tmp_path / "x.jsonl"
    n = write_jsonl(items, p)
    assert n == 2
    back = list(read_jsonl(p))
    assert back == items
