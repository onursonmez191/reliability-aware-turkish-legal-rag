import json

from scripts import import_hf_statutes


def test_hf_import_writes_selected_law_with_suffix_article(monkeypatch, tmp_path):
    dataset_rows = [
        {
            "Kanun Adı": "İCRA VE İFLAS KANUNU",
            "kanun_numarasi": "2004",
            "url": "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=2004",
            "maddeler": [
                {"madde_numarasi": "Madde 3", "text": "İcra ve iflas işleri bir dairede birleştirilebilir."},
                {"madde_numarasi": "MADDE 3 /a", "text": "İcra daireleri başkanlığı kurulabilir."},
            ],
        },
        {
            "Kanun Adı": "Başka Kanun",
            "kanun_numarasi": "9999",
            "url": "",
            "maddeler": [{"madde_numarasi": "Madde 1", "text": "Dışarıda kalmalı."}],
        },
    ]
    monkeypatch.setattr(import_hf_statutes, "load_dataset", lambda *args, **kwargs: dataset_rows)

    output = tmp_path / "law_articles.jsonl"
    report = tmp_path / "report.json"
    result = import_hf_statutes.import_laws(
        law_numbers={"2004"},
        output_path=output,
        report_path=report,
        dataset_name="dummy",
        split="train",
        replace_laws=True,
        min_chars=20,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result["found_laws"] == ["2004"]
    assert [row["passage_id"] for row in rows] == ["ART-IIK-0003", "ART-IIK-0003A"]
    assert rows[1]["source_sha256"] == "hf:dummy"
