import json

from scripts.build_index import _load_curated_passages


def test_load_curated_passages_skips_non_indexable(tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    rows = [
        {"passage_id": "ART-TEST-0001", "text": "active", "indexable": True},
        {"passage_id": "ART-TEST-0002", "text": "repealed", "indexable": False},
    ]
    with (curated / "law_articles.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    passages, skipped = _load_curated_passages(curated)

    assert [row["passage_id"] for row in passages] == ["ART-TEST-0001"]
    assert skipped == 1
