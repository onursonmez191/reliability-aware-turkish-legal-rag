import json
from pathlib import Path

from rag_turkish_law.evaluation.eval_set import build_eval_set, load_manual_eval


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manual_eval_loader_normalizes_rows(tmp_path):
    manual_path = tmp_path / "manual.jsonl"
    write_jsonl(
        manual_path,
        [
            {
                "qid": "M-1",
                "question": "Soru?",
                "type": "risk",
                "expected_verdict": "risk",
                "note": "note",
            }
        ],
    )

    rows = load_manual_eval(manual_path)

    assert rows == [
        {
            "qid": "M-1",
            "question": "Soru?",
            "type": "risk",
            "gold_passage_id": None,
            "expected_verdict": "risk",
            "note": "note",
            "source": "manual",
        }
    ]


def test_combined_eval_set_contains_heldout_and_manual(tmp_path):
    heldout_path = tmp_path / "heldout.jsonl"
    manual_path = tmp_path / "manual.jsonl"
    write_jsonl(heldout_path, [{"qid": "H-1", "question": "Heldout?", "gold_passage_id": "P-1"}])
    write_jsonl(manual_path, [{"qid": "M-1", "question": "Manual?", "expected_verdict": "risk"}])

    rows = build_eval_set(heldout_path, manual_path, eval_set="combined")

    assert [row["qid"] for row in rows] == ["H-1", "M-1"]
    assert rows[0]["source"] == "heldout"
    assert rows[1]["source"] == "manual"
