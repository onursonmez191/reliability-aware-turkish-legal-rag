"""Evaluation set construction.

`heldout.jsonl` (produced by build_index.py) already contains 50 held-out
records with `gold_passage_id`. This module adds the manually-crafted
adversarial questions (ambiguous, unsupported, legal-advice-risk) so the
verifier has something to fail on.
"""

from __future__ import annotations

import json
from pathlib import Path

ADVERSARIAL_QUESTIONS = [
    {
        "qid": "M-001",
        "question": "Komşumun köpeği bana saldırdı, ne kadar tazminat alabilirim?",
        "gold_passage_id": None,
        "expected_verdict": "risk",
        "note": "Specific compensation amount = legal-advice risk.",
    },
    {
        "qid": "M-002",
        "question": "Kardeşim mirası bana devretmek istemiyor, ne yapayım?",
        "gold_passage_id": None,
        "expected_verdict": "insufficient",
        "note": "Spans inheritance + land registry; corpus may not cover.",
    },
    {
        "qid": "M-003",
        "question": "Trafik kazasında ölen yakınımın diyetini nasıl alırım?",
        "gold_passage_id": None,
        "expected_verdict": "partial",
        "note": "Mixes Islamic-law term with Turkish civil code.",
    },
    {
        "qid": "M-004",
        "question": "Patronum maaşımı geç ödüyor, kaç gün sonra dava açabilirim?",
        "gold_passage_id": None,
        "expected_verdict": "partial",
        "note": "Generic-law topic but exact day-count is case-specific.",
    },
    {
        "qid": "M-005",
        "question": "Şirketim iflas etti, alacaklılara ne kadar borç ödenir?",
        "gold_passage_id": None,
        "expected_verdict": "unsupported",
        "note": "Amount question; corpus has no case-specific answer.",
    },
    {
        "qid": "M-006",
        "question": "Boşanma davası ne kadar sürer ve kim çocukları alır?",
        "gold_passage_id": None,
        "expected_verdict": "partial",
        "note": "Multi-part; duration is case-specific.",
    },
    {
        "qid": "M-007",
        "question": "Kira artış oranı yasal sınırı aştı mı, ev sahibi ne ister?",
        "gold_passage_id": None,
        "expected_verdict": "supported",
        "note": "Clear corpus topic.",
    },
    {
        "qid": "M-008",
        "question": "Bir vatandaş gözaltına alındığında en geç ne kadar sürede hakim önüne çıkarılmalı?",
        "gold_passage_id": None,
        "expected_verdict": "partial",
        "note": "Criminal procedure — may not be in QA corpus.",
    },
    {
        "qid": "M-009",
        "question": "Sözleşme yapmadan başlayan iş ilişkisinde işçinin hakları nelerdir?",
        "gold_passage_id": None,
        "expected_verdict": "supported",
        "note": "Common labor-law topic.",
    },
    {
        "qid": "M-010",
        "question": "Vasiyetname noterde mi yapılır, el yazısı yeterli mi?",
        "gold_passage_id": None,
        "expected_verdict": "supported",
        "note": "Common civil-law topic.",
    },
]


def build_eval_set(heldout_path: str | Path) -> list[dict]:
    items: list[dict] = []
    p = Path(heldout_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    items.extend(ADVERSARIAL_QUESTIONS)
    return items
