"""Split passages into corpus and held-out evaluation seed."""

from __future__ import annotations

import random
from typing import Sequence


def make_heldout(
    passages: Sequence[dict],
    heldout_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return (corpus_passages, heldout_records).

    Held-out items stay in the corpus — Recall@k measures whether the
    retriever can find the gold passage when given the question, so the
    gold must be present in the index. Each held-out record records the
    original passage_id as `gold_passage_id` for evaluation.
    """
    rng = random.Random(seed)
    indices = list(range(len(passages)))
    rng.shuffle(indices)
    held_idx = indices[:heldout_size]

    heldout: list[dict] = []
    for i in held_idx:
        p = passages[i]
        heldout.append(
            {
                "qid": f"H-{i:06d}",
                "question": p["title"],
                "gold_passage_id": p["passage_id"],
                "expected_verdict": None,
            }
        )
    return list(passages), heldout
