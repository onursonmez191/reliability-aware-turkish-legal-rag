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

    Held-out items are removed from the corpus so retrieval never sees the
    gold passage. Each held-out record records the original passage_id as
    `gold_passage_id` for later Recall@k computation.
    """
    rng = random.Random(seed)
    indices = list(range(len(passages)))
    rng.shuffle(indices)
    held_idx = set(indices[:heldout_size])

    corpus: list[dict] = []
    heldout: list[dict] = []
    for i, p in enumerate(passages):
        if i in held_idx:
            heldout.append(
                {
                    "qid": f"H-{i:06d}",
                    "question": p["title"],
                    "gold_passage_id": p["passage_id"],
                    "expected_verdict": None,
                }
            )
        else:
            corpus.append(p)
    return corpus, heldout
