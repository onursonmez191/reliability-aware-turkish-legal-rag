"""Load the Turkish legal QA dataset from Hugging Face."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

from datasets import load_dataset

from ..config import load_config

log = logging.getLogger(__name__)


def iter_raw_records(
    dataset_name: str | None = None,
    split: str | None = None,
) -> Iterator[dict]:
    cfg = load_config()
    name = dataset_name or cfg.data.dataset_name
    split_name = split or cfg.data.dataset_split

    log.info("Loading dataset %s (split=%s) at %s", name, split_name, datetime.utcnow().isoformat())
    ds = load_dataset(name, split=split_name)
    for i, row in enumerate(ds):
        row.setdefault("_row_index", i)
        row.setdefault("_source_dataset", name)
        yield dict(row)


def dataset_size(dataset_name: str | None = None, split: str | None = None) -> int:
    cfg = load_config()
    name = dataset_name or cfg.data.dataset_name
    split_name = split or cfg.data.dataset_split
    ds = load_dataset(name, split=split_name)
    return len(ds)
