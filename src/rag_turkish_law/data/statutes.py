"""Article-level statute passages from the public mevzuat.gov.tr dataset.

The OrionCAF QA corpus does not contain base statute article text (e.g. TBK
m. 67), so for known gaps the system relied on hand-written curated patches.
This module instead loads core codes article-by-article from the public
`muhammetakkurt/mevzuat-gov-dataset` (MIT, sourced from mevzuat.gov.tr) and
turns each article into a passage compatible with the QA passages. The FAISS
index can then return the actual legal article instead of a case-by-case patch.

The upstream parser appends the *next* article's side-heading (kenar başlığı)
to the end of each article's text, e.g.

    "...açık veya örtülü olabilir. 2. İkinci derecedeki noktalar"

`_strip_trailing_heading` removes that trailing heading run conservatively.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

from datasets import load_dataset

from ..config import load_config
from .clean import normalize_text

log = logging.getLogger(__name__)

# A single side-heading clause looks like an enumerator (1. / II. / a.) followed
# by Title-case words that carry no sentence punctuation. Several can chain, e.g.
# "II. Öneri ve kabul 1. Süreli öneri". Anchored at the end so it only ever
# trims a trailing heading run, never mid-article text.
_ENUM = r"(?:\d{1,3}|[IVXLCDM]{1,6}|[a-zçğıöşü])\."
_TRAILING_HEADING_RE = re.compile(rf"(?:\s+{_ENUM}\s+[^.!?]+?)+\s*$")

# Repealed / relocated stub articles ("Mülga", "...yerine işlenmiştir.") carry
# no usable legal content.
_REPEALED_RE = re.compile(
    r"(mülga|yürürlükten kaldırıl|yerine işlenmiş)",
    re.IGNORECASE,
)

_MADDE_NUM_RE = re.compile(r"\d+")

DEFAULT_DATASET = "muhammetakkurt/mevzuat-gov-dataset"


def _strip_trailing_heading(text: str) -> str:
    cleaned = _TRAILING_HEADING_RE.sub("", text).strip()
    if cleaned == text:
        return text
    # Only accept the trim if what remains still looks like a complete sentence.
    # This guards against eating real article content that happened to end in an
    # enumerated clause without sentence punctuation.
    if len(cleaned) >= 20 and cleaned[-1] in ".!?…":
        return cleaned
    return text


def _article_id_parts(num_raw: str) -> tuple[str, str]:
    """Return (id_suffix, short_label) for an article number string.

    Handles plain, "GEÇİCİ MADDE" and "EK MADDE" forms so ids stay unique.
    """
    up = num_raw.upper()
    m = _MADDE_NUM_RE.search(num_raw)
    slug = m.group(0) if m else re.sub(r"\s+", "", num_raw) or "X"
    if "GEÇİCİ" in up or "GECICI" in up:
        return f"GECICI-{slug}", f"Geçici Madde {slug}"
    if up.startswith("EK"):
        return f"EK-{slug}", f"Ek Madde {slug}"
    return slug, f"Madde {slug}"


def _article_passages_for_law(law: dict, min_chars: int) -> Iterator[dict]:
    law_name = normalize_text(str(law.get("Kanun Adı", "")))
    law_no = str(law.get("kanun_numarasi", "")).strip()
    url = str(law.get("url", ""))

    for art in law.get("maddeler") or []:
        num_raw = normalize_text(str(art.get("madde_numarasi", "")))
        text = _strip_trailing_heading(normalize_text(str(art.get("text", ""))))
        if len(text) < min_chars:
            continue
        if _REPEALED_RE.search(text) and len(text) < 160:
            continue

        suffix, short_label = _article_id_parts(num_raw)
        yield {
            "passage_id": f"LAW-{law_no}-M{suffix}",
            "text": f"Kanun: {law_name} ({law_no})\n{short_label}: {text}",
            "snippet": text[:400],
            "title": f"{law_name} - {short_label}",
            "source_dataset": f"{law_no} sayılı {law_name}",
            "law_name": law_name,
            "law_number": law_no,
            "article_number": num_raw,
            "source_url": url,
            "tag": "Kanun Maddesi",
        }


def iter_statute_passages() -> Iterator[dict]:
    """Yield article-level passages for the configured statute codes."""
    cfg = load_config()
    st = cfg.data.get("statutes", {})
    if not st.get("enabled", False):
        return

    dataset_name = st.get("dataset_name", DEFAULT_DATASET)
    split = st.get("dataset_split", "train")
    include = {str(x).strip() for x in (st.get("include_law_numbers") or [])}
    min_chars = int(st.get("min_article_chars", 40))

    log.info(
        "Loading statute dataset %s (split=%s, laws=%s)",
        dataset_name,
        split,
        sorted(include) or "ALL",
    )
    ds = load_dataset(dataset_name, split=split)

    seen_ids: set[str] = set()
    for law in ds:
        law_no = str(law.get("kanun_numarasi", "")).strip()
        if include and law_no not in include:
            continue
        for passage in _article_passages_for_law(dict(law), min_chars):
            pid = passage["passage_id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            yield passage


def load_statute_passages() -> list[dict]:
    return list(iter_statute_passages())
