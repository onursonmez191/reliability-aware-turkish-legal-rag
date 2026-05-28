"""Small deterministic retrieval query expansion for Turkish legal terms."""

from __future__ import annotations

from ..config import load_config


ANIMAL_TRIGGERS = (
    "hayvan",
    "köpek",
    "kopek",
    "kedi",
    "ısır",
    "isir",
    "saldır",
    "saldir",
)

ANIMAL_EXPANSIONS = (
    "hayvan bulunduranın sorumluluğu",
    "hayvanın verdiği zarar tazminat",
    "hayvan bulunduran tazminat",
    "kusursuz sorumluluk hayvan",
)

STRAY_TRIGGERS = (
    "sahipsiz",
    "başıboş",
    "basibos",
    "belediye",
)

STRAY_EXPANSIONS = (
    "sahipsiz hayvan belediye sorumluluğu",
)


def _norm(text: str) -> str:
    return text.casefold()


def expand_retrieval_queries(query: str) -> list[str]:
    cfg = load_config()
    expansion_cfg = cfg.retrieval.get("query_expansion", {})
    if not expansion_cfg.get("enabled", True):
        return [query]

    normalized = _norm(query)
    queries = [query]
    if any(term in normalized for term in ANIMAL_TRIGGERS):
        queries.extend(ANIMAL_EXPANSIONS)
    if any(term in normalized for term in STRAY_TRIGGERS):
        queries.extend(STRAY_EXPANSIONS)

    seen: set[str] = set()
    unique: list[str] = []
    for item in queries:
        key = _norm(item).strip()
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique
