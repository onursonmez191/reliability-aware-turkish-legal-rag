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

LABOR_TRIGGERS = (
    "iş sözleşmesi",
    "is sozlesmesi",
    "işçi",
    "isci",
    "işveren",
    "isveren",
    "fesih",
    "feshi",
    "kıdem",
    "kidem",
    "ihbar",
    "haksız çıkarma",
    "haksiz cikarma",
    "işten çıkarıldım",
    "isten cikarildim",
    "fazla mesai",
    "yıllık izin",
    "yillik izin",
    "deneme süresi",
)

LABOR_EXPANSIONS = (
    "iş sözleşmesi feshi bildirim süresi",
    "belirsiz süreli iş sözleşmesi fesih",
    "işçi hakları iş kanunu 4857",
    "iş kanunu fesih geçerli neden",
    "iş güvencesi kıdem tazminatı",
)

INHERITANCE_TRIGGERS = (
    "miras",
    "veraset",
    "tenkis",
    "vasiyet",
    "tereke",
    "mirasçı",
    "mirasci",
)

INHERITANCE_EXPANSIONS = (
    "miras hukuku mirasçı hakları",
    "tenkis davası saklı pay",
    "vasiyet iptali miras kanunu",
    "türk medeni kanunu miras",
)


_TR_ASCII = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")


def _norm(text: str) -> str:
    """Casefold + ASCII-fold Turkish diacritics for reliable trigger matching.

    Python's casefold() maps İ (U+0130) → i + U+0307 (combining dot), so
    'İş' != 'iş' after casefold alone. Translating Turkish chars to ASCII
    first avoids this mismatch.
    """
    return text.translate(_TR_ASCII).casefold()


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
    if any(term in normalized for term in LABOR_TRIGGERS):
        queries.extend(LABOR_EXPANSIONS)
    if any(term in normalized for term in INHERITANCE_TRIGGERS):
        queries.extend(INHERITANCE_EXPANSIONS)

    seen: set[str] = set()
    unique: list[str] = []
    for item in queries:
        key = _norm(item).strip()
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique
