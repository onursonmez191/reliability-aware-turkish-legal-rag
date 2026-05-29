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

TENANCY_TRIGGERS = (
    "kira",
    "kiracı",
    "kiraci",
    "kiralanan",
    "kiraya veren",
    "ev sahibi",
)

TENANCY_EARLY_RETURN_TRIGGERS = (
    "süresi dolmadan",
    "suresi dolmadan",
    "sözleşme bitmeden",
    "sozlesme bitmeden",
    "bitmeden",
    "erken çık",
    "erken cik",
    "çıkabilir",
    "cikabilir",
    "çıkmak",
    "cikmak",
    "boşalt",
    "bosalt",
    "tahliye",
)

TENANCY_EARLY_RETURN_EXPANSIONS = (
    "kiracı fesih dönemine uymaksızın kiralananı geri verdiğinde borçları makul süre devam eder",
    "kiracı sözleşme süresine uymaksızın kiralananı geri verirse makul süre kira borcu",
    "kiracının kabul edilebilir yeni kiracı bulması kira borçları sona erer",
    "Türk Borçlar Kanunu madde 325 erken tahliye makul süre yeni kiracı",
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
    "vefat",
    "ölen",
    "olen",
    "murisin",
    "muris",
    "annemizin",
    "babamızın",
    "babamizin",
)

INHERITANCE_EXPANSIONS = (
    "miras hukuku mirasçı hakları",
    "mirasçılık belgesi veraset ilamı",
    "miras kalan taşınmaz tapu intikali",
    "tapu intikali miras",
    "elbirliği mülkiyeti miras ortaklığı",
    "miras ortaklığının giderilmesi ortaklığın giderilmesi izale-i şuyu",
    "miras payı taşınmaz paylaşımı",
    "tenkis davası saklı pay",
    "vasiyet iptali miras kanunu",
    "türk medeni kanunu miras",
)

INHERITANCE_PROPERTY_TRIGGERS = (
    "tapu",
    "tapusunu",
    "tapuyu",
    "intikal",
    "devretmiyor",
    "devretmeme",
    "devir",
    "dairenin",
    "daire",
    "taşınmaz",
    "tasinmaz",
    "ortaklığın giderilmesi",
    "ortakligin giderilmesi",
    "izale-i şuyu",
    "izalei şuyu",
    "izale-i suyu",
    "izalei suyu",
)

FAMILY_INHERITANCE_TRIGGERS = (
    "kardeş",
    "kardes",
    "anne",
    "ana",
    "baba",
    "ebeveyn",
    "annem",
    "babam",
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
    has_tenancy_signal = any(term in normalized for term in TENANCY_TRIGGERS)
    has_early_return_signal = any(term in normalized for term in TENANCY_EARLY_RETURN_TRIGGERS)
    if has_tenancy_signal and has_early_return_signal:
        queries.extend(TENANCY_EARLY_RETURN_EXPANSIONS)
    if any(term in normalized for term in LABOR_TRIGGERS):
        queries.extend(LABOR_EXPANSIONS)

    has_inheritance_signal = any(term in normalized for term in INHERITANCE_TRIGGERS)
    has_property_signal = any(term in normalized for term in INHERITANCE_PROPERTY_TRIGGERS)
    has_family_signal = any(term in normalized for term in FAMILY_INHERITANCE_TRIGGERS)
    if has_inheritance_signal or (has_property_signal and has_family_signal):
        queries.extend(INHERITANCE_EXPANSIONS)

    seen: set[str] = set()
    unique: list[str] = []
    for item in queries:
        key = _norm(item).strip()
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique
