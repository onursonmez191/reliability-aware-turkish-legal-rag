"""
Download Turkish law PDFs from mevzuat.gov.tr and extract article-level passages.

Each article becomes a passage in data/curated/law_articles.jsonl, enabling
direct article lookup via both keyword matching and FAISS vector search.

After scraping, rebuild the FAISS index:
    py -3 scripts/build_index.py

Usage:
    py -3 scripts/scrape_mevzuat.py                       # scrape all predefined laws
    py -3 scripts/scrape_mevzuat.py --laws 6098 4721      # specific law numbers
    py -3 scripts/scrape_mevzuat.py --dry-run             # preview without writing
    py -3 scripts/scrape_mevzuat.py --delay 2.0           # set request delay (seconds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterator

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not installed. Run: pip install pymupdf")

log = logging.getLogger("scrape_mevzuat")

# ---------------------------------------------------------------------------
# Law catalog — (code, law_number, law_name, tertip)
# Tertip 5 = 1983-present, Tertip 4 = 1964-1982, Tertip 3 = 1943-1963
# ---------------------------------------------------------------------------
LAWS: list[dict] = [
    {"code": "TBK",    "number": "6098", "name": "Türk Borçlar Kanunu",                              "tertip": "5"},
    {"code": "TMK",    "number": "4721", "name": "Türk Medeni Kanunu",                               "tertip": "5"},
    {"code": "TCK",    "number": "5237", "name": "Türk Ceza Kanunu",                                 "tertip": "5"},
    {"code": "ISK",    "number": "4857", "name": "İş Kanunu",                                        "tertip": "5"},
    {"code": "HMK",    "number": "6100", "name": "Hukuk Muhakemeleri Kanunu",                        "tertip": "5"},
    {"code": "TTK",    "number": "6102", "name": "Türk Ticaret Kanunu",                              "tertip": "5"},
    {"code": "CMK",    "number": "5271", "name": "Ceza Muhakemesi Kanunu",                           "tertip": "5"},
    {"code": "SSGSSK", "number": "5510", "name": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu","tertip": "5"},
    {"code": "KMK",    "number": "634",  "name": "Kat Mülkiyeti Kanunu",                             "tertip": "5"},
    {"code": "IYUK",   "number": "2577", "name": "İdari Yargılama Usulü Kanunu",                     "tertip": "5"},
    {"code": "KVK",    "number": "5520", "name": "Kurumlar Vergisi Kanunu",                          "tertip": "5"},
    {"code": "KDVK",   "number": "3065", "name": "Katma Değer Vergisi Kanunu",                       "tertip": "5"},
    {"code": "AY",     "number": "2709", "name": "Türkiye Cumhuriyeti Anayasası",                    "tertip": "5"},
    {"code": "VUK",    "number": "213",  "name": "Vergi Usul Kanunu",                                "tertip": "4"},
    {"code": "GVK",    "number": "193",  "name": "Gelir Vergisi Kanunu",                             "tertip": "4"},
    {"code": "BK",     "number": "818",  "name": "Borçlar Kanunu (Eski)",                            "tertip": "3"},
]

MEVZUAT_PDF_URL = "https://www.mevzuat.gov.tr/MevzuatMetin/1.{tertip}.{number}.pdf"

# Regex for article markers: "MADDE 49-" / "Madde 49-" / "Madde 49 -"
# Laws vary: TBK uses all-caps, TMK/TCK/ISK use title-case.
MADDE_RE = re.compile(r"^MADDE\s+(\d+)\s*[-–]", re.MULTILINE | re.IGNORECASE)

# Lines that look like section/subsection headings (not article text)
HEADING_RE = re.compile(
    r"^(?:"
    r"[A-ZÇĞİÖŞÜ]\.\s"                    # A. B. C.
    r"|[IVX]+\.\s"                         # I. II. III. IV.
    r"|\d+\.\s"                            # 1. 2. 3.
    r"|[a-zçğışöü]\)\s"                    # a) b) c)
    r"|[a-zçğışöü]\.\s"                    # a. b. c.
    r"|(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|"
    r"ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s"  # ordinals
    r"|KISIM|BÖLÜM|AYIRIM|MADDE|EK\s"     # structural keywords
    r")"
)

# ---------------------------------------------------------------------------
# Turkish ↔ ASCII normalisation helpers
# ---------------------------------------------------------------------------
_TR_ASCII = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")  # ı→i, İ→I


def to_ascii(text: str) -> str:
    return text.translate(_TR_ASCII)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------
def _pdf_url(law: dict) -> str:
    return MEVZUAT_PDF_URL.format(tertip=law["tertip"], number=law["number"])


def download_pdf(law: dict, delay: float = 1.5) -> bytes:
    url = _pdf_url(law)
    log.info("Downloading %s (%s) from %s", law["code"], law["name"], url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TurkishLegalRAG/1.0; "
            "research use; +https://github.com)"
        ),
        "Accept": "application/pdf",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    time.sleep(delay)
    return data


# ---------------------------------------------------------------------------
# PDF text extraction (character-level to bypass font encoding issues)
# ---------------------------------------------------------------------------
def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        doc = fitz.open(str(tmp_path))
        page_texts: list[str] = []

        for page in doc:
            rawdict = page.get_text("rawdict")
            lines: list[str] = []
            for block in rawdict.get("blocks", []):
                for line in block.get("lines", []):
                    chars: list[str] = []
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            c = ch.get("c", "")
                            if c:
                                chars.append(c)
                    text = "".join(chars).strip()
                    if text:
                        lines.append(text)
            page_texts.append("\n".join(lines))

        doc.close()
        return "\n".join(page_texts)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------
def _split_into_articles(full_text: str) -> Iterator[tuple[int, str, str]]:
    """Yield (article_number, title, article_text) from extracted PDF text."""
    segments = MADDE_RE.split(full_text)
    # segments[0] = preamble (law header)
    # segments[1] = first article number, segments[2] = first article text, ...
    # Pattern: [preamble, num1, text1, num2, text2, ...]
    if len(segments) < 3:
        return

    lines_before_madde: list[str] = segments[0].splitlines()

    for i in range(1, len(segments) - 1, 2):
        art_num = int(segments[i])
        art_raw = segments[i + 1].strip() if i + 1 < len(segments) else ""

        # Remove the (continued on next page) fragments — short dangling tokens
        # Article text ends at the next MADDE marker (already split out) or EOF
        art_text = _clean_article_text(art_raw)

        # Determine the title: last meaningful heading before this MADDE.
        # Fallback: use the first sentence of the article text itself so that
        # laws without separate heading lines (e.g. ISK) still get useful titles.
        title = _find_preceding_title(lines_before_madde)
        if not title or title.lower().startswith("madde"):
            title = _title_from_text(art_text)

        yield art_num, title, art_text

        # Update lines_before_madde for next iteration
        lines_before_madde = art_raw.splitlines()


def _title_from_text(art_text: str, max_chars: int = 90) -> str:
    """Extract a title from the first sentence of article text (fallback)."""
    if not art_text:
        return ""
    # Take up to the first period, semicolon, or max_chars
    end = len(art_text)
    for sep in (".", ";", "\n"):
        pos = art_text.find(sep)
        if 0 < pos < end:
            end = pos
    return art_text[:min(end, max_chars)].strip()


_TRAILING_HEADING_RE = re.compile(
    r"(?:\s+(?:\d{1,3}|[IVXLCDM]{1,6}|[a-zçğıöşü])\.\s+[^.!?]+?)+\s*$"
)


def _strip_trailing_heading(text: str) -> str:
    """Remove next-article side-heading fragments appended to article text.

    The PDF parser often attaches the next article's section heading to the
    end of the current article, e.g. '...açık veya örtülü olabilir.
    2. İkinci derecedeki noktalar'. This strips that trailing run.
    """
    cleaned = _TRAILING_HEADING_RE.sub("", text).strip()
    if cleaned == text:
        return text
    if len(cleaned) >= 20 and cleaned[-1] in ".!?…":
        return cleaned
    return text


def _clean_article_text(raw: str) -> str:
    """Remove page-break artefacts, normalise whitespace, strip heading noise."""
    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and single-digit page numbers
        if not stripped or re.fullmatch(r"\d{1,4}", stripped):
            continue
        cleaned.append(stripped)
    return _strip_trailing_heading(" ".join(cleaned))


_TRIVIAL_HEADING = re.compile(
    r"^(?:[IVX]+\.|[A-ZÇĞİÖŞÜ]\.|[a-zçğışöü][.)]\s*\d*|"
    r"\d+\.\s*(?:Genel olarak|Alt başlık|Kapsam|Tanım))\s*$",
    re.IGNORECASE,
)


def _find_preceding_title(lines: list[str]) -> str:
    """Return the most descriptive heading before an article marker.

    Prefers a substantive section heading (e.g. 'Haksız Fiiller') over a
    generic structural label (e.g. 'I. Genel olarak'). Scans up to 20 lines
    back, collecting up to 2 candidate headings, and returns the longest one.
    """
    candidates: list[str] = []
    for line in reversed(lines[-20:]):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip KISIM / BÖLÜM / AYIRIM structural lines and any "MADDE X" references
        # (those come from the previous article's text, not structural headings)
        if re.match(
            r"^(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|"
            r"ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU|"
            r"ONBİRİNCİ|ONİKİNCİ|KISIM|BÖLÜM|AYIRIM|MADDE\s+\d)",
            stripped,
        ):
            continue
        # Capture heading-like lines (letter/numeral prefix + content)
        if HEADING_RE.match(stripped) and 4 < len(stripped) <= 100:
            # Remove the heading prefix (e.g. "B. " or "I. ") to get the label
            label = re.sub(r"^[A-ZÇĞİÖŞÜa-zçğışöüIVX]+[.)]\s*", "", stripped).strip()
            if len(label) >= 4:
                candidates.append(label)
                if len(candidates) >= 2:
                    break
    if not candidates:
        return ""
    # Return the longest / most descriptive candidate
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Match-term generation
# Generic legal/structural words that appear in almost every article and
# would match too broadly if added to match_terms.
_STOP_WORDS: frozenset[str] = frozenset({
    "madde", "kisim", "bolum", "ayirim", "kanun", "genel", "olarak",
    "hukum", "kural", "esas", "diger", "veya", "ile", "icin", "olan",
    "hukuki", "taraf", "sure", "kapsam", "tanimi", "tanim", "islem",
    "durum", "hali", "sekil", "yolu", "usul", "sure",
    # Generic qualifying words that match too broadly across articles
    "kosullar", "kosullari", "sartlar", "sartlari", "haller", "halleri",
    "iliskin", "hakkinda", "itiraz", "istisnai", "kapsaminda",
    "hususlar", "esaslar", "hukumler", "hukumleri",
})

# Common Turkish inflectional suffixes — strip these to get a usable stem so
# "feshinden" → "fesih", "sözleşmelerinin" → "sözleşme" etc. match queries.
# Ordered longest-first so greedy matching works correctly.
_SUFFIXES = (
    "lerinden", "larından", "lerinin", "larının", "lerine", "larına",
    "lerinde", "larında", "leriyle", "larıyla",
    "lerin", "larin", "lere", "lara", "lerde", "larda",
    "inden", "indan", "ından", "inden",
    "nden", "ndan", "nden",
    "inde", "inde", "ında", "ünde",
    "iyle", "ıyla", "üyle",
    "inin", "ının", "unun", "ünün",
    "nin", "nın", "nun", "nün",
    "den", "dan", "ten", "tan",
    "ine", "ına", "une", "üne",
    "ile", "ila",
    "de", "da", "te", "ta",
    "in", "ın", "un", "ün",
    "ye", "ya",
    "i", "ı", "u", "ü",
    "e", "a",
)


def _stems(word: str) -> list[str]:
    """Return the word plus any stripped-suffix stems (≥ 4 chars)."""
    results = [word]
    for suf in _SUFFIXES:
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if len(stem) >= 4 and stem not in results:
                results.append(stem)
                break  # one stem per word is enough
    return results


# ---------------------------------------------------------------------------
def _make_match_terms(law: dict, art_num: int, title: str) -> list[str]:
    code = law["code"].lower()
    code_ascii = to_ascii(code)
    num = str(art_num)

    terms: list[str] = []

    # Law code + article number variants
    for prefix in [code, code_ascii]:
        terms += [
            f"{prefix} {num}",
            f"{prefix} m.{num}",
            f"{prefix} m. {num}",
            f"{prefix} madde {num}",
        ]

    # Law number + article reference (e.g. "6098 m.49")
    terms += [
        f"{law['number']} m.{num}",
        f"{law['number']} m. {num}",
        f"{law['number']} madde {num}",
        f"madde {num}",  # generic — useful when law is clear from context
    ]

    # Key words from the article title (skip structural/stop words)
    if title:
        title_norm = _norm(title)
        title_ascii = to_ascii(title_norm)
        words = [
            w for w in re.findall(r"[a-zçğışöüÀ-ɏ]{4,}", title_norm)
            if w not in _STOP_WORDS and to_ascii(w) not in _STOP_WORDS
        ]
        words_ascii = [
            w for w in re.findall(r"[a-z]{4,}", title_ascii)
            if w not in _STOP_WORDS
        ]
        # Multi-word phrase from title (up to 3 content words)
        content_words = [w for w in title_norm.split() if len(w) >= 4
                         and w not in _STOP_WORDS and to_ascii(w) not in _STOP_WORDS]
        if len(content_words) >= 2:
            terms.append(" ".join(content_words[:3]))
        terms.extend(words[:8])
        terms.extend(w for w in words_ascii[:8] if w not in terms)
        # Add suffix-stripped stems so "feshinden" → "fesih" also matches
        for w in words[:6]:
            for stem in _stems(to_ascii(w)):
                if stem != to_ascii(w) and stem not in terms and stem not in _STOP_WORDS:
                    terms.append(stem)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in terms:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Passage builder
# ---------------------------------------------------------------------------
def _make_passage(law: dict, art_num: int, title: str, art_text: str) -> dict:
    num_str = str(art_num).zfill(4)
    passage_id = f"ART-{law['code']}-{num_str}"

    title_display = title or f"Madde {art_num}"
    full_title = f"{law['name']} m. {art_num} — {title_display}"
    text = f"{full_title}\n\n{art_text}"
    snippet = art_text[:400]

    source_url = _pdf_url(law)

    match_terms = _make_match_terms(law, art_num, title)

    return {
        "passage_id": passage_id,
        "title": full_title,
        "text": text,
        "snippet": snippet,
        "tag": "Curated Turkish Law",
        "source_dataset": f"{law['number']} sayılı {law['name']}",
        "source_url": source_url,
        "law_code": law["code"],
        "law_number": law["number"],
        "article_number": art_num,
        "article_title": title_display,
        "match_terms": match_terms,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    pid = row.get("passage_id", "")
                    if pid:
                        ids.add(pid)
                except json.JSONDecodeError:
                    pass
    return ids


def _append_passages(passages: list[dict], path: Path) -> int:
    existing = _load_existing_ids(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    with path.open("a", encoding="utf-8") as f:
        for p in passages:
            if p["passage_id"] in existing:
                log.debug("Skip duplicate: %s", p["passage_id"])
                continue
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            existing.add(p["passage_id"])
            new_count += 1
    return new_count


# ---------------------------------------------------------------------------
# Main scrape routine
# ---------------------------------------------------------------------------
def scrape_law(law: dict, delay: float) -> list[dict]:
    try:
        pdf_bytes = download_pdf(law, delay=delay)
    except Exception as exc:
        log.error("Failed to download %s: %s", law["code"], exc)
        return []

    full_text = _extract_text_from_bytes(pdf_bytes)
    log.info("Extracted %d chars from %s PDF", len(full_text), law["code"])

    passages: list[dict] = []
    for art_num, title, art_text in _split_into_articles(full_text):
        if len(art_text) < 20:
            log.debug("Skipping very short article %s m.%d", law["code"], art_num)
            continue
        passages.append(_make_passage(law, art_num, title, art_text))

    log.info("Parsed %d articles from %s", len(passages), law["code"])
    return passages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _law_by_number(number: str) -> dict | None:
    for law in LAWS:
        if law["number"] == number:
            return law
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape mevzuat.gov.tr law PDFs into curated JSONL.")
    parser.add_argument(
        "--laws", nargs="+", metavar="NUMBER",
        help="Law numbers to scrape (e.g. 6098 4721). Default: all predefined laws.",
    )
    parser.add_argument(
        "--output", default="data/curated/law_articles.jsonl",
        help="Output JSONL file (default: data/curated/law_articles.jsonl).",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds to wait between HTTP requests (default: 1.5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be scraped without writing files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    if args.laws:
        target_laws: list[dict] = []
        for num in args.laws:
            law = _law_by_number(num)
            if law:
                target_laws.append(law)
            else:
                # Unknown law number: construct a minimal entry
                log.warning("Law %s not in catalog — will try tertip 5", num)
                target_laws.append({
                    "code": f"LAW{num}",
                    "number": num,
                    "name": f"Kanun {num}",
                    "tertip": "5",
                })
    else:
        target_laws = LAWS

    output_path = Path(args.output)

    if args.dry_run:
        print(f"Would scrape {len(target_laws)} laws -> {output_path}")
        for law in target_laws:
            print(f"  {law['code']:10s} {law['number']:6s}  {law['name']}")
            print(f"             {_pdf_url(law)}")
        return

    total_new = 0
    for law in target_laws:
        passages = scrape_law(law, delay=args.delay)
        if passages:
            added = _append_passages(passages, output_path)
            log.info("Added %d new passages from %s → %s", added, law["code"], output_path)
            total_new += added

    log.info("Done. Total new passages written: %d", total_new)
    if total_new > 0:
        log.info("Rebuild the FAISS index to enable vector search:")
        log.info("  py -3 scripts/build_index.py")


if __name__ == "__main__":
    main()
