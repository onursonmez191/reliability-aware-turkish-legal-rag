"""
Download Turkish law PDFs from mevzuat.gov.tr and extract article-level passages.

Each article becomes a passage in data/curated/law_articles.jsonl, enabling
direct article lookup via both keyword matching and FAISS vector search.

After scraping, rebuild the FAISS index:
    py -3 scripts/build_index.py

Usage:
    py -3 scripts/scrape_mevzuat.py                       # scrape all predefined laws
    py -3 scripts/scrape_mevzuat.py --laws 6098 4721      # specific law numbers
    py -3 scripts/scrape_mevzuat.py --domains enforcement property
    py -3 scripts/scrape_mevzuat.py --replace-laws --domains enforcement
    py -3 scripts/scrape_mevzuat.py --text-source-dir data/raw/law_text --no-pdf-fallback
    py -3 scripts/scrape_mevzuat.py --dry-run             # preview without writing
    py -3 scripts/scrape_mevzuat.py --delay 2.0           # set request delay (seconds)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import logging
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

log = logging.getLogger("scrape_mevzuat")

# ---------------------------------------------------------------------------
# Law catalog — (code, law_number, law_name, domain, priority, optional tertip)
# Tertip 5 = 1983-present, Tertip 4 = 1964-1982, Tertip 3 = 1943-1963
# ---------------------------------------------------------------------------
LAWS: list[dict] = [
    {"code": "TBK",    "number": "6098", "name": "Türk Borçlar Kanunu",                               "tertip": "5", "domain": "civil", "priority": 1},
    {"code": "TMK",    "number": "4721", "name": "Türk Medeni Kanunu",                                "tertip": "5", "domain": "civil", "priority": 1},
    {"code": "TCK",    "number": "5237", "name": "Türk Ceza Kanunu",                                  "tertip": "5", "domain": "criminal", "priority": 1},
    {"code": "ISK",    "number": "4857", "name": "İş Kanunu",                                         "tertip": "5", "domain": "labor", "priority": 1},
    {"code": "HMK",    "number": "6100", "name": "Hukuk Muhakemeleri Kanunu",                         "tertip": "5", "domain": "procedure", "priority": 1},
    {"code": "TTK",    "number": "6102", "name": "Türk Ticaret Kanunu",                               "tertip": "5", "domain": "commerce", "priority": 2},
    {"code": "CMK",    "number": "5271", "name": "Ceza Muhakemesi Kanunu",                            "tertip": "5", "domain": "criminal", "priority": 1},
    {"code": "SSGSSK", "number": "5510", "name": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu", "tertip": "5", "domain": "labor", "priority": 2},
    {"code": "KMK",    "number": "634",  "name": "Kat Mülkiyeti Kanunu",                              "tertip": "5", "domain": "property", "priority": 1},
    {"code": "IYUK",   "number": "2577", "name": "İdari Yargılama Usulü Kanunu",                      "tertip": "5", "domain": "administrative", "priority": 2},
    {"code": "KVK",    "number": "5520", "name": "Kurumlar Vergisi Kanunu",                           "tertip": "5", "domain": "tax", "priority": 3},
    {"code": "KDVK",   "number": "3065", "name": "Katma Değer Vergisi Kanunu",                        "tertip": "5", "domain": "tax", "priority": 3},
    {"code": "AY",     "number": "2709", "name": "Türkiye Cumhuriyeti Anayasası",                     "tertip": "5", "domain": "constitutional", "priority": 2},
    {"code": "VUK",    "number": "213",  "name": "Vergi Usul Kanunu",                                 "tertip": "4", "domain": "tax", "priority": 3},
    {"code": "GVK",    "number": "193",  "name": "Gelir Vergisi Kanunu",                              "tertip": "4", "domain": "tax", "priority": 3},
    {"code": "BK",     "number": "818",  "name": "Borçlar Kanunu (Eski)",                             "tertip": "3", "domain": "legacy", "priority": 4},

    # Demo-focused expansion batch.
    {"code": "IIK",    "number": "2004", "name": "İcra ve İflas Kanunu",                              "domain": "enforcement", "priority": 1},
    {"code": "AATUHK", "number": "6183", "name": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",     "domain": "enforcement", "priority": 2},
    {"code": "TAPU",   "number": "2644", "name": "Tapu Kanunu",                                        "domain": "property", "priority": 1},
    {"code": "KADK",   "number": "3402", "name": "Kadastro Kanunu",                                    "domain": "property", "priority": 2},
    {"code": "NOTK",   "number": "1512", "name": "Noterlik Kanunu",                                    "domain": "property", "priority": 2},
    {"code": "TKHK",   "number": "6502", "name": "Tüketicinin Korunması Hakkında Kanun",               "domain": "consumer", "priority": 1},
    {"code": "KTK",    "number": "2918", "name": "Karayolları Trafik Kanunu",                          "domain": "traffic", "priority": 1},
    {"code": "IMK",    "number": "7036", "name": "İş Mahkemeleri Kanunu",                              "domain": "labor", "priority": 1},
    {"code": "ISG",    "number": "6331", "name": "İş Sağlığı ve Güvenliği Kanunu",                     "domain": "labor", "priority": 2},
    {"code": "STISK",  "number": "6356", "name": "Sendikalar ve Toplu İş Sözleşmesi Kanunu",           "domain": "labor", "priority": 2},
    {"code": "KVKK",   "number": "6698", "name": "Kişisel Verilerin Korunması Kanunu",                 "domain": "privacy", "priority": 1},
    {"code": "HAYK",   "number": "5199", "name": "Hayvanları Koruma Kanunu",                           "domain": "local_government", "priority": 1},
    {"code": "BELED",  "number": "5393", "name": "Belediye Kanunu",                                    "domain": "local_government", "priority": 2},
    {"code": "BSEHIR", "number": "5216", "name": "Büyükşehir Belediyesi Kanunu",                       "domain": "local_government", "priority": 2},
    {"code": "TEBK",   "number": "7201", "name": "Tebligat Kanunu",                                    "domain": "procedure", "priority": 2},
    {"code": "KABK",   "number": "5326", "name": "Kabahatler Kanunu",                                  "domain": "procedure", "priority": 2},
    {"code": "DMK",    "number": "657",  "name": "Devlet Memurları Kanunu",                            "domain": "administrative", "priority": 2},
]

MEVZUAT_PDF_URL = "https://www.mevzuat.gov.tr/MevzuatMetin/1.{tertip}.{number}.pdf"
DEFAULT_TERTIP_CANDIDATES = ("5", "4", "3")

# Regex for article markers: "MADDE 49-" / "Madde 49-" / "Madde 49 -"
# Web/plain-text sources often render the marker on its own line: "MADDE 49".
# Laws vary: TBK uses all-caps, TMK/TCK/ISK use title-case.
MADDE_RE = re.compile(r"^MADDE\s+(\d+)\s*(?:[-–—]|\r?\n|$)", re.MULTILINE | re.IGNORECASE)

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
    asciiish = text.translate(_TR_ASCII)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", asciiish)
        if unicodedata.category(ch) != "Mn"
    )


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("\u0307", "").strip()


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------
@dataclass
class DownloadedPdf:
    data: bytes
    url: str
    tertip: str
    sha256: str


@dataclass
class DownloadedText:
    text: str
    source_url: str
    source_sha256: str


@dataclass
class ScrapeResult:
    law: dict
    passages: list[dict] = field(default_factory=list)
    source_url: str = ""
    source_sha256: str = ""
    resolved_tertip: str = ""
    chars_extracted: int = 0
    parsed_articles: int = 0
    skipped_short_articles: int = 0
    duplicate_ids: int = 0
    status: str = "ok"
    error: str = ""

    def report_row(self, added: int = 0) -> dict:
        return {
            "code": self.law.get("code"),
            "law_number": self.law.get("number"),
            "name": self.law.get("name"),
            "domain": self.law.get("domain", "uncategorized"),
            "priority": self.law.get("priority"),
            "status": self.status,
            "source_url": self.source_url,
            "resolved_tertip": self.resolved_tertip,
            "source_sha256": self.source_sha256,
            "chars_extracted": self.chars_extracted,
            "parsed_articles": self.parsed_articles,
            "skipped_short_articles": self.skipped_short_articles,
            "duplicate_ids": self.duplicate_ids,
            "added_articles": added,
            "error": self.error,
        }


def _pdf_url(law: dict, tertip: str | None = None) -> str:
    selected = tertip or law.get("tertip") or DEFAULT_TERTIP_CANDIDATES[0]
    return MEVZUAT_PDF_URL.format(tertip=selected, number=law["number"])


def _candidate_tertips(law: dict) -> list[str]:
    tertip = law.get("tertip")
    if tertip:
        return [str(tertip)]
    return list(DEFAULT_TERTIP_CANDIDATES)


def download_pdf(law: dict, delay: float = 1.5, *, timeout: float = 60.0) -> DownloadedPdf:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TurkishLegalRAG/1.0; "
            "research use; +https://github.com)"
        ),
        "Accept": "application/pdf",
    }
    errors: list[str] = []
    for tertip in _candidate_tertips(law):
        url = _pdf_url(law, tertip=tertip)
        log.info("Downloading %s (%s) from %s", law["code"], law["name"], url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            time.sleep(delay)
            return DownloadedPdf(
                data=data,
                url=url,
                tertip=tertip,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        except urllib.error.HTTPError as exc:
            errors.append(f"{url}: HTTP {exc.code}")
            if exc.code not in {403, 404}:
                break
        except (TimeoutError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc!r}")
            continue
        except Exception as exc:
            errors.append(f"{url}: {exc!r}")
            break
    raise RuntimeError("; ".join(errors) or f"No PDF URL candidates for {law['code']}")


# ---------------------------------------------------------------------------
# PDF text extraction (character-level to bypass font encoding issues)
# ---------------------------------------------------------------------------
def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

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
# HTML / plain-text source extraction
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe"}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self.parts.append(data)

    def text(self) -> str:
        lines = []
        for line in "".join(self.parts).splitlines():
            cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


def _decode_source_bytes(data: bytes, content_type: str = "") -> str:
    match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "cp1254", "iso-8859-9"])
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_like_html(text: str, source_url: str = "") -> bool:
    if source_url.lower().endswith((".html", ".htm")):
        return True
    sample = text[:1000].lower()
    return bool(
        "<!doctype html" in sample
        or re.search(r"<(?:html|body|main|article|section|div|p|h[1-6]|table|span|br)\b", sample)
    )


def _html_to_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return parser.text()


def _clean_source_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\xa0", " ").replace("\u00ad", "")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    lines = []
    for line in normalized.splitlines():
        cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _source_bytes_to_text(data: bytes, *, content_type: str = "", source_url: str = "") -> str:
    decoded = _decode_source_bytes(data, content_type=content_type)
    if _looks_like_html(decoded, source_url):
        decoded = _html_to_text(decoded)
    return _clean_source_text(decoded)


def _download_text_url(url: str, *, timeout: float = 60.0) -> tuple[bytes, str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TurkishLegalRAG/1.0; "
            "research use; +https://github.com)"
        ),
        "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.5",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        content_type = resp.headers.get("content-type", "")
        resolved_url = resp.geturl()
    return data, content_type, resolved_url


def _read_text_source(source: dict, *, timeout: float = 60.0) -> DownloadedText:
    path_value = source.get("path")
    url_value = source.get("url")
    if path_value:
        path = Path(path_value)
        data = path.read_bytes()
        source_url = str(source.get("source_url") or path)
        text = _source_bytes_to_text(data, source_url=source_url)
    elif url_value:
        data, content_type, resolved_url = _download_text_url(str(url_value), timeout=timeout)
        source_url = str(source.get("source_url") or resolved_url)
        text = _source_bytes_to_text(data, content_type=content_type, source_url=source_url)
    else:
        raise ValueError("Text source row must include either 'path' or 'url'.")

    return DownloadedText(
        text=text,
        source_url=source_url,
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


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
        title = _clean_article_title(title)

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


_TITLE_FOOTNOTE_RE = re.compile(r"(?<=[^\W\d_])\d{1,2}$", re.UNICODE)
_INLINE_FOOTNOTE_RE = re.compile(r"(?<=[.!?…])\d{1,3}(?=\s)")
_TRAILING_STRUCTURAL_HEADING_RE = re.compile(
    r"\s+(?:[A-ZÇĞİÖŞÜ]+(?:NCİ|NCI|NCU|NCÜ|İNCİ|INCI|UNCU|ÜNCÜ)\s+)?"
    r"(?:KİTAP|KISIM|BÖLÜM|AYIRIM)\b"
)
_TRAILING_SIDE_HEADING_START_RE = re.compile(
    r"\s(?:\d{1,3}\.\s*|[IVXLCDM]{1,6}\s*[-.]\s*|[A-ZÇĞİÖŞÜa-zçğıöşü][).]\s*)"
)


def _ends_like_sentence(text: str) -> bool:
    if not text:
        return False
    if text[-1] in ".!?…":
        return True
    return len(text) >= 2 and text[-1] == ")" and text[-2] in ".!?…"


def _ends_like_clause(text: str) -> bool:
    if not text:
        return False
    return text[-1].isalnum() or _ends_like_sentence(text)


def _looks_like_heading_tail(text: str) -> bool:
    words = re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", text)
    return 2 <= len(words) <= 9 and len(text) <= 160


def _looks_like_list_continuation(candidate: str, tail: str) -> bool:
    marker = r"(?:\d{1,3}\.|[a-zçğıöşü]\))"
    if not re.match(rf"^{marker}", tail.strip()):
        return False
    return bool(re.search(rf"[:;]\s*{marker}", candidate))


def _strip_trailing_side_heading(text: str) -> str:
    for match in reversed(list(_TRAILING_SIDE_HEADING_START_RE.finditer(text))):
        candidate = text[: match.start()].strip()
        tail = text[match.start():].strip()
        if (
            len(candidate) >= 20
            and _ends_like_sentence(candidate)
            and _looks_like_heading_tail(tail)
            and not _looks_like_list_continuation(candidate, tail)
        ):
            return candidate
    return text


def _strip_trailing_heading(text: str) -> str:
    """Remove next-article side-heading fragments appended to article text.

    The PDF parser often attaches the next article's section heading to the
    end of the current article, e.g. '...açık veya örtülü olabilir.
    2. İkinci derecedeki noktalar'. This strips that trailing run.
    """
    cleaned = _INLINE_FOOTNOTE_RE.sub("", text.strip())
    for match in _TRAILING_STRUCTURAL_HEADING_RE.finditer(cleaned):
        candidate = cleaned[: match.start()].strip()
        if len(candidate) >= 20 and _ends_like_clause(candidate):
            cleaned = candidate
            break
    return _strip_trailing_side_heading(cleaned)


def _clean_article_title(title: str) -> str:
    """Normalise a PDF-derived article side heading."""
    cleaned = re.sub(r"\s+", " ", title).strip(" -–")
    # mevzuat PDFs sometimes attach footnote numbers to side headings:
    # "geri verilmesi2" -> "geri verilmesi".
    cleaned = _TITLE_FOOTNOTE_RE.sub("", cleaned).strip()
    return cleaned


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
def _make_match_terms(law: dict, art_num: int | str, title: str) -> list[str]:
    code = law["code"].lower()
    code_ascii = to_ascii(code)
    _, label, number, suffix = _article_number_parts(art_num)
    ref_variants = [label]
    if number not in ref_variants:
        ref_variants.append(number)
    if suffix:
        ref_variants.append(f"{number}{suffix.lower()}")

    terms: list[str] = []

    # Law code + article number variants
    for num in ref_variants:
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
_REPEALED_ARTICLE_RE = re.compile(r"^\s*\(?\s*Mülga\b", re.IGNORECASE)
_ARTICLE_NUMBER_PART_RE = re.compile(
    r"(\d+)\s*(?:[/.-]\s*([A-Za-zÇĞİÖŞÜçğıöşü]+))?",
    re.IGNORECASE,
)


def _article_number_parts(art_num: int | str) -> tuple[str, str, str, str]:
    """Return (id_suffix, label, numeric_part, ascii_suffix) for article refs."""
    raw = str(art_num).strip()
    normalized = to_ascii(_norm(raw)).upper()

    prefix = ""
    if "GECICI" in normalized:
        prefix = "GECICI"
    elif normalized.startswith("EK") or "EK MADDE" in normalized:
        prefix = "EK"

    match = _ARTICLE_NUMBER_PART_RE.search(raw)
    if match:
        number = match.group(1)
        suffix = to_ascii(match.group(2) or "").upper()
    else:
        fallback = re.sub(r"[^0-9A-Za-z]+", "", to_ascii(raw)).upper() or "X"
        number = fallback
        suffix = ""

    id_suffix = number.zfill(4) if number.isdigit() else number
    if suffix:
        id_suffix = f"{id_suffix}{suffix}"
    if prefix:
        id_suffix = f"{prefix}-{id_suffix}"

    if prefix == "GECICI":
        label = f"geçici madde {number}"
    elif prefix == "EK":
        label = f"ek madde {number}"
    elif suffix:
        label = f"{number}/{suffix.lower()}"
    else:
        label = number

    return id_suffix, label, number, suffix


def _article_number_label(art_num: int | str) -> str:
    return _article_number_parts(art_num)[1]


def _article_status(art_text: str) -> str:
    text = re.sub(r"^\s*[-–—]\s*", "", art_text)
    if _REPEALED_ARTICLE_RE.match(text):
        return "repealed"
    return "active"


def _make_passage(
    law: dict,
    art_num: int | str,
    title: str,
    art_text: str,
    *,
    source_url: str | None = None,
    source_sha256: str = "",
    scraped_at: str = "",
) -> dict:
    id_suffix, article_label, _, _ = _article_number_parts(art_num)
    passage_id = f"ART-{law['code']}-{id_suffix}"

    title_display = _clean_article_title(title) or f"Madde {art_num}"
    title_prefix = f"{law['name']} {article_label}" if "madde" in article_label else f"{law['name']} m. {article_label}"
    full_title = f"{title_prefix} — {title_display}"
    text = f"{full_title}\n\n{art_text}"
    snippet = art_text[:400]

    resolved_source_url = source_url or _pdf_url(law)

    match_terms = _make_match_terms(law, art_num, title_display)
    status = _article_status(art_text)

    return {
        "passage_id": passage_id,
        "title": full_title,
        "text": text,
        "snippet": snippet,
        "tag": "Curated Turkish Law",
        "source_dataset": f"{law['number']} sayılı {law['name']}",
        "source_url": resolved_source_url,
        "law_code": law["code"],
        "law_number": law["number"],
        "domain": law.get("domain", "uncategorized"),
        "priority": law.get("priority"),
        "article_number": art_num,
        "article_title": title_display,
        "article_status": status,
        "indexable": status != "repealed",
        "scraped_at": scraped_at,
        "source_sha256": source_sha256,
        "match_terms": match_terms,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _load_existing_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _write_passages(
    passages: list[dict],
    path: Path,
    *,
    replace_law_numbers: set[str] | None = None,
) -> tuple[int, set[str]]:
    replace_law_numbers = replace_law_numbers or set()
    existing_rows = _load_existing_rows(path)
    kept_rows = [
        row for row in existing_rows
        if str(row.get("law_number", "")) not in replace_law_numbers
    ]
    existing_ids = {
        row.get("passage_id", "")
        for row in kept_rows
        if row.get("passage_id")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    duplicate_ids: set[str] = set()
    with path.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for passage in passages:
            if passage["passage_id"] in existing_ids:
                log.debug("Skip duplicate: %s", passage["passage_id"])
                duplicate_ids.add(passage["passage_id"])
                continue
            f.write(json.dumps(passage, ensure_ascii=False) + "\n")
            existing_ids.add(passage["passage_id"])
            new_count += 1
    return new_count, duplicate_ids


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scrape routine
# ---------------------------------------------------------------------------
def scrape_law(law: dict, delay: float, *, scraped_at: str, timeout: float = 60.0) -> ScrapeResult:
    result = ScrapeResult(law=law)
    try:
        pdf = download_pdf(law, delay=delay, timeout=timeout)
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        log.error("Failed to download %s: %s", law["code"], exc)
        return result

    result.source_url = pdf.url
    result.resolved_tertip = pdf.tertip
    result.source_sha256 = pdf.sha256

    full_text = _extract_text_from_bytes(pdf.data)
    result.chars_extracted = len(full_text)
    log.info("Extracted %d chars from %s PDF", len(full_text), law["code"])

    passages: list[dict] = []
    for art_num, title, art_text in _split_into_articles(full_text):
        if len(art_text) < 20:
            log.debug("Skipping very short article %s m.%d", law["code"], art_num)
            result.skipped_short_articles += 1
            continue
        passages.append(
            _make_passage(
                law,
                art_num,
                title,
                art_text,
                source_url=pdf.url,
                source_sha256=pdf.sha256,
                scraped_at=scraped_at,
            )
        )

    result.passages = passages
    result.parsed_articles = len(passages)
    log.info("Parsed %d articles from %s", len(passages), law["code"])
    return result


def _passages_from_text(
    law: dict,
    full_text: str,
    *,
    source_url: str,
    source_sha256: str,
    scraped_at: str,
) -> tuple[list[dict], int]:
    passages: list[dict] = []
    skipped_short_articles = 0
    for art_num, title, art_text in _split_into_articles(full_text):
        if len(art_text) < 20:
            log.debug("Skipping very short article %s m.%d", law["code"], art_num)
            skipped_short_articles += 1
            continue
        passages.append(
            _make_passage(
                law,
                art_num,
                title,
                art_text,
                source_url=source_url,
                source_sha256=source_sha256,
                scraped_at=scraped_at,
            )
        )
    return passages, skipped_short_articles


def scrape_law_text_source(
    law: dict,
    source: dict,
    *,
    scraped_at: str,
    timeout: float = 60.0,
) -> ScrapeResult:
    result = ScrapeResult(law=law)
    try:
        text_source = _read_text_source(source, timeout=timeout)
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        log.error("Failed to read text source for %s: %s", law["code"], exc)
        return result

    result.source_url = text_source.source_url
    result.source_sha256 = text_source.source_sha256
    result.resolved_tertip = "text"
    result.chars_extracted = len(text_source.text)

    passages, skipped_short_articles = _passages_from_text(
        law,
        text_source.text,
        source_url=text_source.source_url,
        source_sha256=text_source.source_sha256,
        scraped_at=scraped_at,
    )
    result.passages = passages
    result.parsed_articles = len(passages)
    result.skipped_short_articles = skipped_short_articles
    log.info("Parsed %d articles from %s text source", len(passages), law["code"])
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _law_by_number(number: str) -> dict | None:
    for law in LAWS:
        if law["number"] == number:
            return law
    return None


def _law_key(law: dict) -> tuple[int, str, str]:
    return (int(law.get("priority") or 99), str(law.get("domain", "")), str(law.get("code", "")))


def _select_laws(numbers: list[str] | None, domains: list[str] | None) -> list[dict]:
    selected: list[dict] = []
    selected_numbers: set[str] = set()

    def add(law: dict) -> None:
        number = str(law["number"])
        if number not in selected_numbers:
            selected.append(law)
            selected_numbers.add(number)

    if domains:
        wanted_domains = {domain.casefold() for domain in domains}
        for law in sorted(LAWS, key=_law_key):
            if str(law.get("domain", "")).casefold() in wanted_domains:
                add(law)

    if numbers:
        for number in numbers:
            law = _law_by_number(number)
            if law:
                add(law)
            else:
                # Unknown law number: construct a minimal entry and let URL
                # resolution try the known tertip values.
                log.warning("Law %s not in catalog — will try known tertip values", number)
                add({
                    "code": f"LAW{number}",
                    "number": number,
                    "name": f"Kanun {number}",
                    "domain": "uncategorized",
                    "priority": 99,
                })

    if not selected:
        selected = sorted(LAWS, key=_law_key)

    return selected


def _source_number(source: dict) -> str:
    number = source.get("law_number") or source.get("number")
    if not number:
        raise ValueError("Text source row must include 'law_number' or 'number'.")
    return str(number)


def _load_text_source_manifest(path: Path) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            sources[_source_number(row)] = row
    return sources


def _load_text_source_dir(path: Path) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for source_path in sorted(path.iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in {".txt", ".html", ".htm"}:
            continue
        match = re.match(r"(\d+)", source_path.stem)
        if not match:
            continue
        number = match.group(1)
        sources[number] = {"number": number, "path": str(source_path)}
    return sources


def _load_text_sources(manifest: str | None, source_dir: str | None) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    if source_dir:
        sources.update(_load_text_source_dir(Path(source_dir)))
    if manifest:
        # The manifest wins over auto-discovered files so source_url metadata
        # can be attached without renaming local files.
        sources.update(_load_text_source_manifest(Path(manifest)))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape mevzuat.gov.tr law PDFs into curated JSONL.")
    parser.add_argument(
        "--laws", nargs="+", metavar="NUMBER",
        help="Law numbers to scrape (e.g. 6098 4721). Default: all predefined laws.",
    )
    parser.add_argument(
        "--domains", nargs="+", metavar="DOMAIN",
        help="Scrape catalog laws in these domains (e.g. enforcement property consumer).",
    )
    parser.add_argument(
        "--output", default="data/curated/law_articles.jsonl",
        help="Output JSONL file (default: data/curated/law_articles.jsonl).",
    )
    parser.add_argument(
        "--report-output", default="data/curated/law_articles_report.json",
        help="Audit report JSON path (default: data/curated/law_articles_report.json).",
    )
    parser.add_argument(
        "--replace-laws", action="store_true",
        help="Replace existing rows for scraped law numbers instead of only appending new IDs.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds to wait between HTTP requests (default: 1.5).",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-URL download timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--text-source-manifest",
        help=(
            "JSONL source map for HTML/plain-text law sources. Each row needs "
            "'number' or 'law_number' plus 'url' or 'path'."
        ),
    )
    parser.add_argument(
        "--text-source-dir",
        help=(
            "Directory containing local law text/HTML files named by law number "
            "(e.g. data/raw/law_text/2004.html)."
        ),
    )
    parser.add_argument(
        "--no-pdf-fallback", action="store_true",
        help="When text sources are configured, skip laws with no matching text source instead of downloading PDFs.",
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

    target_laws = _select_laws(args.laws, args.domains)
    text_sources = _load_text_sources(args.text_source_manifest, args.text_source_dir)

    output_path = Path(args.output)
    report_path = Path(args.report_output)

    if args.dry_run:
        print(f"Would scrape {len(target_laws)} laws -> {output_path}")
        if args.replace_laws:
            print("Would replace existing rows for target law numbers before writing.")
        if text_sources:
            print(f"Text sources configured for {len(text_sources)} law numbers.")
        for law in target_laws:
            domain = law.get("domain", "uncategorized")
            priority = law.get("priority", "-")
            print(f"  {law['code']:10s} {law['number']:6s}  {domain:18s} p{priority}  {law['name']}")
            source = text_sources.get(str(law["number"]))
            if source:
                locator = source.get("path") or source.get("url")
                print(f"             text source: {locator}")
            elif args.no_pdf_fallback:
                print("             no text source; would skip")
            else:
                for tertip in _candidate_tertips(law):
                    print(f"             {_pdf_url(law, tertip=tertip)}")
        return

    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[ScrapeResult] = []
    all_passages: list[dict] = []
    for law in target_laws:
        source = text_sources.get(str(law["number"]))
        if source:
            result = scrape_law_text_source(law, source, scraped_at=scraped_at, timeout=args.timeout)
        elif args.no_pdf_fallback and text_sources:
            result = ScrapeResult(
                law=law,
                status="skipped",
                error="No matching text source and --no-pdf-fallback was set.",
            )
            log.warning("Skipping %s: no matching text source", law["code"])
        else:
            result = scrape_law(law, delay=args.delay, scraped_at=scraped_at, timeout=args.timeout)
        results.append(result)
        all_passages.extend(result.passages)

    replace_numbers = (
        {str(result.law["number"]) for result in results if result.passages}
        if args.replace_laws
        else set()
    )
    total_new, duplicate_ids = _write_passages(
        all_passages,
        output_path,
        replace_law_numbers=replace_numbers,
    )

    for result in results:
        result.duplicate_ids = sum(1 for passage in result.passages if passage["passage_id"] in duplicate_ids)
        added = len(result.passages) - result.duplicate_ids
        log.info("Added %d new passages from %s → %s", added, result.law["code"], output_path)

    report = {
        "generated_at": scraped_at,
        "output": str(output_path),
        "replace_laws": args.replace_laws,
        "requested_laws": [law["number"] for law in target_laws],
        "requested_domains": args.domains or [],
        "text_source_manifest": args.text_source_manifest or "",
        "text_source_dir": args.text_source_dir or "",
        "no_pdf_fallback": args.no_pdf_fallback,
        "law_count": len(results),
        "article_count": sum(result.parsed_articles for result in results),
        "added_articles": total_new,
        "duplicate_ids": len(duplicate_ids),
        "skipped_short_articles": sum(result.skipped_short_articles for result in results),
        "laws": [result.report_row(added=len(result.passages) - result.duplicate_ids) for result in results],
    }
    _write_report(report, report_path)
    log.info("Wrote scrape report → %s", report_path)

    log.info("Done. Total new passages written: %d", total_new)
    if total_new > 0:
        log.info("Rebuild the FAISS index to enable vector search:")
        log.info("  py -3 scripts/build_index.py")


if __name__ == "__main__":
    main()
