"""Normalize the checked-in scraped statute JSONL without re-scraping PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scrape_mevzuat import (  # noqa: E402
    LAWS,
    _article_number_label,
    _article_status,
    _clean_article_title,
    _make_match_terms,
    _pdf_url,
    _strip_trailing_heading,
)

DEFAULT_PATH = Path("data/curated/law_articles.jsonl")
LAW_BY_NUMBER = {law["number"]: law for law in LAWS}


def _law_from_row(row: dict[str, Any]) -> dict[str, str]:
    number = str(row.get("law_number", ""))
    law = LAW_BY_NUMBER.get(number)
    if law:
        return dict(law)

    title = str(row.get("title", ""))
    law_name = title.split(" m. ", 1)[0] if " m. " in title else ""
    if not law_name:
        source = str(row.get("source_dataset", ""))
        law_name = source.split(" sayılı ", 1)[1] if " sayılı " in source else source
    return {
        "code": str(row.get("law_code", "")),
        "number": number,
        "name": law_name,
        "tertip": "5",
        "domain": str(row.get("domain", "uncategorized")),
        "priority": row.get("priority"),
    }


def _article_title_from_row(row: dict[str, Any]) -> str:
    article_title = str(row.get("article_title") or "")
    if article_title:
        return article_title

    title = str(row.get("title") or "")
    if " — " in title:
        return title.split(" — ", 1)[1]
    return ""


def _article_body_from_row(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "")
    if "\n\n" in text:
        body = text.split("\n\n", 1)[1]
    else:
        body = str(row.get("snippet") or text)
    return _strip_trailing_heading(body)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    law = _law_from_row(row)
    art_num = row["article_number"]
    article_label = _article_number_label(art_num)
    article_title = _clean_article_title(_article_title_from_row(row))
    if not article_title:
        article_title = f"Madde {article_label}"

    title_prefix = f"{law['name']} {article_label}" if "madde" in article_label else f"{law['name']} m. {article_label}"
    full_title = f"{title_prefix} — {article_title}"
    body = _article_body_from_row(row)

    row["title"] = full_title
    row["text"] = f"{full_title}\n\n{body}"
    row["snippet"] = body[:400]
    row["source_dataset"] = f"{law['number']} sayılı {law['name']}"
    row["source_url"] = row.get("source_url") or _pdf_url(law)
    row["law_code"] = law["code"]
    row["law_number"] = law["number"]
    row["domain"] = law.get("domain", row.get("domain", "uncategorized"))
    row["priority"] = law.get("priority", row.get("priority"))
    row["article_title"] = article_title
    row["article_status"] = _article_status(body)
    row["indexable"] = row["article_status"] != "repealed"
    row.setdefault("scraped_at", "")
    row.setdefault("source_sha256", "")
    row["match_terms"] = _make_match_terms(law, art_num, article_title)
    return row


def normalize_file(path: Path) -> int:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(normalize_row(json.loads(line)))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    count = normalize_file(Path(args.path))
    print(f"Normalized {count} statute article rows in {args.path}")


if __name__ == "__main__":
    main()
