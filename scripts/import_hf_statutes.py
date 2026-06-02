"""Import selected public mevzuat.gov.tr statute rows into law_articles.jsonl."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_turkish_law.data.clean import normalize_text  # noqa: E402
from rag_turkish_law.data.statutes import DEFAULT_DATASET, _strip_trailing_heading  # noqa: E402
from scrape_mevzuat import (  # noqa: E402
    LAWS,
    _article_number_label,
    _make_passage,
    _write_passages,
)

log = logging.getLogger("import_hf_statutes")

DEFAULT_OUTPUT = Path("data/curated/law_articles.jsonl")
DEFAULT_REPORT = Path("data/curated/law_articles_import_report.json")


def _law_catalog() -> dict[str, dict[str, Any]]:
    return {str(law["number"]): dict(law) for law in LAWS}


def _law_from_dataset_row(row: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    law_no = str(row.get("kanun_numarasi", "")).strip()
    law = dict(catalog.get(law_no) or {})
    if not law:
        law = {
            "code": f"LAW{law_no}",
            "number": law_no,
            "name": normalize_text(str(row.get("Kanun Adı", ""))) or f"Kanun {law_no}",
            "domain": "uncategorized",
            "priority": 99,
        }
    return law


def _article_text(article: dict[str, Any]) -> str:
    text = normalize_text(str(article.get("text", "")))
    return _strip_trailing_heading(text)


def _article_title(num_raw: str) -> str:
    label = _article_number_label(num_raw)
    return label if "madde" in label else f"Madde {label}"


def import_laws(
    *,
    law_numbers: set[str],
    output_path: Path,
    report_path: Path,
    dataset_name: str,
    split: str,
    replace_laws: bool,
    min_chars: int,
) -> dict[str, Any]:
    catalog = _law_catalog()
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = load_dataset(dataset_name, split=split)

    passages: list[dict] = []
    law_rows: list[dict[str, Any]] = []
    found: set[str] = set()

    for row in rows:
        law_no = str(row.get("kanun_numarasi", "")).strip()
        if law_no not in law_numbers:
            continue

        found.add(law_no)
        law = _law_from_dataset_row(dict(row), catalog)
        source_url = str(row.get("url", ""))
        law_passages: list[dict] = []
        skipped_short = 0

        for article in row.get("maddeler") or []:
            num_raw = normalize_text(str(article.get("madde_numarasi", "")))
            text = _article_text(dict(article))
            if len(text) < min_chars:
                skipped_short += 1
                continue
            law_passages.append(
                _make_passage(
                    law,
                    num_raw,
                    _article_title(num_raw),
                    text,
                    source_url=source_url,
                    source_sha256=f"hf:{dataset_name}",
                    scraped_at=scraped_at,
                )
            )

        passages.extend(law_passages)
        law_rows.append({
            "code": law.get("code"),
            "law_number": law_no,
            "name": law.get("name"),
            "source_url": source_url,
            "raw_article_count": len(row.get("maddeler") or []),
            "parsed_articles": len(law_passages),
            "skipped_short_articles": skipped_short,
        })

    replace_numbers = found if replace_laws else set()
    added, duplicate_ids = _write_passages(
        passages,
        output_path,
        replace_law_numbers=replace_numbers,
    )

    report = {
        "generated_at": scraped_at,
        "dataset_name": dataset_name,
        "split": split,
        "output": str(output_path),
        "requested_laws": sorted(law_numbers),
        "found_laws": sorted(found),
        "missing_laws": sorted(law_numbers - found),
        "replace_laws": replace_laws,
        "article_count": len(passages),
        "added_articles": added,
        "duplicate_ids": len(duplicate_ids),
        "laws": law_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laws", nargs="+", required=True, metavar="NUMBER")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--replace-laws", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    report = import_laws(
        law_numbers={str(law) for law in args.laws},
        output_path=Path(args.output),
        report_path=Path(args.report_output),
        dataset_name=args.dataset_name,
        split=args.split,
        replace_laws=args.replace_laws,
        min_chars=args.min_chars,
    )

    log.info(
        "Imported %d article passages from %d laws into %s",
        report["added_articles"],
        len(report["found_laws"]),
        report["output"],
    )
    log.info("Wrote import report → %s", args.report_output)


if __name__ == "__main__":
    main()
