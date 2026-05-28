"""Print retrieval diagnostics for a single question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_turkish_law.config import load_config  # noqa: E402
from rag_turkish_law.retrieval.confidence import assess_retrieval_confidence  # noqa: E402
from rag_turkish_law.retrieval.query_expansion import expand_retrieval_queries  # noqa: E402
from rag_turkish_law.retrieval.search import retrieve  # noqa: E402


LEGAL_TERMS = (
    "hayvan bulunduran",
    "hayvanın verdiği zarar",
    "hayvanin verdigi zarar",
    "kusursuz sorumluluk",
    "tazminat",
    "sahipsiz hayvan",
    "belediye sorumluluğu",
    "köpek",
    "kopek",
)


def _norm(text: str) -> str:
    return text.casefold()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _term_hits(rows: list[dict], terms: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    hits: list[tuple[str, str, list[str]]] = []
    for row in rows:
        blob = _norm(" ".join(str(row.get(k, "")) for k in ("title", "text", "snippet")))
        matched = [term for term in terms if _norm(term) in blob]
        if matched:
            hits.append((row.get("passage_id", "?"), row.get("title", "")[:120], matched))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug retrieval coverage and top-k hits.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Köpeğim birine zarar verirse veya bir köpek bana zarar verirse tazminatı kimden istemeliyim?",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--coverage-limit", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config()
    passages_path = Path(cfg.paths.passages_file)
    curated_path = Path(cfg.paths.get("curated_sources_file", "data/curated/legal_sources.jsonl"))

    passages = _load_jsonl(passages_path)
    curated = _load_jsonl(curated_path)

    print("QUESTION")
    print(args.question)
    print()

    expanded = expand_retrieval_queries(args.question)
    print("RETRIEVAL QUERIES")
    for i, query in enumerate(expanded, 1):
        print(f"{i}. {query}")
    print()

    print("CORPUS TERM COVERAGE")
    processed_hits = _term_hits(passages, LEGAL_TERMS)
    curated_hits = _term_hits(curated, LEGAL_TERMS)
    print(f"- processed passages with target terms: {len(processed_hits)} / {len(passages)}")
    for pid, title, matched in processed_hits[: args.coverage_limit]:
        print(f"  - {pid}: {title} | terms={matched}")
    print(f"- curated passages with target terms: {len(curated_hits)} / {len(curated)}")
    for pid, title, matched in curated_hits[: args.coverage_limit]:
        print(f"  - {pid}: {title} | terms={matched}")
    print()

    hits = retrieve(args.question, k=args.k)
    confidence = assess_retrieval_confidence(hits)
    print("CONFIDENCE")
    print(f"- label={confidence.label}")
    print(f"- top_score={confidence.top_score:.4f}")
    print(f"- avg_top3_score={confidence.avg_top3_score:.4f}")
    print(f"- reason={confidence.reason}")
    print()

    print("TOP HITS")
    for i, hit in enumerate(hits, 1):
        blob = _norm(f"{hit.title} {hit.snippet} {hit.text}")
        matched = [term for term in LEGAL_TERMS if _norm(term) in blob]
        print(f"{i}. {hit.passage_id} score={hit.score:.4f} tag={hit.tag or '-'}")
        print(f"   title={hit.title}")
        print(f"   matched_terms={matched or []}")
        print(f"   snippet={hit.snippet[:260].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
