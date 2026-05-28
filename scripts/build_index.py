"""End-to-end: load → clean → passages → embed → FAISS index → write metadata."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src/` importable when the package is not installed yet.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_turkish_law.config import load_config  # noqa: E402
from rag_turkish_law.data.clean import clean_records  # noqa: E402
from rag_turkish_law.data.load import iter_raw_records  # noqa: E402
from rag_turkish_law.data.passages import read_jsonl, to_passages, write_jsonl  # noqa: E402
from rag_turkish_law.data.splits import make_heldout  # noqa: E402
from rag_turkish_law.retrieval.embed import embed_passages  # noqa: E402
from rag_turkish_law.retrieval.index import build_index, save_index, save_meta  # noqa: E402


def _write_report(stats: dict, path: Path, *, corpus_size: int, heldout_size: int) -> None:
    lines = [
        "# Preprocessing Report",
        "",
        "## Raw → cleaned",
        "",
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Corpus splits",
        f"- corpus passages indexed: {corpus_size}",
        f"- held-out questions: {heldout_size}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for Turkish legal QA.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records (debug).")
    parser.add_argument("--no-heldout", action="store_true", help="Skip held-out split.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    log = logging.getLogger("build_index")

    cfg = load_config()

    log.info("Loading raw records from %s", cfg.data.dataset_name)
    raw_iter = iter_raw_records()
    if args.limit:
        raw_iter = (row for i, row in enumerate(raw_iter) if i < args.limit)

    log.info("Cleaning records")
    clean = clean_records(raw_iter)
    log.info("Clean stats: %s", clean.stats.as_dict())

    log.info("Building passages")
    passages = list(to_passages(clean.records))

    if args.no_heldout:
        corpus, heldout = passages, []
    else:
        corpus, heldout = make_heldout(passages, cfg.data.heldout_size, cfg.data.seed)

    curated_path = Path(cfg.paths.get("curated_sources_file", "data/curated/legal_sources.jsonl"))
    curated = list(read_jsonl(curated_path)) if curated_path.exists() else []
    if curated:
        log.info("Adding %d curated legal passages from %s", len(curated), curated_path)
        corpus.extend(curated)

    log.info("Writing %d corpus passages → %s", len(corpus), cfg.paths.passages_file)
    write_jsonl(corpus, cfg.paths.passages_file)

    if heldout:
        log.info("Writing %d held-out items → %s", len(heldout), cfg.paths.heldout_file)
        write_jsonl(heldout, cfg.paths.heldout_file)

    log.info("Embedding %d passages with %s", len(corpus), cfg.retrieval.embedding_model)
    texts = [p["text"] for p in corpus]
    vectors = embed_passages(texts)

    log.info("Building FAISS index (dim=%d)", vectors.shape[1])
    index = build_index(vectors)
    save_index(index, cfg.paths.faiss_index)
    save_meta(corpus, cfg.paths.passage_meta)

    _write_report(
        clean.stats.as_dict(),
        Path(cfg.paths.preprocessing_report),
        corpus_size=len(corpus),
        heldout_size=len(heldout),
    )
    log.info("Done. Index: %s | Meta: %s", cfg.paths.faiss_index, cfg.paths.passage_meta)


if __name__ == "__main__":
    main()
