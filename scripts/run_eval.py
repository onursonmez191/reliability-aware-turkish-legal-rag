"""Run retrieval evaluation against the held-out set + ablations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_turkish_law.config import load_config  # noqa: E402
from rag_turkish_law.data.passages import read_jsonl  # noqa: E402
from rag_turkish_law.evaluation.ablation import run_topk_ablation  # noqa: E402
from rag_turkish_law.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument("--ablation", choices=["none", "topk", "rerank"], default="none")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    log = logging.getLogger("eval")

    cfg = load_config()
    heldout_path = Path(cfg.paths.heldout_file)
    if not heldout_path.exists():
        log.error("Held-out file missing (%s). Run scripts/build_index.py first.", heldout_path)
        sys.exit(2)

    heldout = list(read_jsonl(heldout_path))
    out_dir = Path(cfg.paths.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation == "topk":
        results = run_topk_ablation(heldout, ks=(3, 5, 8))
        path = out_dir / "ablation_topk.json"
    elif args.ablation == "rerank":
        from rag_turkish_law.evaluation.ablation import run_rerank_ablation

        results = run_rerank_ablation(heldout)
        path = out_dir / "ablation_rerank.json"
    else:
        results = evaluate_retrieval(heldout, k=cfg.retrieval.top_k)
        path = out_dir / "retrieval_metrics.json"

    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
