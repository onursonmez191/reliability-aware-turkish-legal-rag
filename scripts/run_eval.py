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
from rag_turkish_law.evaluation.ablation import run_topk_ablation  # noqa: E402
from rag_turkish_law.evaluation.eval_set import DEFAULT_MANUAL_EVAL_PATH, build_eval_set  # noqa: E402
from rag_turkish_law.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402
from rag_turkish_law.evaluation.verifier_eval import evaluate_verifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation suites.")
    parser.add_argument("--ablation", choices=["none", "topk", "rerank"], default="none")
    parser.add_argument("--eval-set", choices=["heldout", "manual", "combined"], default="heldout")
    parser.add_argument("--suite", choices=["retrieval", "verifier"], default="retrieval")
    parser.add_argument("--manual-path", default=str(DEFAULT_MANUAL_EVAL_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    log = logging.getLogger("eval")

    cfg = load_config()
    heldout_path = Path(cfg.paths.heldout_file)
    if args.eval_set in {"heldout", "combined"} and not heldout_path.exists():
        log.error("Held-out file missing (%s). Run scripts/build_index.py first.", heldout_path)
        sys.exit(2)

    items = build_eval_set(heldout_path, args.manual_path, eval_set=args.eval_set)
    if not items:
        log.error("No evaluation items loaded for eval-set=%s.", args.eval_set)
        sys.exit(2)

    out_dir = Path(cfg.paths.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "verifier":
        if args.ablation != "none":
            log.error("Verifier suite does not support --ablation.")
            sys.exit(2)
        results = evaluate_verifier(items, k=cfg.retrieval.top_k)
        path = out_dir / f"verifier_{args.eval_set}_metrics.json"
    elif args.ablation == "topk":
        results = run_topk_ablation(items, ks=(3, 5, 8))
        path = out_dir / f"ablation_topk_{args.eval_set}.json"
    elif args.ablation == "rerank":
        from rag_turkish_law.evaluation.ablation import run_rerank_ablation

        results = run_rerank_ablation(items)
        path = out_dir / f"ablation_rerank_{args.eval_set}.json"
    else:
        results = evaluate_retrieval(items, k=cfg.retrieval.top_k)
        path = out_dir / (
            "retrieval_metrics.json"
            if args.eval_set == "heldout"
            else f"retrieval_{args.eval_set}_diagnostics.json"
        )

    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
