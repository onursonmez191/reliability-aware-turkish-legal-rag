# Reliability-Aware RAG for Turkish Legal QA — Final Report

> Working skeleton. Fill in numeric results and analyses as they become available.

## 1. Introduction

Brief statement of the problem and the proposed system. Reference the
project proposal (`docs/original_materials/CS455_Proposal_Sonmez_Sozer_Erseker.pdf`).

## 2. System design

### 2.1 Overview
React UI → FastAPI backend → (retrieval → generation → verification).

### 2.2 Data
- Source: `OrionCAF/turkish_law_qa_dataset` (Hugging Face).
- Preprocessing: deduplication by normalized question hash, drop empty or
  too-short answers, NFKC + HTML strip + whitespace normalize.
- Held-out: 50 records reserved with `gold_passage_id` for retrieval
  evaluation.
- See `data/processed/preprocessing_report.md` for the exact counts.

### 2.3 Retrieval
- Embedding model: `intfloat/multilingual-e5-base` with the required
  `query: ` / `passage: ` prefixes.
- Index: FAISS `IndexFlatIP` over L2-normalized vectors (cosine).
- Optional reranker: cross-encoder, disabled by default; enabled in
  ablation with a larger candidate pool before keeping the final top-k.
- Confidence gate: refuses to generate when top retrieval scores are too
  weak, while still showing retrieved passages for inspection.

### 2.4 Generation
- Local Ollama by default, with Hugging Face Inference API as a fallback.
- System prompt enforces source-only answers, `[n]` citations, refusal on
  insufficient context, and bans case-specific legal advice.
- LLM-only baseline uses the same model with no source block.

### 2.5 Verification
- Claim splitter (sentence-based).
- Per-claim LLM verifier returns JSON: `{status, source_ids, reason}`.
- Claims keep both answer citations and verifier-supported source IDs, so
  citation/source mismatches can be downgraded.
- Deterministic risk gate catches money amounts, imperatives, and
  certainty words even when the LLM marks a claim as `supported`.
- Verifier backend failures are surfaced as `error`, separate from
  `insufficient`.
- Aggregator turns claim verdicts into an overall `{key, score, risk}`.

## 3. Evaluation

### 3.1 Retrieval
- Metrics: Recall@3, Recall@5, MRR.
- Sets: generated held-out questions with `gold_passage_id`, plus the
  tracked manual/adversarial questions in
  `evaluation/annotations/manual_eval.jsonl`.
- Results: *TBD — fill from `evaluation/results/retrieval_metrics.json`.*

### 3.2 Answer quality (manual rubric)
- Criteria: correctness, source_faithfulness, clarity, refusal_behavior,
  safety.
- Sample size: *TBD.*
- Results: *TBD — fill from rubric aggregation script output.*

### 3.3 Verifier
- Labels: supported, partial, unsupported, insufficient, risk, error.
- Method: human annotation on a sample, compared with verifier output.
- Results: *TBD.*

### 3.4 Ablations
- top_k ∈ {3, 5, 8}.
- Rerank: off vs on.
- Strict vs permissive prompt (optional, time permitting).

## 4. Error analysis

*Pick 5–10 representative failures across the three pipelines and
discuss the root causes.*

## 5. Limitations

- Corpus is QA/explanation style, not official statute text.
- Verifier is itself an LLM and inherits its biases.
- Free Inference API rate limits affect throughput.

## 6. Ethics and educational-use framing

See `docs/ethics/disclaimer.md`.

## 7. Future work

- Add statute-snippet sub-corpus for grounding completeness.
- Replace sentence-based claim splitter with an LLM-based atomizer.
- Try a fine-tuned Turkish reranker on the corpus.

## 8. References

See proposal Section 4.
