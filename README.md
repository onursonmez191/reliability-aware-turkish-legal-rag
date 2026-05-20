# Reliability-Aware RAG for Turkish Legal QA

CS 455 LLM course project. A Turkish legal question-answering RAG system with:

- corpus preparation from the public `OrionCAF/turkish_law_qa_dataset`
- multilingual sentence embeddings + a FAISS index
- source-grounded answer generation through a local Ollama server (default) or the Hugging Face Inference API
- a verifier layer that labels each claim as `supported / partial / unsupported / insufficient / risk`
- a React demo UI (in `app/`) wired to a FastAPI backend
- evaluation scaffolding for Recall@k, MRR, manual rubric, and verifier metrics

> Educational and informational use only — not professional legal advice.
> See [docs/ethics/disclaimer.md](docs/ethics/disclaimer.md).

## Structure

```text
app/                         React prototype UI (CDN-loaded, no build step)
configs/default.yaml         Embedding model, top-k, paths, HF model, etc.
data/
  raw/                       Reserved for raw snapshots
  processed/                 passages.jsonl, faiss.index, passage_meta.jsonl, heldout.jsonl
  evaluation/                Annotation files
docs/
  original_materials/        Course handouts and accepted proposal
  reports/                   Final report skeleton
  ethics/                    Disclaimer and limitations
evaluation/
  annotations/               Annotation worksheets
  results/                   Metric JSONs and ablation outputs
scripts/
  build_index.py             load → clean → embed → FAISS
  serve.py                   FastAPI + static React UI
  run_eval.py                Retrieval metrics and ablations
src/rag_turkish_law/
  config.py                  YAML config loader
  data/                      load, clean, passages, splits
  retrieval/                 embed (e5), index (FAISS), search, optional rerank
  generation/                HF Inference client, Turkish prompts, citation parser
  verification/              claim splitter, per-claim verifier, risk patterns, aggregator
  evaluation/                eval set, retrieval metrics, rubric, verifier metrics, ablations
  api/                       Pydantic schemas, pipeline orchestration, FastAPI app
tests/                       pytest sanity tests for the data, prompts, and verifier
```

## Setup

```bash
python3.10 -m venv .venv               # Python 3.10 or 3.11 recommended
. .venv/bin/activate                # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .                    # makes `rag_turkish_law` importable
cp .env.example .env                # optional, only needed for HF_API_TOKEN/RAG_CONFIG
```

GPU is optional but recommended. If you have an NVIDIA GPU with current
drivers, install a matching PyTorch wheel (e.g. CUDA 12.4):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

### LLM backend (default: local Ollama)

The default config talks to a local [Ollama](https://ollama.com) server
on `http://127.0.0.1:11434`. This avoids HF Inference API rate limits and
keeps the demo reproducible.

Install Ollama using the official installer or your package manager. On macOS
with Homebrew:

```bash
brew install ollama
```

On Linux, the official installer is usually the simplest option:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama and pull the default model:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull qwen2.5:7b-instruct       # ~4.7 GB Q4_K_M, fits in 6 GB VRAM
```

Smaller / faster alternatives (override in [configs/default.yaml](configs/default.yaml)
under `generation.hf_model`):

- `qwen2.5:3b-instruct` — ~2 GB, ~2× faster, slightly weaker Turkish
- `llama3.1:8b-instruct-q4_K_M` — comparable size, alternative family

### Falling back to Hugging Face Inference API

Set `generation.provider: hf` in [configs/default.yaml](configs/default.yaml)
and put `HF_API_TOKEN=hf_...` in `.env`. The HF tier is noisier and
often slow for verifier-style multi-call workloads, so Ollama is the
recommended default.

## Build the index

```bash
python scripts/build_index.py
# debug run on a small subset:
python scripts/build_index.py --limit 200
```

Output goes to `data/processed/`. Re-run when you change the cleaning
rules, the embedding model, or the corpus.

## Run the demo

```bash
python scripts/serve.py
# open http://127.0.0.1:8000
```

The FastAPI app serves the React UI at `/` and exposes:

- `POST /api/ask` — `{question, mode, k}` → answer + sources + verdict + timings
- `GET /api/health` — liveness check

UI modes (selectable in the header tabs and the Tweaks panel):

- **A. LLM-only** — single generation call, no retrieval, ungrounded.
- **B. RAG** — embed → retrieve → generate with citations.
- **C. RAG + Verifier** — adds claim-level verification with a reliability banner.

## CLI sanity checks

```bash
python -m rag_turkish_law.generation.generate "Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?"
python -m rag_turkish_law.generation.generate --llm-only "Aynı soru."
```

## Evaluation

```bash
python scripts/run_eval.py                       # Recall@3/@5, MRR over held-out
python scripts/run_eval.py --ablation topk       # k ∈ {3,5,8}
python scripts/run_eval.py --ablation rerank     # rerank on vs off
```

Results land in `evaluation/results/`.

### Interpreting retrieval numbers

The script-generated held-out set is useful as a smoke test, but it should not
be treated as the final evaluation because its questions are derived from the
dataset itself. Final reported numbers should include a manually written set
with paraphrased, ambiguous, unsupported, and legal-advice-risk questions.

Use the notebooks in `notebooks/` for exploratory checks, and use
`scripts/run_eval.py` for reproducible metrics once the evaluation set is
stable.

## Tests

```bash
pytest tests/ -v
```

The FAISS round-trip test is skipped automatically when `faiss-cpu` is
not installed.

## Configuration

Edit [configs/default.yaml](configs/default.yaml) or point `RAG_CONFIG`
at a different file. Common knobs:

- `retrieval.embedding_model` — swap to `paraphrase-multilingual-MiniLM-L12-v2` for ablation
- `retrieval.top_k` — default 5
- `retrieval.rerank.enabled` — turn on cross-encoder reranking
- `generation.provider` — `ollama` (default) or `hf`
- `generation.base_url` — Ollama endpoint (only used when `provider: ollama`)
- `generation.hf_model` — Ollama model tag (e.g. `qwen2.5:7b-instruct`) or HF model id depending on provider
- `data.min_answer_chars` / `data.heldout_size` — preprocessing thresholds

## Notes on safety

- The system prompt forbids the model from inventing facts beyond the
  retrieved passages and from giving case-specific legal advice.
- The verifier flags any claim that mentions numeric compensation,
  imperative actions, or certainty words as `risk`, even when the LLM
  marks it `supported`.
- The UI always shows the disclaimer banner.

## Feedback to track

- Define the manual annotation process clearly before evaluation.
- Reserve time for error analysis and refinement after initial metrics.
- Keep compute assumptions flexible for Colab, Kaggle, and local inference.
- Discuss limitations and ethical constraints thoroughly in the final report.
