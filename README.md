# Reliability-Aware RAG for Turkish Legal QA

CS 455 LLM course project. A Turkish legal question-answering RAG system with:

- corpus preparation from the public `OrionCAF/turkish_law_qa_dataset`
  plus scraped article-level statute passages under `data/curated/`
- multilingual sentence embeddings + a FAISS index
- source-grounded answer generation through a local Ollama server (default) or the Hugging Face Inference API
- a verifier layer that labels each claim as `supported / partial / unsupported / insufficient / risk / error`
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
  annotations/               Manual/adversarial eval JSONL and annotation worksheets
  results/                   Metric JSONs and ablation outputs
scripts/
  build_index.py             load → clean → embed → FAISS
  scrape_mevzuat.py          scrape article-level statute passages from mevzuat.gov.tr PDFs
  normalize_law_articles.py  clean committed statute JSONL metadata after scraper fixes
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
rules, the embedding model, or the corpus. The default config indexes the
OrionCAF QA corpus plus every JSONL file in `data/curated/`, including the
scraped article-level statute corpus in `data/curated/law_articles.jsonl`
(15 laws, 5492 articles). The older Hugging Face statute loader is disabled
by default to avoid duplicate statute rows.

The scraper catalog also contains a demo-focused expansion batch for
enforcement, tapu/property, consumer, traffic, labor procedure, privacy,
local-government/animal, and common procedure questions. To refresh the active
statute JSONL from those domains:

```bash
python scripts/scrape_mevzuat.py \
  --domains enforcement property consumer traffic labor privacy local_government procedure \
  --replace-laws
python scripts/normalize_law_articles.py
python scripts/build_index.py
```

The scraper writes an audit report to
`data/curated/law_articles_report.json`. Use `--dry-run` to preview target
laws and candidate PDF URLs, and `--timeout 10` if the public mevzuat site is
slow or unreachable. `--replace-laws` only replaces rows for laws that were
successfully downloaded and parsed, so failed downloads do not delete existing
law rows.

For the current demo gap batch, the safer fallback is the public
`muhammetakkurt/mevzuat-gov-dataset` cache, which is sourced from
`mevzuat.gov.tr` and includes the target laws even when the live PDF endpoint is
unreliable:

```bash
python scripts/import_hf_statutes.py \
  --laws 2004 2644 3402 6502 2918 7036 6698 5199 5393 7201 \
  --replace-laws
python scripts/normalize_law_articles.py
python scripts/build_index.py
```

If the public PDF endpoint is down, the same scraper can ingest plain text or
saved HTML pages. Put local files under `data/raw/law_text/` using the law
number as the filename, for example `data/raw/law_text/2004.html` or
`data/raw/law_text/2644.txt`, then run:

```bash
python scripts/scrape_mevzuat.py \
  --laws 2004 2644 3402 6502 2918 7036 6698 5199 5393 7201 \
  --text-source-dir data/raw/law_text \
  --no-pdf-fallback \
  --replace-laws
python scripts/normalize_law_articles.py
python scripts/build_index.py
```

For better source auditing, use a JSONL manifest instead of only relying on
filenames:

```jsonl
{"number": "2004", "path": "data/raw/law_text/2004.html", "source_url": "https://example.test/icra-ve-iflas-kanunu-2004"}
{"number": "2644", "url": "https://example.test/tapu-kanunu-2644"}
```

Then pass `--text-source-manifest data/raw/law_text_sources.jsonl`. Prefer
official, public, or institutionally mirrored statute text where possible; a
private legal platform is fine for checking content manually, but it should not
become the project's hard-coded bulk source.

If the statute scraper logic changes but you do not need to download the PDFs
again, normalize the checked-in JSONL and then rebuild:

```bash
python scripts/normalize_law_articles.py
python scripts/build_index.py
```

## Run the demo

```bash
python scripts/serve.py
# open http://127.0.0.1:8000
```

The FastAPI app serves the React UI at `/` and exposes:

- `POST /api/ask` — `{question, mode, k, model}` → answer + sources + verdict + timings
- `POST /api/ask/stream` — same payload, Server-Sent Events with step, source, token, verdict, and final events
- `GET /api/models` — configured Ollama models, installed/running state
- `POST /api/models/load` — pre-load one configured model and unload other configured resident models
- `POST /api/models/unload` — eject a configured model from Ollama memory
- `GET /api/health` — liveness check

UI modes (selectable in the header tabs and the Tweaks panel):

- **A. LLM-only** — single generation call, no retrieval, ungrounded.
- **B. RAG** — embed → retrieve → generate with citations.
- **C. RAG + Verifier** — adds claim-level verification with a reliability banner.

The Model Runtime card in the right column lets you switch between the
allowlisted Ollama models and load/eject them manually. Loading a model through
the UI unloads any other configured model that is already resident, which avoids
keeping multiple large dense models in memory on a local laptop.

## CLI sanity checks

```bash
python -m rag_turkish_law.generation.generate "Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?"
python -m rag_turkish_law.generation.generate --llm-only "Aynı soru."
```

## Evaluation

```bash
python scripts/run_eval.py                                  # held-out retrieval smoke test
python scripts/run_eval.py --eval-set manual                # manual/adversarial retrieval diagnostics
python scripts/run_eval.py --eval-set combined              # held-out + manual diagnostics
python scripts/run_eval.py --ablation topk                  # k in {3,5,8}
python scripts/run_eval.py --ablation rerank                # retrieve candidate pool, then rerank
python scripts/run_eval.py --suite verifier --eval-set manual
python scripts/debug_retrieval.py "Köpeğim birine zarar verirse tazminatı kimden isterim?"
```

Results land in `evaluation/results/`.

### Interpreting retrieval numbers

The script-generated held-out set is useful as a smoke test, but it should not
be treated as the final evaluation because its questions are derived from the
dataset itself. Final reported numbers should include a manually written set
with paraphrased, ambiguous, unsupported, and legal-advice-risk questions.
The tracked starter set is [evaluation/annotations/manual_eval.jsonl](evaluation/annotations/manual_eval.jsonl).

Manual retrieval metrics are strict passage-id diagnostics when a
`gold_passage_id` is present. Missing the exact gold article can still leave
the system with a useful on-topic QA passage, so treat those numbers as a
retrieval debugging signal rather than as final answer correctness.

The rerank ablation retrieves a larger candidate pool first and then applies
the cross-encoder reranker, so `--ablation rerank` is a real off/on comparison.

The retrieval confidence gate can refuse to generate when top retrieval scores
are too weak. This avoids presenting low-coverage answers as grounded legal
answers, but it is still a heuristic: high similarity does not guarantee legal
relevance.

The active legal sources now come from the OrionCAF QA passages plus the
article-level statute corpus in `data/curated/law_articles.jsonl`. Earlier
hand-written source patches have been removed where the corresponding statute
article is now available. `scripts/debug_retrieval.py` prints the expanded
retrieval queries, target-term coverage, top-k hits, and confidence labels for
one question.

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
- `retrieval.top_k` — default 8
- `retrieval.rerank.enabled` — turn on cross-encoder reranking
- `retrieval.rerank.candidate_k` — candidate pool size for rerank ablations
- `retrieval.confidence.*` — low-confidence retrieval refusal thresholds
- `generation.provider` — `ollama` (default) or `hf`
- `generation.base_url` — Ollama endpoint (only used when `provider: ollama`)
- `generation.hf_model` — Ollama model tag (e.g. `qwen2.5:7b-instruct`) or HF model id depending on provider
- `verification.hf_model` / `verification.temperature` / `verification.max_new_tokens` — verifier call settings
- `models.available` — UI allowlist for selectable Ollama models
- `models.default` — model selected by default in the UI
- `models.keep_alive` — how long Ollama keeps a loaded model resident
- `data.statutes.enabled` — legacy HF statute loader, disabled by default because scraped statutes load from `data/curated/`
- `data.min_answer_chars` / `data.heldout_size` — preprocessing thresholds

## Notes on safety

- The system prompt forbids the model from inventing facts beyond the
  retrieved passages and from giving case-specific legal advice.
- The verifier flags any claim that mentions numeric compensation,
  imperative actions, or certainty words as `risk`, even when the LLM
  marks it `supported`.
- Verifier backend failures are reported as `error`, not as ordinary
  insufficient evidence.
- The UI always shows the disclaimer banner.

## Feedback to track

- Define the manual annotation process clearly before evaluation.
- Reserve time for error analysis and refinement after initial metrics.
- Keep compute assumptions flexible for Colab, Kaggle, and local inference.
- Discuss limitations and ethical constraints thoroughly in the final report.
