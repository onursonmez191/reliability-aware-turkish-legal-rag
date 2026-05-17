# Reliability-Aware RAG for Turkish Legal QA

General scaffold for the CS455 LLM course project.

This repository is intended for an educational Turkish legal question-answering RAG system with:

- corpus preparation for Turkish legal QA/explanation passages
- retrieval experiments with multilingual embeddings and a vector index
- source-grounded answer generation
- a verifier layer for support, insufficiency, and legal-advice-risk labels
- manual evaluation, error analysis, limitations, and ethics documentation
- a small demo UI for the final project

No implementation has been added yet.

## Structure

```text
app/                         Demo UI entry point later, e.g. Streamlit or Gradio
configs/                     Experiment and model configuration files
data/
  raw/                       Original dataset snapshots
  processed/                 Cleaned passages and derived data
  evaluation/                Held-out questions, gold passages, and labels
docs/
  original_materials/        Course handouts, proposal template, accepted proposal
  reports/                   Final report drafts and submitted report assets
  ethics/                    Limitations, disclaimer, and legal-use notes
evaluation/
  annotations/               Manual annotation rubric and labeled evaluation files
  results/                   Metrics, ablation outputs, and error analysis tables
notebooks/                   Exploration and course-friendly experiments
scripts/                     Reproducible command-line helpers later
src/rag_turkish_law/
  data/                      Data loading and preprocessing code later
  retrieval/                 Embedding, indexing, and retrieval code later
  generation/                Prompting and answer-generation code later
  verification/              Reliability verifier code later
  evaluation/                Metrics and analysis code later
tests/                       Focused tests for implemented components
```

## Feedback To Track

- Define the manual annotation process clearly before evaluation.
- Reserve time for error analysis and refinement after initial metrics.
- Keep compute assumptions flexible for Colab, Kaggle, and local inference.
- Discuss limitations and ethical constraints thoroughly in the final report.
