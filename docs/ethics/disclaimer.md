# Disclaimer and Ethics

## Educational and informational use only

The Reliability-Aware Turkish Legal RAG system is built as a coursework
project for CS 455 (Spring 2025/2026). The system retrieves passages from
a public Turkish legal QA dataset and produces source-grounded answers
with reliability labels. It is **not a substitute for professional legal
advice**. Users facing a concrete legal matter should consult a qualified
Turkish attorney.

## What the system tries to do

- Retrieve the most relevant passages from a corpus of Turkish legal
  QA/explanation entries.
- Generate a short Turkish answer that cites the retrieved passages by
  number (`[1]`, `[2]`, ...) and refuses or qualifies when context is
  insufficient.
- Run a verifier pass that labels each atomic claim as `supported`,
  `partial`, `unsupported`, `insufficient`, `risk`, or `error`, and
  surfaces an overall reliability score.

## What the system does NOT do

- It does not replace a lawyer's judgment on a specific case.
- It does not estimate damages, deadlines, or court outcomes for a
  specific situation. Any output that does so is flagged as
  legal-advice risk.
- It does not guarantee that the underlying corpus is complete or
  up-to-date. The corpus consists of QA/explanation entries rather than
  official statute text, and may diverge from current statutes or case
  law.
- It does not store user queries or train on them.

## Data sources

- Primary corpus: `OrionCAF/turkish_law_qa_dataset` from Hugging Face
  (Apache-2.0 license at time of snapshot). See
  `data/processed/preprocessing_report.md` for the snapshot date and
  cleaning statistics applied at build time.

## Limitations users should know

- Retrieval quality depends on phrasing. Questions far from the corpus's
  vocabulary may surface low-similarity passages or no passages at all,
  in which case the system reports `insufficient context`.
- The generator can still produce errors even when the verifier marks
  claims as supported. The verifier is itself an LLM-judged pass and is
  not a guarantee of correctness.
- A verifier `error` means the reliability check failed technically; it
  does not mean the answer was verified or that the legal context is only
  insufficient.
- The system is Turkish-first. Mixing English or Arabic legal terms may
  degrade retrieval and verification quality.

## Reporting issues

If you observe a misleading or risky output, please save the question,
the answer, and the displayed source list, and share them with the
project team so we can investigate.
