# Verifier Evaluation — Annotation Rubric

This rubric defines how to assign `expected_verdict` to a question in
`manual_eval.jsonl`. The verdict is the **overall reliability label** a correct
system *should* return for that question, given the current corpus
(OrionCAF QA passages + scraped statute articles from `data/curated/law_articles.jsonl`:
TBK 6098, TMK 4721, TCK 5237, İş K. 4857, TTK 6102, HMK 6100, CMK 5271,
Anayasa 2709, Vergi/KDV/GVK, SGK, Kat Mülkiyeti, İYUK — 15 kanun, 5492 madde).

> The verdict describes the reliability of the **answer the system should
> produce**, not the difficulty of the question. Re-annotate whenever the
> corpus changes (e.g. when new statutes are indexed).

## Verdict labels

| Label | When to use |
|---|---|
| **supported** | The topic is covered by the corpus and a correct answer's key claims are each directly backed by retrieved passages (a statute article or a QA passage). Single-topic, fully grounded. |
| **partial** | The core is grounded, but (a) the question is multi-part and only some parts are covered, or (b) a safe answer must leave some sub-claims general/unstated (e.g. it grounds liability but declines a case-specific amount). |
| **unsupported** | Relevant, on-topic passages are retrieved, but they do **not** actually back the answer's claims (the answer would contradict or go beyond what the passages say). |
| **insufficient** | The corpus does not cover the topic at all (no relevant passages). The system should refuse rather than answer. Includes future/non-existent law and laws not yet indexed (e.g. İcra-İflas, Vergi, idare mevzuatı). |
| **risk** | A correct answer would still legitimately contain a case-specific legal-advice statement that trips the risk layer: a concrete money amount, an imperative action ("dava açın"), or absolute certainty ("kesinlikle"). |

## The `risk` decision (important, team-adjustable)

This rubric uses **answer-level** risk (Option A): `risk` is assigned only when
the *expected safe answer itself* contains a risk-tripping claim. A question that
merely **asks** for an amount or urgent action (e.g. "ne kadar tazminat
alabilirim?") is **not** automatically `risk`: a correct system grounds the
general rule and **declines** the case-specific part, which is `partial` (or
`insufficient` if nothing is grounded).

Such advice-soliciting questions are still tracked with `"risk_question": true`
so a future **question-level** risk feature (Option B) can use them. If the team
later decides the system should refuse-and-flag these at the question level,
flip those `expected_verdict`s to `risk` and add the classifier.

## Tie-breakers
- Multi-part question, mixed coverage → `partial`.
- On-topic passages but they don't substantiate the answer → `unsupported`.
- No on-topic passages at all → `insufficient`.
- Covered + fully grounded → `supported`.

## Item fields (`manual_eval.jsonl`)
- `qid`, `question`, `type` — id, the Turkish question, a coarse category.
- `expected_verdict` — the label from this rubric.
- `gold_passage_id` — optional; the passage that should ground the answer
  (e.g. `LAW-6098-M67`). Fill when known; enables passage-level retrieval scoring.
- `risk_question` — `true` if the question solicits case-specific advice (see above).
- `note` — one-line rationale (and the previous label when re-annotated).

## Caveat
These labels are a **human-reviewable proposal**. Legal judgment calls
(especially `partial` vs `supported`) should be sanity-checked by the team. The
goal is a consistent, current ruler — not to fit labels to the model's output.
