"""Mode-aware orchestration of embed → retrieve → (rerank) → generate → verify."""

from __future__ import annotations

import time
from typing import Sequence

from ..config import load_config
from ..generation.generate import generate_grounded, generate_llm_only, stream_grounded_text, stream_llm_only_text
from ..retrieval.confidence import assess_retrieval_confidence
from ..retrieval.rerank import rerank
from ..retrieval.search import RetrievedPassage, retrieve
from ..verification.aggregate import aggregate
from ..verification.verify import verify_answer
from .schemas import AskResponse, ClaimItem, SourceItem, StepTiming, Verdict


def _passages_to_sources(hits: Sequence[RetrievedPassage]) -> list[SourceItem]:
    return [
        SourceItem(
            id=h.passage_id,
            title=h.title or "Untitled passage",
            snippet=h.snippet,
            tag=h.tag,
            score=h.score,
        )
        for h in hits
    ]


def _timed(label: str, fn, timings: list[StepTiming]):
    t0 = time.perf_counter()
    result = fn()
    timings.append(StepTiming(id=label, ms=int((time.perf_counter() - t0) * 1000)))
    return result


def _event(name: str, data: dict) -> dict:
    return {"event": name, "data": data}


def run_pipeline(
    question: str,
    mode: str,
    k: int,
    model: str | None = None,
    verifier_model: str | None = None,
) -> AskResponse:
    cfg = load_config()
    timings: list[StepTiming] = []
    verifier_model = verifier_model or model

    if mode == "llm":
        answer = _timed("generate", lambda: generate_llm_only(question, model=model), timings)
        return AskResponse(
            question=question,
            mode="llm",
            answer=answer.text,
            llm_only=answer.text,
            sources=[],
            verdict=None,
            timings=timings,
            model=model,
            verifier_model=None,
        )

    hits = _timed("retrieve", lambda: retrieve(question, k=k), timings)

    # Assess coverage on the bi-encoder retrieval scores. The cross-encoder
    # reranker emits scores on a different scale (0-1 relevance) that are not
    # comparable to the confidence thresholds, so it must not feed the gate —
    # rerank only reorders the passages retrieval already found.
    confidence = _timed("confidence", lambda: assess_retrieval_confidence(hits), timings)

    if cfg.retrieval.rerank.enabled:
        hits = _timed("rerank", lambda: rerank(question, hits), timings)

    if confidence.label == "low":
        answer = (
            "Mevcut kaynaklar bu soruyu yeterince kapsamıyor. "
            "Bu nedenle kaynaklara dayalı güvenilir bir yanıt üretemiyorum."
        )
        if mode == "rag":
            return AskResponse(
                question=question,
                mode="rag",
                answer=answer,
                llm_only=None,
                sources=_passages_to_sources(hits),
                verdict=None,
                timings=timings,
                model=model,
                verifier_model=None,
            )

        verdict = Verdict(
            key="insufficient",
            score=0.0,
            risk="medium",
            claims=[
                ClaimItem(
                    text=answer,
                    status="insufficient",
                    src=[],
                    cited=[],
                )
            ],
        )
        return AskResponse(
            question=question,
            mode="verified",
            answer=answer,
            llm_only=None,
            sources=_passages_to_sources(hits),
            verdict=verdict,
            timings=timings,
            model=model,
            verifier_model=verifier_model,
        )

    grounded = _timed(
        "generate",
        lambda: generate_grounded(question, [h.to_dict() for h in hits], model=model),
        timings,
    )

    if mode == "rag":
        return AskResponse(
            question=question,
            mode="rag",
            answer=grounded.text,
            llm_only=None,
            sources=_passages_to_sources(hits),
            verdict=None,
            timings=timings,
            model=model,
            verifier_model=None,
        )

    # verified mode: also run the verifier.
    claim_verdicts = _timed(
        "verify",
        lambda: verify_answer(grounded.text, [h.to_dict() for h in hits], model=verifier_model),
        timings,
    )
    overall = aggregate(claim_verdicts)

    verdict = Verdict(
        key=overall.key,
        score=overall.score,
        risk=overall.risk,
        claims=[ClaimItem(**c) for c in overall.claims],
    )

    return AskResponse(
        question=question,
        mode="verified",
        answer=grounded.text,
        llm_only=None,
        sources=_passages_to_sources(hits),
        verdict=verdict,
        timings=timings,
        model=model,
        verifier_model=verifier_model,
    )


def run_pipeline_stream(
    question: str,
    mode: str,
    k: int,
    model: str | None = None,
    verifier_model: str | None = None,
):
    cfg = load_config()
    timings: list[StepTiming] = []
    verifier_model = verifier_model or model

    if mode == "llm":
        yield _event("step", {"id": "generate"})
        t0 = time.perf_counter()
        chunks: list[str] = []
        for chunk in stream_llm_only_text(question, model=model):
            chunks.append(chunk)
            yield _event("chunk", {"text": chunk})
        timings.append(StepTiming(id="generate", ms=int((time.perf_counter() - t0) * 1000)))
        answer = "".join(chunks)
        response = AskResponse(
            question=question,
            mode="llm",
            answer=answer,
            llm_only=answer,
            sources=[],
            verdict=None,
            timings=timings,
            model=model,
            verifier_model=None,
        )
        yield _event("final", response.model_dump())
        return

    yield _event("step", {"id": "embed"})
    yield _event("step", {"id": "retrieve"})
    hits = _timed("retrieve", lambda: retrieve(question, k=k), timings)

    # Coverage gate runs on the bi-encoder scores, before the cross-encoder
    # reranker rescales them (see run_pipeline for the rationale).
    confidence = _timed("confidence", lambda: assess_retrieval_confidence(hits), timings)

    if cfg.retrieval.rerank.enabled:
        yield _event("step", {"id": "rerank"})
        hits = _timed("rerank", lambda: rerank(question, hits), timings)

    sources = _passages_to_sources(hits)
    yield _event("sources", {"sources": [s.model_dump() for s in sources]})

    yield _event("step", {"id": "confidence"})
    if confidence.label == "low":
        answer = (
            "Mevcut kaynaklar bu soruyu yeterince kapsamıyor. "
            "Bu nedenle kaynaklara dayalı güvenilir bir yanıt üretemiyorum."
        )
        yield _event("chunk", {"text": answer})
        verdict = None
        if mode == "verified":
            verdict = Verdict(
                key="insufficient",
                score=0.0,
                risk="medium",
                claims=[
                    ClaimItem(
                        text=answer,
                        status="insufficient",
                        src=[],
                        cited=[],
                    )
                ],
            )
            yield _event("verdict", {"verdict": verdict.model_dump()})

        response = AskResponse(
            question=question,
            mode=mode,
            answer=answer,
            llm_only=None,
            sources=sources,
            verdict=verdict,
            timings=timings,
            model=model,
            verifier_model=verifier_model if mode == "verified" else None,
        )
        yield _event("final", response.model_dump())
        return

    yield _event("step", {"id": "generate"})
    t0 = time.perf_counter()
    chunks = []
    passage_dicts = [h.to_dict() for h in hits]
    for chunk in stream_grounded_text(question, passage_dicts, model=model):
        chunks.append(chunk)
        yield _event("chunk", {"text": chunk})
    timings.append(StepTiming(id="generate", ms=int((time.perf_counter() - t0) * 1000)))
    answer = "".join(chunks)

    if mode == "rag":
        response = AskResponse(
            question=question,
            mode="rag",
            answer=answer,
            llm_only=None,
            sources=sources,
            verdict=None,
            timings=timings,
            model=model,
            verifier_model=None,
        )
        yield _event("final", response.model_dump())
        return

    yield _event("step", {"id": "verify"})
    claim_verdicts = _timed(
        "verify",
        lambda: verify_answer(answer, passage_dicts, model=verifier_model),
        timings,
    )
    overall = aggregate(claim_verdicts)
    verdict = Verdict(
        key=overall.key,
        score=overall.score,
        risk=overall.risk,
        claims=[ClaimItem(**c) for c in overall.claims],
    )
    yield _event("verdict", {"verdict": verdict.model_dump()})

    response = AskResponse(
        question=question,
        mode="verified",
        answer=answer,
        llm_only=None,
        sources=sources,
        verdict=verdict,
        timings=timings,
        model=model,
        verifier_model=verifier_model,
    )
    yield _event("final", response.model_dump())
