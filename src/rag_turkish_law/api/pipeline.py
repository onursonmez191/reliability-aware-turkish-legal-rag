"""Mode-aware orchestration of embed → retrieve → (rerank) → generate → verify."""

from __future__ import annotations

import time
from typing import Sequence

from ..config import load_config
from ..generation.generate import generate_grounded, generate_llm_only
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


def run_pipeline(question: str, mode: str, k: int) -> AskResponse:
    cfg = load_config()
    timings: list[StepTiming] = []

    if mode == "llm":
        answer = _timed("generate", lambda: generate_llm_only(question), timings)
        return AskResponse(
            question=question,
            mode="llm",
            answer=answer.text,
            llm_only=answer.text,
            sources=[],
            verdict=None,
            timings=timings,
        )

    hits = _timed("retrieve", lambda: retrieve(question, k=k), timings)

    if cfg.retrieval.rerank.enabled:
        hits = _timed("rerank", lambda: rerank(question, hits), timings)

    confidence = _timed("confidence", lambda: assess_retrieval_confidence(hits), timings)
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
        )

    grounded = _timed(
        "generate",
        lambda: generate_grounded(question, [h.to_dict() for h in hits]),
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
        )

    # verified mode: also run the verifier.
    claim_verdicts = _timed(
        "verify",
        lambda: verify_answer(grounded.text, [h.to_dict() for h in hits]),
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
    )
