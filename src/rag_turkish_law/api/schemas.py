"""Pydantic models for the API.

Field names mirror the mock objects in `app/data.jsx` so the React frontend
needs zero changes beyond swapping its data source from `window.ANSWERS`
to a `fetch` call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["llm", "rag", "verified"]
VerdictKey = Literal["supported", "partial", "unsupported", "insufficient", "risk", "error"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    mode: Mode = "verified"
    k: int = Field(8, ge=1, le=20)


class SourceItem(BaseModel):
    id: str
    title: str
    snippet: str
    tag: str
    score: float


class ClaimItem(BaseModel):
    text: str
    status: VerdictKey
    src: list[int] = Field(default_factory=list)
    cited: list[int] = Field(default_factory=list)


class Verdict(BaseModel):
    key: VerdictKey
    score: float
    risk: Literal["low", "medium", "high"]
    claims: list[ClaimItem]


class StepTiming(BaseModel):
    id: str
    ms: int


class AskResponse(BaseModel):
    question: str
    mode: Mode
    answer: str
    llm_only: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    verdict: Verdict | None = None
    timings: list[StepTiming] = Field(default_factory=list)
