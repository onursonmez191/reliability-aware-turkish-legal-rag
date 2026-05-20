"""Aggregate per-claim verdicts into an overall verdict the UI can show.

Weights mirror what users perceive as reliable: `supported` is full credit,
`partial` half, `risk` zero with a risk flag, `unsupported`/`insufficient`
zero. The chosen `key` follows a priority order so a single risky claim
dominates the banner color (matches UI's `VerdictCard` behavior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .verify import ClaimVerdict

STATUS_WEIGHTS = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
    "insufficient": 0.0,
    "risk": 0.0,
}


@dataclass
class OverallVerdict:
    key: str
    score: float
    risk: str
    claims: list[dict]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "score": round(self.score, 2),
            "risk": self.risk,
            "claims": self.claims,
        }


def _pick_overall_key(claims: Sequence[ClaimVerdict]) -> str:
    if not claims:
        return "insufficient"
    statuses = {c.status for c in claims}
    if "risk" in statuses:
        return "risk"
    if statuses == {"supported"}:
        return "supported"
    if "unsupported" in statuses and "supported" not in statuses and "partial" not in statuses:
        return "unsupported"
    if statuses <= {"insufficient", "unsupported"}:
        return "insufficient"
    return "partial"


def _risk_level(claims: Sequence[ClaimVerdict], score: float) -> str:
    if any(c.status == "risk" for c in claims):
        return "high"
    if score < 0.5:
        return "medium"
    return "low"


def aggregate(claims: Sequence[ClaimVerdict]) -> OverallVerdict:
    if not claims:
        return OverallVerdict(key="insufficient", score=0.0, risk="medium", claims=[])

    score = sum(STATUS_WEIGHTS.get(c.status, 0.0) for c in claims) / len(claims)
    key = _pick_overall_key(claims)
    risk = _risk_level(claims, score)
    return OverallVerdict(
        key=key,
        score=score,
        risk=risk,
        claims=[c.to_dict() for c in claims],
    )
