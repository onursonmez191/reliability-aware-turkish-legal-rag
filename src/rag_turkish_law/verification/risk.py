"""Detect legal-advice-risk patterns.

Two layers:
1. Deterministic Turkish patterns — numeric compensation, imperative "do X",
   absolute certainty words.
2. Optional LLM check (caller decides) for borderline cases.

The deterministic layer is cheap and runs on every claim; the LLM layer is
only invoked from `verify.py` when a claim is otherwise marked `supported`
but still trips the risk pattern.
"""

from __future__ import annotations

import re

# Numeric amounts in TL / EUR / USD or generic "X bin/milyon lira" patterns.
MONEY_RE = re.compile(
    r"(?ix)(\d[\d\.\,]*\s*(tl|lira|try|euro|eur|usd|dolar)\b|"
    r"\d+\s*(bin|milyon)\s*(lira|tl)\b)"
)

IMPERATIVE_RE = re.compile(
    r"(?i)\b("
    r"dava\s+a[çc]\w*|"
    r"şikayet\s+et\w*|"
    r"noter[e']?\s+git\w*|"
    r"tapuya\s+git\w*|"
    r"i?cra\s+başlat\w*|"
    r"polise\s+git\w*|"
    r"hemen\s+başvur\w*|"
    r"derhal\s+başvur\w*"
    r")"
)

CERTAINTY_RE = re.compile(
    r"(?i)\b(kesinlikle|mutlaka|kesin\s+olarak|her\s+durumda)\b"
)


def is_case_specific_advice(claim_text: str) -> bool:
    """True if the claim looks like personalized legal action advice."""
    return bool(
        MONEY_RE.search(claim_text)
        or IMPERATIVE_RE.search(claim_text)
        or CERTAINTY_RE.search(claim_text)
    )
