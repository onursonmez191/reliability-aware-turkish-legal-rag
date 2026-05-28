"""Split a generated answer into atomic claims.

The first iteration is sentence-based with a Turkish-aware splitter. Each
claim keeps the citations attached to its source sentence so the verifier
knows which passages the answer used for that statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITE_RE = re.compile(r"\[(\d+)\]")
# Turkish-friendly sentence end: ., !, ?, ; followed by whitespace or EOS.
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")
DOT_SENTINEL = "<DOT>"
ABBREVIATION_RE = re.compile(
    r"\b(?:m|md|mad|bkz|vb|vs|vd|no|dr|prof|av)\.",
    re.IGNORECASE,
)
LEGAL_ARTICLE_RE = re.compile(r"\b([mM])\.(?=\s*\d)")


@dataclass
class Claim:
    text: str
    cited: list[int]

    def text_without_citations(self) -> str:
        text = CITE_RE.sub("", self.text)
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\s+([.,;:!?])", r"\1", text)

    def to_dict(self) -> dict:
        return {"text": self.text_without_citations(), "cited": self.cited}


def split_into_claims(answer_text: str) -> list[Claim]:
    text = answer_text.strip()
    if not text:
        return []
    protected = _protect_abbreviation_periods(text)
    sentences = [_restore_abbreviation_periods(s) for s in SENT_SPLIT_RE.split(protected)]
    claims: list[Claim] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        cited = sorted({int(m.group(1)) for m in CITE_RE.finditer(s)})
        claims.append(Claim(text=s, cited=cited))
    return claims


def _protect_abbreviation_periods(text: str) -> str:
    text = text.replace("T.C.", f"T{DOT_SENTINEL}C{DOT_SENTINEL}")
    text = text.replace("A.Ş.", f"A{DOT_SENTINEL}Ş{DOT_SENTINEL}")
    text = text.replace("Ltd. Şti.", f"Ltd{DOT_SENTINEL} Şti{DOT_SENTINEL}")
    text = LEGAL_ARTICLE_RE.sub(lambda m: f"{m.group(1)}{DOT_SENTINEL}", text)
    return ABBREVIATION_RE.sub(lambda m: m.group(0).replace(".", DOT_SENTINEL), text)


def _restore_abbreviation_periods(text: str) -> str:
    return text.replace(DOT_SENTINEL, ".")
