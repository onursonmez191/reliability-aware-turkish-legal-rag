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


@dataclass
class Claim:
    text: str
    cited: list[int]

    def text_without_citations(self) -> str:
        return CITE_RE.sub("", self.text).replace("  ", " ").strip()

    def to_dict(self) -> dict:
        return {"text": self.text_without_citations(), "cited": self.cited}


def split_into_claims(answer_text: str) -> list[Claim]:
    text = answer_text.strip()
    if not text:
        return []
    sentences = SENT_SPLIT_RE.split(text)
    claims: list[Claim] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        cited = sorted({int(m.group(1)) for m in CITE_RE.finditer(s)})
        claims.append(Claim(text=s, cited=cited))
    return claims
