"""Per-claim verifier — labels each claim against the retrieved passages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ..generation import client, prompts as gen_prompts
from .claims import Claim, split_into_claims
from .risk import is_case_specific_advice

VALID_STATUSES = {"supported", "partial", "unsupported", "insufficient", "risk"}

SYSTEM_VERIFIER = (
    "Sen bir doğrulama (verifier) modülüsün. Bir hukuki cevaptan çıkarılmış "
    "tek bir iddiayı, sana verilen KAYNAKLAR ile karşılaştırırsın. Çıkarımda "
    "bulunmadan, yalnızca verilen pasajların açıkça söylediği bilgilere "
    "dayanarak karar verirsin. Şu etiketlerden BİRİNİ döndürürsün:\n"
    "- supported: iddia bir veya birden fazla pasaj tarafından açıkça destekleniyor.\n"
    "- partial: iddia kısmen destekleniyor; bazı parçaları pasajlarda yok ya da fazla geniş.\n"
    "- unsupported: pasajlar iddiayı desteklemiyor (söylemiyor ya da reddediyor).\n"
    "- insufficient: pasajlar bu konuyu hiç ele almıyor.\n"
    "Yanıtını YALNIZCA aşağıdaki JSON biçiminde ver, başka hiçbir şey yazma:\n"
    "{\"status\": \"<etiket>\", \"source_ids\": [<int>...], \"reason\": \"<kısa Türkçe gerekçe>\"}"
)


@dataclass
class ClaimVerdict:
    text: str
    status: str
    source_ids: list[int]
    reason: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "status": self.status,
            "src": self.source_ids,
            "reason": self.reason,
        }


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verifier_json(raw: str) -> dict:
    m = _JSON_RE.search(raw or "")
    if not m:
        return {"status": "unsupported", "source_ids": [], "reason": "Verifier output not parseable."}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"status": "unsupported", "source_ids": [], "reason": "Verifier output not valid JSON."}


def verify_claim(
    claim: Claim,
    passages: Sequence[dict],
    model: str | None = None,
) -> ClaimVerdict:
    src_block = gen_prompts.format_sources(passages)
    user = (
        f"İDDİA:\n{claim.text_without_citations()}\n\n"
        f"KAYNAKLAR:\n{src_block}\n\n"
        "Yalnızca JSON döndür."
    )
    raw = client.chat(
        [
            {"role": "system", "content": SYSTEM_VERIFIER},
            {"role": "user", "content": user},
        ],
        model=model,
    )
    parsed = _parse_verifier_json(raw)
    status = parsed.get("status", "unsupported")
    if status not in VALID_STATUSES:
        status = "unsupported"

    # Risk gate: even if the model says "supported", flag obvious
    # case-specific imperatives / numeric amounts as `risk`.
    if status in {"supported", "partial"} and is_case_specific_advice(claim.text):
        status = "risk"

    source_ids = parsed.get("source_ids") or []
    try:
        source_ids = [int(s) for s in source_ids]
    except (TypeError, ValueError):
        source_ids = []

    return ClaimVerdict(
        text=claim.text_without_citations(),
        status=status,
        source_ids=source_ids,
        reason=str(parsed.get("reason", ""))[:300],
    )


def verify_answer(
    answer_text: str,
    passages: Sequence[dict],
    model: str | None = None,
) -> list[ClaimVerdict]:
    claims = split_into_claims(answer_text)
    return [verify_claim(c, passages, model=model) for c in claims]
