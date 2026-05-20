"""Batch verifier — labels every claim in a single HF call."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Sequence

from ..generation import client, prompts as gen_prompts
from .claims import Claim, split_into_claims
from .risk import is_case_specific_advice

log = logging.getLogger(__name__)

VALID_STATUSES = {"supported", "partial", "unsupported", "insufficient", "risk"}

SYSTEM_VERIFIER_BATCH = (
    "Sen bir doğrulama (verifier) modülüsün. Bir hukuki cevaptan çıkarılmış "
    "BİRDEN FAZLA iddiayı, sana verilen KAYNAKLAR ile karşılaştırırsın. "
    "Çıkarımda bulunmadan, yalnızca verilen pasajların açıkça söylediği "
    "bilgilere dayanarak her iddia için karar verirsin. Etiketler:\n"
    "- supported: iddia bir veya birden fazla pasaj tarafından açıkça destekleniyor.\n"
    "- partial: iddia kısmen destekleniyor; bazı parçaları pasajlarda yok ya da fazla geniş.\n"
    "- unsupported: pasajlar iddiayı desteklemiyor.\n"
    "- insufficient: pasajlar bu konuyu hiç ele almıyor.\n"
    "Yanıtını YALNIZCA şu JSON biçiminde, hiçbir ek metin olmadan ver. "
    "Sırayı koru — her giriş için karşılığını döndür:\n"
    '{"verdicts": ['
    '{"index": <int>, "status": "<etiket>", "source_ids": [<int>...], "reason": "<kısa Türkçe gerekçe>"}'
    ", ... ]}"
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


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_batch_json(raw: str) -> list[dict]:
    m = _JSON_OBJ_RE.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    return verdicts if isinstance(verdicts, list) else []


def _coerce_verdict(claim: Claim, parsed: dict | None) -> ClaimVerdict:
    parsed = parsed or {}
    status = parsed.get("status", "unsupported")
    if status not in VALID_STATUSES:
        status = "unsupported"

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
    if not claims:
        return []

    src_block = gen_prompts.format_sources(passages)
    claims_block = "\n".join(
        f"{i+1}. {c.text_without_citations()}" for i, c in enumerate(claims)
    )
    user = (
        f"İDDİALAR:\n{claims_block}\n\n"
        f"KAYNAKLAR:\n{src_block}\n\n"
        "Her iddia için tek tek karar ver ve YALNIZCA JSON döndür."
    )
    try:
        raw = client.chat(
            [
                {"role": "system", "content": SYSTEM_VERIFIER_BATCH},
                {"role": "user", "content": user},
            ],
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully so UI shows partial info
        log.exception("Batch verifier failed: %s", exc)
        return [
            ClaimVerdict(
                text=c.text_without_citations(),
                status="insufficient",
                source_ids=[],
                reason="Verifier call failed.",
            )
            for c in claims
        ]

    parsed = _parse_batch_json(raw)
    by_index: dict[int, dict] = {}
    for v in parsed:
        if isinstance(v, dict):
            idx = v.get("index")
            if isinstance(idx, int) and 1 <= idx <= len(claims):
                by_index[idx] = v

    return [_coerce_verdict(c, by_index.get(i + 1)) for i, c in enumerate(claims)]
