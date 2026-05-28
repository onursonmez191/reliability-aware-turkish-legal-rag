"""Batch verifier — labels every claim in a single HF call."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from ..config import load_config
from ..generation import client, prompts as gen_prompts
from .claims import Claim, split_into_claims
from .risk import is_case_specific_advice

log = logging.getLogger(__name__)

VALID_STATUSES = {"supported", "partial", "unsupported", "insufficient", "risk", "error"}

SYSTEM_VERIFIER_BATCH = (
    "Sen bir doğrulama (verifier) modülüsün. Bir hukuki cevaptan çıkarılmış "
    "BİRDEN FAZLA iddiayı, sana verilen KAYNAKLAR ile karşılaştırırsın. "
    "Çıkarımda bulunmadan, yalnızca verilen pasajların açıkça söylediği "
    "bilgilere dayanarak her iddia için karar verirsin. Etiketler:\n"
    "- supported: iddia bir veya birden fazla pasaj tarafından açıkça destekleniyor.\n"
    "- partial: iddia kısmen destekleniyor; bazı parçaları pasajlarda yok ya da fazla geniş.\n"
    "- unsupported: pasajlar iddiayı desteklemiyor.\n"
    "- insufficient: pasajlar bu konuyu hiç ele almıyor.\n"
    "Dar ve açık terminoloji eşleştirmelerine izin ver: köpek/kedi gibi ifadeler "
    "hayvan kapsamındadır; 'giderim' ve 'tazminat' aynı zarar karşılama bağlamında "
    "kullanılabilir; 'hayvan bulunduran' ifadesi hayvanın bakımını/yönetimini "
    "üstlenen kişiyi anlatır. Ancak belediye, sahipsiz hayvan, kesin tazminat "
    "miktarı veya dava yolu gibi ek iddiaları kaynaklarda açık dayanak yoksa "
    "supported yapma.\n"
    "Eğer iddia cevapta kaynak numarasıyla verilmişse, source_ids alanında "
    "öncelikle o kaynak numaralarından açıkça destekleyenleri döndür. "
    "Kaynak numarası yoksa veya verilen kaynak iddiayı desteklemiyorsa "
    "supported verme.\n"
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
    source_ids: list[int] = field(default_factory=list)
    reason: str = ""
    cited: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "status": self.status,
            "src": self.source_ids,
            "cited": self.cited,
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


def _downgrade_for_citation_mismatch(status: str) -> str:
    if status == "supported":
        return "partial"
    if status == "partial":
        return "unsupported"
    return status


def _coerce_verdict(claim: Claim, parsed: dict | None, *, num_sources: int) -> ClaimVerdict:
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
    source_ids = [s for s in source_ids if 1 <= s <= num_sources]

    reason = str(parsed.get("reason", ""))[:300]
    if status in {"supported", "partial"}:
        if not source_ids:
            status = _downgrade_for_citation_mismatch(status)
            reason = (reason + " Kaynak iddiası doğrulanamadı.").strip()
        elif not claim.cited:
            status = _downgrade_for_citation_mismatch(status)
            reason = (reason + " Cevapta kaynak numarası yok.").strip()
        elif not set(source_ids).intersection(claim.cited):
            status = _downgrade_for_citation_mismatch(status)
            reason = (reason + " Doğrulayıcı kaynakları cevapta verilen atıflarla örtüşmüyor.").strip()

    return ClaimVerdict(
        text=claim.text_without_citations(),
        status=status,
        source_ids=source_ids,
        reason=reason[:300],
        cited=claim.cited,
    )


def verify_answer(
    answer_text: str,
    passages: Sequence[dict],
    model: str | None = None,
) -> list[ClaimVerdict]:
    cfg = load_config()
    claims = split_into_claims(answer_text)
    if not claims:
        return []

    src_block = gen_prompts.format_sources(passages)
    claims_block = "\n".join(
        f"{i+1}. {c.text_without_citations()}\n"
        f"   Cevapta kullanılan kaynak numaraları: {c.cited or []}"
        for i, c in enumerate(claims)
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
            model=model or cfg.verification.hf_model,
            max_new_tokens=cfg.verification.max_new_tokens,
            temperature=cfg.verification.temperature,
        )
    except Exception as exc:  # noqa: BLE001 — degrade gracefully so UI shows partial info
        log.exception("Batch verifier failed: %s", exc)
        return [
            ClaimVerdict(
                text=c.text_without_citations(),
                status="error",
                source_ids=[],
                reason="Verifier call failed.",
                cited=c.cited,
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

    return [
        _coerce_verdict(c, by_index.get(i + 1), num_sources=len(passages))
        for i, c in enumerate(claims)
    ]
