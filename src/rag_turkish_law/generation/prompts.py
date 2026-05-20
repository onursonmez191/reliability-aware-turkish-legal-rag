"""Prompt templates for answer generation and the LLM-only baseline.

All prompts are in Turkish so the model stays in-language. The grounded
prompt is strict about three things: cite-or-don't-claim, refuse-on-
insufficient-context, and never give case-specific legal advice.
"""

from __future__ import annotations

from typing import Sequence


SYSTEM_GROUNDED = (
    "Sen bir Türk hukuku alanında bilgi veren bir yardımcısın. "
    "GÖREVİN: Kullanıcının sorusunu, yalnızca aşağıda verilen 'KAYNAKLAR' "
    "bölümündeki pasajlara dayanarak Türkçe ve kısa biçimde yanıtlamak.\n"
    "KURALLAR:\n"
    "1) Sadece KAYNAKLAR'da yazan bilgileri kullan. Kaynaklarda olmayan bir "
    "iddiayı asla ekleme.\n"
    "2) Her cümlenin sonunda, o cümleyi destekleyen kaynak numarasını köşeli "
    "parantezle belirt: [1], [2], gerekirse [1][3].\n"
    "3) Kaynaklar soruyu yeterince kapsamıyorsa: 'Mevcut kaynaklar bu soruyu "
    "yeterince kapsamıyor.' diyerek açıkça söyle ve uydurma yapma.\n"
    "4) Davalık, somut tazminat tutarı, kesin süre veya 'şunu yap' tarzında "
    "vaka-özel hukuki tavsiye verme. Genel kuralları açıkla ve gerekirse "
    "kullanıcıyı bir avukata yönlendir.\n"
    "5) Avukat değilsin; kendini avukat olarak sunma.\n"
)

SYSTEM_LLM_ONLY = (
    "Sen bir Türk hukuku konusunda bilgili bir asistansın. Kullanıcının "
    "sorusunu Türkçe ve kısa biçimde yanıtla. (Bu yanıt herhangi bir kaynağa "
    "dayanmamaktadır; yalnızca temel modelin bildiklerinden üretilmiştir.)"
)


def format_sources(passages: Sequence[dict]) -> str:
    """Format passages as a numbered source block.

    `passages` items must have `snippet` (or `text`) and `title` keys.
    Numbering starts at 1 to match the citation convention in the prompt.
    """
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "").strip()
        snippet = (p.get("snippet") or p.get("text") or "").strip()
        if title:
            lines.append(f"[{i}] {title}\n{snippet}")
        else:
            lines.append(f"[{i}] {snippet}")
    return "\n\n".join(lines)


def build_grounded_messages(question: str, passages: Sequence[dict]) -> list[dict]:
    src_block = format_sources(passages)
    user = (
        f"SORU:\n{question}\n\n"
        f"KAYNAKLAR:\n{src_block}\n\n"
        "Yukarıdaki kurallara uyarak yanıtla. Her cümlede kaynak numarası belirt."
    )
    return [
        {"role": "system", "content": SYSTEM_GROUNDED},
        {"role": "user", "content": user},
    ]


def build_llm_only_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_LLM_ONLY},
        {"role": "user", "content": question},
    ]
