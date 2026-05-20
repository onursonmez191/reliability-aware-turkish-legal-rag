// Frontend data wiring.
// Example questions live here. Real answers come from the FastAPI backend
// (POST /api/ask) — there is no mock ANSWERS object anymore.

const SAMPLE_QUESTIONS = [
  {
    id: "q1",
    q: "Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?",
    kind: "supported",
  },
  {
    id: "q2",
    q: "İşten haksız yere çıkarıldığımı düşünüyorum — ne yapabilirim?",
    kind: "partial",
  },
  {
    id: "q3",
    q: "Komşumun köpeği beni ısırdı — kimden tazminat isteyebilirim?",
    kind: "risk",
  },
  {
    id: "q4",
    q: "Kardeşim, vefat eden annemizin dairesinin tapusunu devretmiyor. Ne yapmalıyım?",
    kind: "insufficient",
  },
];

const PIPELINE_STEPS = [
  { id: "embed", label: "Sorguyu vektörle", sub: "multilingual-e5-base" },
  { id: "retrieve", label: "Vektör arama", sub: "FAISS · top-k" },
  { id: "rerank", label: "Yeniden sırala", sub: "cross-encoder" },
  { id: "generate", label: "Cevap üret", sub: "kaynak-temelli" },
  { id: "verify", label: "Doğrulayıcı", sub: "iddia bazlı kontrol" },
];

async function postAsk({ question, mode, k }) {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, k }),
  });
  if (!resp.ok) {
    let detail = "";
    try {
      const j = await resp.json();
      detail = j.detail || JSON.stringify(j);
    } catch {
      detail = await resp.text();
    }
    throw new Error(`API ${resp.status}: ${detail}`);
  }
  return resp.json();
}

function normalizeAnswerPayload(mainResp, llmResp) {
  return {
    answer: mainResp.answer || "",
    llmOnly: (llmResp && llmResp.answer) || mainResp.llm_only || "",
    sources: (mainResp.sources || []).map((s) => ({
      id: s.id,
      title: s.title,
      snippet: s.snippet,
      score: s.score,
      tag: s.tag,
    })),
    verdict: mainResp.verdict
      ? {
          key: mainResp.verdict.key,
          score: mainResp.verdict.score,
          risk: mainResp.verdict.risk,
          claims: mainResp.verdict.claims,
          label: mainResp.verdict.key,
        }
      : null,
    timings: mainResp.timings || [],
  };
}

window.SAMPLE_QUESTIONS = SAMPLE_QUESTIONS;
window.PIPELINE_STEPS = PIPELINE_STEPS;
window.postAsk = postAsk;
window.normalizeAnswerPayload = normalizeAnswerPayload;
