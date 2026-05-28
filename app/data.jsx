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
  { id: "confidence", label: "Güven eşiği", sub: "retrieval gate" },
  { id: "generate", label: "Cevap üret", sub: "kaynak-temelli" },
  { id: "verify", label: "Doğrulayıcı", sub: "iddia bazlı kontrol" },
];

async function postAsk({ question, mode, k, model }) {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, k, model }),
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

async function postAskStream({ question, mode, k, model, onEvent, signal }) {
  const resp = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, k, model }),
    signal,
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
  if (!resp.body) {
    const fallback = await postAsk({ question, mode, k, model });
    onEvent("final", fallback);
    return fallback;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalPayload = null;
  let chunkFramesSincePaint = 0;
  let paintedFirstChunk = false;

  const nextPaint = () => new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => setTimeout(resolve, 0));
    } else {
      setTimeout(resolve, 0);
    }
  });

  const consumeFrame = async (frame) => {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const data = JSON.parse(dataLines.join("\n"));
    onEvent(event, data);
    if (event === "final") finalPayload = data;
    if (event === "error") throw new Error(data.message || "Streaming error");
    if (event === "chunk") {
      chunkFramesSincePaint += 1;
      if (!paintedFirstChunk || chunkFramesSincePaint >= 4) {
        paintedFirstChunk = true;
        chunkFramesSincePaint = 0;
        await nextPaint();
      }
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\n\n/);
    buffer = frames.pop() || "";
    for (const frame of frames) {
      if (frame.trim()) await consumeFrame(frame);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) await consumeFrame(buffer);
  return finalPayload;
}

async function fetchModels() {
  const resp = await fetch("/api/models");
  if (!resp.ok) {
    throw new Error(`API ${resp.status}: ${await resp.text()}`);
  }
  return resp.json();
}

async function postModelAction(action, model) {
  const resp = await fetch(`/api/models/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
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
window.postAskStream = postAskStream;
window.fetchModels = fetchModels;
window.postModelAction = postModelAction;
window.normalizeAnswerPayload = normalizeAnswerPayload;
