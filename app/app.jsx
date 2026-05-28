const { useState, useEffect, useRef, useMemo } = React;

// ---------- atoms ----------

const VerdictIcon = ({ k, size = 14 }) => {
  const common = { width: size, height: size, viewBox: "0 0 16 16", fill: "none", strokeWidth: 1.6, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round" };
  if (k === "supported")
    return (
      <svg {...common}>
        <path d="M3 8.5l3.2 3.2L13 5"></path>
      </svg>
    );
  if (k === "partial")
    return (
      <svg {...common}>
        <circle cx="8" cy="8" r="6"></circle>
        <path d="M8 2v6h6"></path>
      </svg>
    );
  if (k === "unsupported")
    return (
      <svg {...common}>
        <path d="M4 4l8 8M12 4l-8 8"></path>
      </svg>
    );
  if (k === "insufficient")
    return (
      <svg {...common}>
        <circle cx="8" cy="8" r="6"></circle>
        <path d="M8 5v3.5M8 11.2v.1"></path>
      </svg>
    );
  if (k === "risk")
    return (
      <svg {...common}>
        <path d="M8 2l6.5 11h-13z"></path>
        <path d="M8 7v3M8 12v.1"></path>
      </svg>
    );
  if (k === "error")
    return (
      <svg {...common}>
        <path d="M4 4l8 8M12 4l-8 8"></path>
        <circle cx="8" cy="8" r="6"></circle>
      </svg>
    );
  return null;
};

const VERDICT_THEME = {
  supported:    { label: "Supported",           en: "high confidence",   color: "var(--ok)",    bg: "var(--ok-bg)",   ring: "var(--ok-ring)"   },
  partial:      { label: "Partially supported", en: "some unsupported",  color: "var(--warn)",  bg: "var(--warn-bg)", ring: "var(--warn-ring)" },
  unsupported:  { label: "Unsupported",         en: "no evidence",       color: "var(--bad)",   bg: "var(--bad-bg)",  ring: "var(--bad-ring)"  },
  insufficient: { label: "Insufficient context",en: "low coverage",      color: "var(--ink-2)", bg: "var(--ink-bg)",  ring: "var(--ink-ring)"  },
  risk:         { label: "Legal-advice risk",   en: "case-specific",     color: "var(--bad)",   bg: "var(--bad-bg)",  ring: "var(--bad-ring)"  },
  error:        { label: "Verifier error",      en: "check backend",      color: "var(--bad)",   bg: "var(--bad-bg)",  ring: "var(--bad-ring)"  },
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// ---------- header ----------

const Header = ({ mode, setMode, k, setK }) => (
  <header className="header" data-screen-label="App Header">
    <div className="brand">
      <div className="seal" aria-hidden="true">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <circle cx="14" cy="14" r="13" stroke="currentColor" strokeWidth="1"/>
          <path d="M14 6v16M8 10l6 12 6-12M5 10h18" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
        </svg>
      </div>
      <div className="brand-text">
        <div className="brand-title">Lex · RAG</div>
        <div className="brand-sub">Reliability-Aware Turkish Legal Question Answering</div>
      </div>
    </div>
    <nav className="mode-switch" role="tablist" aria-label="Pipeline modu">
      {["llm", "rag", "verified"].map((m) => (
        <button
          key={m}
          role="tab"
          aria-selected={mode === m}
          className={`mode-tab ${mode === m ? "active" : ""}`}
          onClick={() => setMode(m)}
        >
          <span className="mode-key">{m === "llm" ? "A" : m === "rag" ? "B" : "C"}</span>
          <span className="mode-label">{m === "llm" ? "LLM-only" : m === "rag" ? "RAG" : "RAG + Verifier"}</span>
        </button>
      ))}
    </nav>
    <div className="header-meta">
      <div className="meta-row">
        <span className="meta-key">corpus</span>
        <span className="meta-val">turkish_law_qa · 18,305</span>
      </div>
      <div className="meta-row">
        <span className="meta-key">top-k</span>
        <div className="k-pill" role="group" aria-label="top-k">
          {[3, 5, 8].map((n) => (
            <button key={n} className={`k ${k === n ? "on" : ""}`} onClick={() => setK(n)}>{n}</button>
          ))}
        </div>
      </div>
    </div>
  </header>
);

// ---------- composer ----------

const Composer = ({ value, setValue, onAsk, busy, suggestions, onPick }) => {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = Math.min(ref.current.scrollHeight, 220) + "px";
    }
  }, [value]);
  return (
    <section className="composer-block" data-screen-label="Composer">
      <div className="composer-label">
        <span className="composer-num">01</span>
        <span>Ask a legal question</span>
        <span className="composer-lang">EN</span>
      </div>
      <div className={`composer ${busy ? "busy" : ""}`}>
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Örnek: Kira sözleşmesi süresi dolmadan kiracı çıkabilir mi?"
          rows={2}
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onAsk();
          }}
        />
        <div className="composer-foot">
          <div className="composer-hint">
            <kbd>⌘</kbd><kbd>↵</kbd>&nbsp;to ask
          </div>
          <button className="ask-btn" onClick={onAsk} disabled={busy || !value.trim()}>
            {busy ? "Working…" : "Ask"}
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <path d="M3 8h10M9 4l4 4-4 4"/>
            </svg>
          </button>
        </div>
      </div>
      <div className="suggestion-row">
        <span className="sug-label">Example questions</span>
        <div className="sug-chips">
          {suggestions.map((s) => (
            <button key={s.id} className="chip" onClick={() => onPick(s)} disabled={busy}>
              <span className="chip-dot" data-kind={s.kind}></span>
              {s.q}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};

// ---------- pipeline strip ----------

const Pipeline = ({ active, mode }) => {
  const steps = window.PIPELINE_STEPS.filter((s) => {
    if (mode === "llm") return s.id === "generate";
    if (mode === "rag") return s.id !== "verify";
    return true;
  });
  return (
    <div className="pipeline" aria-label="Pipeline">
      {steps.map((s, i) => {
        const state =
          active === null ? "idle" :
          active === "done" ? "done" :
          active === s.id ? "running" :
          steps.findIndex((x) => x.id === active) > i ? "done" : "idle";
        return (
          <div key={s.id} className={`pstep ${state}`}>
            <div className="pstep-mark">
              {state === "running" ? <span className="dot-pulse"></span> :
               state === "done" ? <VerdictIcon k="supported" size={11}/> :
               <span className="dot-empty"></span>}
            </div>
            <div className="pstep-body">
              <div className="pstep-label">{s.label}</div>
              <div className="pstep-sub">{s.sub}</div>
            </div>
            {i < steps.length - 1 && <div className="pstep-line"></div>}
          </div>
        );
      })}
    </div>
  );
};

// ---------- answer ----------

const Answer = ({ text, onCite, mode, hovered, setHovered }) => {
  const parts = useMemo(() => text.split(/(\[\d+\])/g), [text]);
  return (
    <div className="answer-text">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (m) {
          const n = parseInt(m[1], 10);
          return (
            <button
              key={i}
              className={`cite ${hovered === n ? "hot" : ""}`}
              onMouseEnter={() => setHovered(n)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onCite(n)}
            >
              {n}
            </button>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </div>
  );
};

// ---------- verdict card ----------

const VerdictCard = ({ v, mode }) => {
  if (mode !== "verified" || !v) return null;
  const t = VERDICT_THEME[v.key] || VERDICT_THEME.partial;
  const supportedCount = v.claims.filter((c) => c.status === "supported").length;
  return (
    <section className="card verdict-card" style={{ "--vc": t.color, "--vbg": t.bg, "--vring": t.ring }} data-screen-label="Verdict">
      <header className="card-head">
        <div className="card-num">03</div>
        <div className="card-title">Güvenilirlik Analizi</div>
        <div className="card-sub">verifier · claim-level</div>
      </header>
      <div className="verdict-banner">
        <div className="verdict-icon">
          <VerdictIcon k={v.key} size={18}/>
        </div>
        <div className="verdict-text">
          <div className="verdict-label">{t.label}</div>
          <div className="verdict-en">{t.en}</div>
        </div>
        <div className="verdict-score">
          <div className="score-bar">
            <div className="score-fill" style={{ width: `${clamp(Math.round(v.score * 100), 0, 100)}%` }}></div>
          </div>
          <div className="score-num">{v.score.toFixed(2)}</div>
        </div>
      </div>
      <div className="claims">
        <div className="claims-head">
          <span>Claims in the answer</span>
          <span>{supportedCount}/{v.claims.length} supported</span>
        </div>
        <ul className="claim-list">
          {v.claims.map((c, i) => {
            const cT = VERDICT_THEME[c.status === "risk" ? "risk" : c.status] || VERDICT_THEME.partial;
            return (
              <li key={i} className="claim" style={{ "--cc": cT.color, "--cbg": cT.bg }}>
                <div className="claim-status">
                  <VerdictIcon k={c.status === "risk" ? "risk" : c.status} size={12}/>
                </div>
                <div className="claim-text">{c.text}</div>
                <div className="claim-src">
                  {c.src.length === 0 ? <span className="no-src">—</span> :
                    c.src.map((s) => <span key={s} className="claim-src-pill">[{s}]</span>)}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
      {(v.key === "risk" || v.key === "insufficient" || v.risk === "high") && (
        <div className="risk-banner">
          <VerdictIcon k="risk" size={14}/>
          <span>
            {v.key === "error"
              ? "The verifier failed, so this answer was not reliability-checked."
              : v.key === "risk"
                ? "This answer may contain case-specific legal advice. Consult a qualified lawyer."
                : "Available sources don't cover this situation well; verify findings with an expert."}
          </span>
        </div>
      )}
    </section>
  );
};

// ---------- sources list ----------

const Sources = ({ items, hovered, setHovered, k }) => {
  const visible = items.slice(0, k);
  const scores = visible.map((s) => Number(s.score)).filter(Number.isFinite);
  const allCosineLike = scores.length > 0 && scores.every((s) => s >= 0 && s <= 1);
  const minScore = scores.length ? Math.min(...scores) : 0;
  const maxScore = scores.length ? Math.max(...scores) : 1;
  const scoreWidth = (raw) => {
    const score = Number(raw);
    if (!Number.isFinite(score)) return 0;
    if (allCosineLike) return clamp(score * 100, 0, 100);
    if (maxScore === minScore) return 100;
    return clamp(((score - minScore) / (maxScore - minScore)) * 100, 0, 100);
  };

  return (
    <section className="card sources-card" data-screen-label="Sources">
      <header className="card-head">
        <div className="card-num">{"04"}</div>
        <div className="card-title">Retrieved Sources</div>
        <div className="card-sub">retriever · top-{k}</div>
      </header>
      <ul className="source-list">
        {visible.map((s, i) => {
        const n = i + 1;
        const score = Number(s.score);
        const scoreText = Number.isFinite(score) ? score.toFixed(2) : "—";
        return (
          <li
            key={s.id}
            className={`source ${hovered === n ? "hot" : ""}`}
            onMouseEnter={() => setHovered(n)}
            onMouseLeave={() => setHovered(null)}
          >
            <div className="source-rank">
              <div className="rank-num">{n}</div>
              <div className="rank-id">{s.id}</div>
            </div>
            <div className="source-body">
              <div className="source-title-row">
                <div className="source-title">{s.title}</div>
                <div className="source-tag">{s.tag}</div>
              </div>
              <div className="source-snippet">"{s.snippet}"</div>
              <div className="source-meta">
                <div className="sim">
                  <span className="sim-label">score</span>
                  <div className="sim-bar"><div className="sim-fill" style={{ width: `${scoreWidth(s.score)}%` }}></div></div>
                  <span className="sim-num">{scoreText}</span>
                </div>
              </div>
            </div>
          </li>
        );
        })}
      </ul>
    </section>
  );
};

// ---------- comparison ----------

const Comparison = ({ mode, a }) => {
  if (mode !== "verified") return null;
  return (
    <section className="card compare-card" data-screen-label="Comparison">
      <header className="card-head">
        <div className="card-num">05</div>
        <div className="card-title">Three-Pipeline Comparison</div>
        <div className="card-sub">ablation</div>
      </header>
      <div className="compare-grid">
        <div className="compare-col">
          <div className="compare-head"><span className="cmp-k">A</span> LLM-only</div>
          <div className="compare-body">{a.llmOnly}</div>
          <div className="compare-foot bad-foot">
            <VerdictIcon k="risk" size={11}/> no sources · unverified
          </div>
        </div>
        <div className="compare-col">
          <div className="compare-head"><span className="cmp-k">B</span> RAG</div>
          <div className="compare-body">{a.answer.replace(/\s*\[\d+\]/g, "")}</div>
          <div className="compare-foot warn-foot">
            <VerdictIcon k="partial" size={11}/> grounded · no verifier
          </div>
        </div>
        <div className="compare-col active">
          <div className="compare-head"><span className="cmp-k">C</span> RAG + Verifier</div>
          <div className="compare-body">{a.answer}</div>
          <div className="compare-foot ok-foot">
            <VerdictIcon k="supported" size={11}/> grounded · claim-by-claim verified
          </div>
        </div>
      </div>
    </section>
  );
};

// ---------- empty state ----------

const EmptyState = () => (
  <div className="empty">
    <div className="empty-mark">
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="22" stroke="currentColor" strokeWidth="0.8" opacity="0.3"/>
        <path d="M24 12v24M14 18l10 18 10-18M9 18h30" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" opacity="0.6"/>
      </svg>
    </div>
    <div className="empty-title">Start by asking a legal question</div>
    <div className="empty-body">
      The system searches a Turkish legal QA corpus for your question, generates a source-grounded
      answer, and verifies each claim against the retrieved passages. Outputs are informational
      and do not constitute legal advice.
    </div>
  </div>
);

// ---------- disclaimer ----------

const Disclaimer = () => (
  <div className="disclaimer" role="note">
    <div className="disc-mark">!</div>
    <div className="disc-text">
      <b>For educational and informational purposes only.</b> This system does not provide
      professional legal advice. For a concrete legal matter, consult a qualified attorney.
    </div>
  </div>
);

// ---------- main app ----------

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "mode": "verified",
  "topK": 8,
  "showPipeline": true,
  "showComparison": true,
  "density": "comfortable"
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [value, setValue] = useState("");
  const [activeQ, setActiveQ] = useState(null);
  const [activeAnswer, setActiveAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(null);
  const [hoveredCite, setHoveredCite] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const mode = t.mode;
  const setMode = (m) => {
    setTweak("mode", m);
    setActiveAnswer(null);
    setActiveQ(null);
    setPipelineStep(null);
    setErrorMsg(null);
  };
  const k = t.topK;
  const setK = (n) => setTweak("topK", n);

  const ask = async (q) => {
    setBusy(true);
    setActiveQ(null);
    setActiveAnswer(null);
    setErrorMsg(null);

    const steps = mode === "llm"
      ? ["generate"]
      : mode === "rag"
      ? ["embed", "retrieve", "rerank", "generate"]
      : ["embed", "retrieve", "rerank", "generate", "verify"];

    // Visual pipeline animation while the request is in flight.
    const animation = (async () => {
      for (const s of steps) {
        setPipelineStep(s);
        await new Promise((r) => setTimeout(r, 400));
      }
    })();

    try {
      const mainPromise = window.postAsk({ question: q.q, mode, k });
      const wantLlmCompare = mode === "verified" && t.showComparison;
      const llmPromise = wantLlmCompare
        ? window.postAsk({ question: q.q, mode: "llm", k }).catch(() => null)
        : Promise.resolve(null);

      const [mainResp, llmResp] = await Promise.all([mainPromise, llmPromise]);
      await animation;

      setPipelineStep("done");
      setActiveQ(q);
      setActiveAnswer(window.normalizeAnswerPayload(mainResp, llmResp));
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || String(err));
      setPipelineStep(null);
    } finally {
      setBusy(false);
    }
  };

  const onAsk = () => {
    if (!value.trim()) return;
    const match = window.SAMPLE_QUESTIONS.find((s) => s.q.trim() === value.trim());
    ask(match || { id: "custom", q: value.trim(), kind: "supported" });
  };

  const onPick = (s) => {
    setValue(s.q);
    ask(s);
  };

  const a = activeAnswer;

  return (
    <div className={`app density-${t.density}`}>
      <Header mode={mode} setMode={setMode} k={k} setK={setK}/>
      <Disclaimer/>

      <main className="main">
        <div className="col col-main">
          <Composer
            value={value}
            setValue={setValue}
            onAsk={onAsk}
            busy={busy}
            suggestions={window.SAMPLE_QUESTIONS}
            onPick={onPick}
          />

          {t.showPipeline && (
            <section className="pipeline-block">
              <div className="block-label">
                <span className="block-num">02</span>
                <span>Pipeline</span>
                <span className="block-meta">{mode === "llm" ? "1 step" : mode === "rag" ? "4 steps" : "5 steps"}</span>
              </div>
              <Pipeline active={pipelineStep} mode={mode}/>
            </section>
          )}

          {a ? (
            <section className="card answer-card" data-screen-label="Answer">
              <header className="card-head">
                <div className="card-num">03</div>
                <div className="card-title">Answer</div>
                <div className="card-sub">
                  {mode === "llm" ? "LLM-only · ungrounded" : mode === "rag" ? "RAG · grounded" : "RAG + Verifier · verified"}
                </div>
              </header>
              <div className="question-recap">
                <span className="qr-label">Question</span>
                <span className="qr-text">{activeQ.q}</span>
              </div>
              {mode === "llm" ? (
                <div className="answer-text plain">{a.llmOnly}</div>
              ) : (
                <Answer text={a.answer} onCite={() => {}} mode={mode} hovered={hoveredCite} setHovered={setHoveredCite}/>
              )}
              {mode === "llm" && (
                <div className="answer-warn">
                  <VerdictIcon k="risk" size={12}/>
                  This answer is not grounded in any source and has not been verified.
                </div>
              )}
            </section>
          ) : errorMsg ? (
            <div
              role="alert"
              style={{
                padding: "16px 20px",
                border: "1px solid var(--bad-ring)",
                background: "var(--bad-bg)",
                color: "var(--bad)",
                borderRadius: 10,
                marginTop: 16,
                fontSize: 14,
                lineHeight: 1.5,
              }}
            >
              <b>Backend hatası:</b> {errorMsg}
              <div style={{ marginTop: 8, opacity: 0.85 }}>
                <code>scripts/build_index.py</code> ile indeksi oluşturduğunuzu ve
                <code> scripts/serve.py</code> ile sunucunun çalıştığını doğrulayın.
              </div>
            </div>
          ) : (
            <EmptyState/>
          )}
        </div>

        <aside className="col col-side">
          {a && mode === "verified" && a.verdict && <VerdictCard v={a.verdict} mode={mode}/>}
          {a && mode !== "llm" && (
            <Sources items={a.sources} hovered={hoveredCite} setHovered={setHoveredCite} k={k}/>
          )}
          {!a && (
            <div className="side-empty">
              <div className="side-empty-title">Sources & verification</div>
              <div className="side-empty-body">
                Once you ask a question, retrieved passages, similarity scores, and the verifier's
                claim-level analysis will appear here.
              </div>
              <div className="side-legend">
                {["supported","partial","unsupported","insufficient","risk"].map((k) => (
                  <div className="legend-row" key={k}>
                    <span className="legend-icon" data-k={k}><VerdictIcon k={k} size={11}/></span>
                    <span className="legend-name">{VERDICT_THEME[k].label}</span>
                    <span className="legend-en">{VERDICT_THEME[k].en}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>
      </main>

      {a && t.showComparison && <Comparison mode={mode} a={a}/>}

      <footer className="footer">
        <div className="footer-left">
          <span className="foot-k">CS 455 · Spring 2025/2026</span>
          <span className="foot-v">Sönmez · Sözer · Erşeker</span>
        </div>
        <div className="footer-right">
          <span>Reliability-Aware RAG for Turkish Legal QA</span>
          <span className="foot-dot">·</span>
          <span>v0.1 prototype</span>
        </div>
      </footer>

      <TweaksPanel title="Tweaks">
        <TweakSection label="Pipeline">
          <TweakRadio label="Mode" value={t.mode} onChange={(v) => setTweak("mode", v)}
            options={[
              { value: "llm", label: "LLM" },
              { value: "rag", label: "RAG" },
              { value: "verified", label: "Verified" },
            ]}/>
          <TweakRadio label="Top-k" value={String(t.topK)} onChange={(v) => setTweak("topK", parseInt(v, 10))}
            options={[
              { value: "3", label: "3" },
              { value: "5", label: "5" },
              { value: "8", label: "8" },
            ]}/>
        </TweakSection>
        <TweakSection label="View">
          <TweakToggle label="Pipeline strip" value={t.showPipeline} onChange={(v) => setTweak("showPipeline", v)}/>
          <TweakToggle label="Comparison panel" value={t.showComparison} onChange={(v) => setTweak("showComparison", v)}/>
          <TweakRadio label="Density" value={t.density} onChange={(v) => setTweak("density", v)}
            options={[
              { value: "comfortable", label: "Comfortable" },
              { value: "compact", label: "Compact" },
            ]}/>
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
