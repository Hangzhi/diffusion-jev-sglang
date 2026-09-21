import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  ArrowRight,
  Braces,
  Check,
  ChevronDown,
  CircleHelp,
  Code2,
  Copy,
  Leaf,
  LoaderCircle,
  Play,
  Plus,
  RotateCcw,
  Sparkles,
  Terminal,
  X,
} from "lucide-react";
import "./style.css";
type Question = {
  type: "noul" | "choice" | "score";
  instructions: unknown;
  criteria?: unknown;
};
type Request = {
  model: string;
  state: unknown;
  questions: Record<string, Question>;
};
type Preset = {
  id: string;
  name: string;
  description: string;
  request: Request;
};
type Answer = {
  type: string;
  noul?: number;
  choice?: string;
  score?: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  legend?: Record<string, string>;
};
type Result = {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
  meta: { latency_ms: number; passes: number; temperature: number };
};
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
function App() {
  const [presets, setPresets] = useState<Preset[]>([]),
    [active, setActive] = useState("emoji");
  const [state, setState] = useState(""),
    [structured, setStructured] = useState(false),
    [questions, setQuestions] = useState<Record<string, Question>>({});
  const [ready, setReady] = useState(false),
    [busy, setBusy] = useState(false),
    [result, setResult] = useState<Result | null>(null),
    [error, setError] = useState("");
  const [tab, setTab] = useState("results"),
    [copied, setCopied] = useState(false),
    [editor, setEditor] = useState(false),
    [questionJson, setQuestionJson] = useState(""),
    [stale, setStale] = useState(false);
  const load = (p: Preset) => {
    setActive(p.id);
    setStructured(typeof p.request.state !== "string");
    setState(
      typeof p.request.state === "string"
        ? p.request.state
        : pretty(p.request.state),
    );
    setQuestions(p.request.questions);
    setResult(null);
    setError("");
    setStale(false);
  };
  useEffect(() => {
    fetch("/api/examples")
      .then((r) => r.json())
      .then((p) => {
        setPresets(p);
        load(p[0]);
      })
      .catch(() =>
        setError("Could not load examples. Check that the API is running."),
      );
    const poll = () =>
      fetch("/health")
        .then((r) => r.json())
        .then((h) => setReady(h.ready))
        .catch(() => setReady(false));
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);
  const request = (): Request => ({
    model: "diffusion-jev",
    state: structured ? JSON.parse(state) : state,
    questions,
  });
  const evaluate = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/v1/systemone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request()),
      });
      const body = await response.json();
      if (!response.ok)
        throw new Error(
          typeof body.detail === "string" ? body.detail : pretty(body.detail),
        );
      setResult(body);
      setTab("results");
      setStale(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setBusy(false);
    }
  };
  const copy = async () => {
    try {
      const data = JSON.stringify(request()).replaceAll("'", "'\\''");
      await navigator.clipboard.writeText(
        `curl '${window.location.origin}/v1/systemone' \\\n  -H 'Content-Type: application/json' \\\n  -d '${data}'`,
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      setError("Could not copy. Check JSON or browser clipboard permissions.");
    }
  };
  const [editorError, setEditorError] = useState("");
  const edit = () => {
    setEditorError("");
    setQuestionJson(pretty(questions));
    setEditor(true);
  };
  const save = () => {
    try {
      const q = JSON.parse(questionJson);
      if (
        !q ||
        Array.isArray(q) ||
        typeof q !== "object" ||
        !Object.keys(q).length
      )
        throw new Error("Questions must be a nonempty JSON object");
      for (const item of Object.values(q) as any[]) {
        if (
          !item ||
          typeof item !== "object" ||
          !["noul", "choice", "score"].includes(item.type) ||
          !("instructions" in item)
        )
          throw new Error("Each question needs a valid type and instructions.");
        if (
          item.type === "choice" &&
          (!item.criteria ||
            Array.isArray(item.criteria) ||
            typeof item.criteria !== "object")
        )
          throw new Error(
            "Choice criteria must be an option-to-description object.",
          );
        if (item.type === "score" && !Array.isArray(item.criteria))
          throw new Error("Score criteria must be an ordered array.");
      }
      setQuestions(q);
      setEditor(false);
      setStale(true);
      setError("");
    } catch (e) {
      setEditorError(e instanceof Error ? e.message : "Invalid JSON");
    }
  };
  return (
    <>
      <header>
        <a className="brand" href="/">
          <span className="brand-icon">
            <Leaf size={20} />
          </span>
          diffusion<span className="brand-light">/ jev</span>
          <span className="version">LAB 0.1</span>
        </a>
        <nav>
          <span className="nav-active">Playground</span>
          <a href="/docs" target="_blank" rel="noreferrer">
            API reference <ArrowUpRight size={14} />
          </a>
          <span className="local-tag">
            <span /> Self-hosted
          </span>
        </nav>
      </header>
      <main>
        <div className="eyebrow">
          <span /> SMALL OUTPUT. BIG DECISIONS.
        </div>
        <div className="intro">
          <div>
            <h1>
              Less generation.
              <br />
              <span>More decision.</span>
            </h1>
            <p>
              Turn context into choices, probabilities, and scores.
              <br />A local playground for diffusion-powered intelligence.
            </p>
          </div>
          <div className="model-card">
            <div className="model-symbol">
              <Sparkles size={22} />
            </div>
            <div>
              <small>POWERED BY</small>
              <strong>LLaDA 2.1 mini</strong>
              <span>SGLang · BF16 · diffusion</span>
            </div>
            <span className={"status " + (ready ? "online" : "")}>
              <i />
              {ready ? "Ready" : "Connecting"}
            </span>
          </div>
        </div>
        <div className="workspace-top">
          <span className="workspace-label">
            <Terminal size={16} /> Decision playground
          </span>
          <div className="preset-tabs">
            {presets.map((p) => (
              <button
                disabled={busy}
                className={active === p.id ? "selected" : ""}
                onClick={() => load(p)}
                key={p.id}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
        <div className="workspace">
          <section className="input-panel">
            <div className="panel-heading">
              <h2>
                <span className="step">01</span> Give it context
              </h2>
              <button
                className="icon-button"
                title="Reset example"
                aria-label="Reset example"
                disabled={busy}
                onClick={() => {
                  const p = presets.find((x) => x.id === active);
                  if (p) load(p);
                }}
              >
                <RotateCcw size={15} />
              </button>
            </div>
            <div className="field-label">
              <label htmlFor="context">STATE</label>
              <button
                className="text-button"
                onClick={() => {
                  setStructured(!structured);
                  setStale(true);
                }}
              >
                {structured ? "JSON object" : "Plain text"}{" "}
                <ChevronDown size={12} />
              </button>
            </div>
            <textarea
              id="context"
              disabled={busy}
              className="context"
              value={state}
              onChange={(e) => {
                setState(e.target.value);
                setStale(true);
              }}
              spellCheck="false"
            />
            <div className="field-hint">
              The text or structured data you want to evaluate.
              <span>{state.length} characters</span>
            </div>
            <div className="questions-heading">
              <h2>
                <span className="step">02</span> Ask your questions{" "}
                <span className="count">{Object.keys(questions).length}</span>
              </h2>
              <button className="text-button" disabled={busy} onClick={edit}>
                <Plus size={14} /> Edit questions
              </button>
            </div>
            <div className="question-list">
              {Object.entries(questions).map(([key, q]) => (
                <div className="question-card" key={key}>
                  <div className="question-card-top">
                    <strong>{key}</strong>
                    <span className={"type-badge " + q.type}>{q.type}</span>
                  </div>
                  <p>
                    {typeof q.instructions === "string"
                      ? q.instructions
                      : pretty(q.instructions)}
                  </p>
                  {q.type === "choice" && q.criteria != null && (
                    <div className="chips">
                      {Object.keys(q.criteria).map((k) => (
                        <span key={k}>{k}</span>
                      ))}
                    </div>
                  )}
                  {q.type === "score" && Array.isArray(q.criteria) && (
                    <div className="score-levels">
                      {q.criteria.map((v, i) => (
                        <span key={i}>
                          {i}
                          <small>{typeof v === "string" ? v : pretty(v)}</small>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="actions">
              <button
                className="evaluate"
                disabled={busy || !ready || !state.trim()}
                onClick={evaluate}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
                {busy ? "Evaluating…" : "Evaluate state"}
                <ArrowRight size={16} />
              </button>
              <button className="curl-button" onClick={copy}>
                {copied ? <Check size={14} /> : <Copy size={14} />}{" "}
                {copied ? "Copied" : "Copy cURL"}
              </button>
            </div>
            <div className="privacy">
              <span /> Context stays on your inference server.
            </div>
          </section>
          <section className="output-panel">
            <div className="output-top">
              <div className="result-tabs">
                <button
                  className={tab === "results" ? "active" : ""}
                  onClick={() => setTab("results")}
                >
                  <Sparkles size={14} /> Results
                </button>
                <button
                  className={tab === "json" ? "active" : ""}
                  onClick={() => setTab("json")}
                >
                  <Braces size={14} /> JSON
                </button>
              </div>
              {result && (
                <span className="latency">
                  {result.meta.latency_ms.toFixed(0)} ms
                </span>
              )}
            </div>
            {error && (
              <div role="alert" className="error">
                <strong>Could not evaluate</strong>
                <pre>{error}</pre>
                <button className="text-button" onClick={() => setError("")}>
                  Dismiss
                </button>
              </div>
            )}
            {stale && result && (
              <div className="stale">
                Inputs changed. Evaluate again to refresh these results.
              </div>
            )}
            {busy ? (
              <div className="empty">
                <LoaderCircle className="spin" size={32} />
                <h3>Reading between the tokens.</h3>
                <p>The model is evaluating your questions.</p>
              </div>
            ) : !result ? (
              <div className="empty">
                <div className="empty-art">
                  <div />
                  <span>
                    <Leaf size={32} />
                  </span>
                  <div />
                </div>
                <h3>
                  A little context.
                  <br />A clearer answer.
                </h3>
                <p>
                  Choose an example or bring your own.
                  <br />
                  Your model’s decisions will appear here.
                </p>
                <span className="empty-tip">
                  <ArrowRight size={13} /> Start with “Evaluate state”
                </span>
              </div>
            ) : tab === "json" ? (
              <pre className="json-output">{pretty(result)}</pre>
            ) : (
              <div className="results">
                {Object.entries(result.answers).map(([key, a]) => (
                  <article className="answer" key={key}>
                    <div className="answer-heading">
                      <strong>{key}</strong>
                      <span className={"type-badge " + a.type}>{a.type}</span>
                    </div>
                    {a.type === "noul" ? (
                      <>
                        <div className="answer-value">
                          {((a.noul || 0) * 100).toFixed(1)}
                          <small>% yes</small>
                        </div>
                        <div className="prob-bar large">
                          <span style={{ width: `${(a.noul || 0) * 100}%` }} />
                        </div>
                        <div className="bar-labels">
                          <span>No</span>
                          <span>Yes</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="answer-value">
                          {a.type === "choice" ? a.choice : a.score?.toFixed(2)}
                          {a.type === "score" && (
                            <small>
                              / {Object.keys(a.probabilities || {}).length - 1}
                            </small>
                          )}
                        </div>
                        {Object.entries(a.probabilities || {}).map(
                          ([label, p]) => (
                            <div className="probability" key={label}>
                              <div>
                                <span>{a.legend?.[label] || label}</span>
                                <span>{(p * 100).toFixed(1)}%</span>
                              </div>
                              <div className="prob-bar">
                                <span style={{ width: `${p * 100}%` }} />
                              </div>
                            </div>
                          ),
                        )}
                      </>
                    )}
                  </article>
                ))}
                <div className="result-foot">
                  <Check size={14} /> Real model logits · {result.meta.passes}{" "}
                  pass{result.meta.passes > 1 ? "es" : ""} · T=
                  {result.meta.temperature.toFixed(2)}
                </div>
              </div>
            )}
            <div className="output-note">
              <CircleHelp size={14} />
              <span>
                Probabilities are normalized over your candidate options.
                <br />A confident answer can still be wrong.
              </span>
            </div>
          </section>
        </div>
        <div className="bottom-notes">
          <span>
            <Code2 size={15} /> Three primitives. One endpoint.
          </span>
          <code>POST /v1/systemone</code>
          <a href="/docs">
            Build something with it <ArrowUpRight size={14} />
          </a>
        </div>
        <footer>
          <span>
            DIFFUSION / JEV <span className="footer-dot">·</span> An open
            research playground
          </span>
          <span>Local by design. Experimental by nature.</span>
        </footer>
      </main>
      {editor && (
        <div className="modal-backdrop">
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
          >
            <div className="panel-heading">
              <h2 id="modal-title">Edit typed questions</h2>
              <button
                className="icon-button"
                aria-label="Close editor"
                onClick={() => setEditor(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p>
              Use noul, choice, or score. Validation errors appear when you
              evaluate.
            </p>
            <textarea
              aria-label="Questions JSON"
              value={questionJson}
              onChange={(e) => setQuestionJson(e.target.value)}
              spellCheck="false"
            />
            {editorError && (
              <div role="alert" className="error">
                {editorError}
              </div>
            )}
            <button className="evaluate" onClick={save}>
              Save questions <Check size={16} />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
