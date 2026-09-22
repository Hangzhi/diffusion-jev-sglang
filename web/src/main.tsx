import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  ArrowRight,
  Braces,
  Check,
  ChevronDown,
  CircleHelp,
  Code2,
  Leaf,
  Image as ImageIcon,
  LoaderCircle,
  Play,
  Pencil,
  Plus,
  RotateCcw,
  Sparkles,
  Terminal,
  Upload,
  X,
} from "lucide-react";
import "./style.css";
import { DrawPad } from "./DrawPad";
type Question = {
  type: "noul" | "choice" | "score";
  instructions: unknown;
  criteria?: unknown;
};
type Request = {
  model: string;
  state: unknown;
  questions: Record<string, Question>;
  images?: string[];
};
type GalleryImage = { id: string; label: string; split: string; benchmark: boolean };
type SelectedImage = { input: string; url: string; label?: string };
const flowerQuestions: Record<string, Question> = {
  flower: { type: "choice", instructions: "Which type of flower is most prominent in the attached image?",
    criteria: { daisy: "A daisy flower", dandelion: "A dandelion flower or seed head", rose: "A rose flower", sunflower: "A sunflower", tulip: "A tulip flower" } },
};
const doodleQuestions: Record<string, Question> = {
  doodle: { type: "choice", instructions: "What object does the attached hand-drawn sketch depict? Choose the closest category.",
    criteria: {
      airplane: "An airplane with wings", apple: "An apple fruit",
      bicycle: "A bicycle with two wheels", cat: "A cat",
      clock: "A clock face with hands", fish: "A fish with fins and a tail",
      pizza: "A pizza or a slice of pizza", umbrella: "An umbrella",
      dog: "A dog", car: "A car with wheels", house: "A house with a roof",
      tree: "A tree with a trunk and branches", sun: "The sun with rays",
      star: "A star shape", cup: "A drinking cup", sailboat: "A boat with a sail",
    } },
};
const visionDemos = {
  flowers: { name: "Flowers", state: "Classify the flower shown in the attached image.", questions: flowerQuestions, split: "test", answerKey: "flower" },
  quickdraw: { name: "Doodle Detective", state: "Identify the object in this sketch from its visual appearance.", questions: doodleQuestions, split: "demo", answerKey: "doodle" },
};
type VisionDemo = keyof typeof visionDemos;
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
  meta: { latency_ms: number; passes: number; temperature: number; probability_source: string };
};
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
async function responseBody(response: Response) {
  try {
    return await response.json();
  } catch {
    throw new Error("Predictions are temporarily unavailable. You can still draw and browse the gallery. Please try again later.");
  }
}
function App() {
  const chosenView = useRef(false);
  const [mode, setMode] = useState<"text" | "vision">("text");
  const [modelName, setModelName] = useState("Connecting to model");
  const [apiDocs, setApiDocs] = useState("/docs");
  const [queued, setQueued] = useState(false);
  const [progress, setProgress] = useState("");
  const [supportsImages, setSupportsImages] = useState(false);
  const [visionDemo, setVisionDemo] = useState<VisionDemo>("quickdraw");
  const [drawing, setDrawing] = useState(false);
  const [drawingKey, setDrawingKey] = useState(0);
  const [strokeActive, setStrokeActive] = useState(false);
  const [galleryError, setGalleryError] = useState("");
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [gallery, setGallery] = useState<GalleryImage[]>([]);
  const [galleryTotal, setGalleryTotal] = useState(0);
  const [galleryOffset, setGalleryOffset] = useState(0);
  const [galleryLabel, setGalleryLabel] = useState("");
  const [dataset, setDataset] = useState<{ title: string; count: number; benchmark_count: number; source: string; license: string; labels: string[] } | null>(null);
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
    [editor, setEditor] = useState(false),
    [questionJson, setQuestionJson] = useState(""),
    [stale, setStale] = useState(false);
  const load = (p: Preset) => {
    chosenView.current = true;
    history.replaceState(null, "", "#text");
    setMode("text");
    setStrokeActive(false);
    setSelectedImage(null);
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
    let pollId: ReturnType<typeof setInterval> | undefined;
    const poll = () =>
      fetch("/health")
        .then((r) => r.json())
        .then((h) => {
          setReady(h.ready);
          setModelName(h.display_name || h.model);
          setApiDocs(h.api_docs || "/docs");
          setSupportsImages(Boolean(h.supports_images));
          setQueued(h.execution === "queued");
          // A cloud health check describes CPU admission, not GPU readiness.
          // Once discovered, it needs no background polling.
          if (h.execution === "queued") clearInterval(pollId);
          return h;
        })
        .catch(() => setReady(false));
    Promise.all([fetch("/api/examples").then(r => r.json()), poll()])
      .then(([p, h]) => {
        setPresets(p);
        if (chosenView.current) return;
        if (h?.supports_images && location.hash !== "#text") {
          openVision(location.hash === "#flowers" ? "flowers" : "quickdraw");
        } else {
          load(p[0]);
        }
      })
      .catch(() => setError("Could not load examples. Check that the API is running."));
    pollId = setInterval(poll, 5000);
    return () => clearInterval(pollId);
  }, []);
  useEffect(() => {
    if (mode !== "vision") return;
    const controller = new AbortController();
    setGallery([]); setGalleryTotal(0); setDataset(null); setGalleryError("");
    if (queued) {
      fetch(`/gallery/${visionDemo}/manifest.json`, { signal: controller.signal })
        .then(async r => { if (!r.ok) throw new Error("Could not load the gallery. You can still draw or upload an image."); return r.json(); })
        .then(data => {
          setDataset(data);
          const selected = data.images.filter((item: GalleryImage) => !galleryLabel || item.label === galleryLabel);
          setGalleryTotal(selected.length);
          setGallery(selected.slice(galleryOffset, galleryOffset + 12));
        })
        .catch(e => { if (e.name !== "AbortError") setGalleryError(e.message); });
      return () => controller.abort();
    }
    fetch(`/api/datasets/${visionDemo}`, { signal: controller.signal })
      .then(async r => { if (!r.ok) throw new Error("This gallery is not prepared yet. You can still upload an image or draw a doodle."); return r.json(); })
      .then(setDataset)
      .catch(e => { if (e.name !== "AbortError") setGalleryError(e.message); });
    fetch(`/api/datasets/${visionDemo}/images?split=${visionDemos[visionDemo].split}&offset=${galleryOffset}&limit=12&label=${encodeURIComponent(galleryLabel)}`, { signal: controller.signal })
      .then(async r => { if (!r.ok) throw new Error("Could not load image gallery."); return r.json(); })
      .then(body => { setGallery(body.items); setGalleryTotal(body.total); })
      .catch(e => { if (e.name !== "AbortError") setGalleryError(e.message); });
    return () => controller.abort();
  }, [mode, visionDemo, galleryOffset, galleryLabel, queued]);
  const openVision = (demo: VisionDemo = visionDemo) => {
    chosenView.current = true;
    history.replaceState(null, "", demo === "quickdraw" ? "#doodle" : "#flowers");
    setMode("vision"); setSelectedImage(null); setResult(null); setError(""); setStale(false);
    setVisionDemo(demo); setDrawing(demo === "quickdraw"); setGalleryLabel(""); setGalleryOffset(0);
    setDrawingKey(key => key + 1);
    setStrokeActive(false);
    setStructured(false); setState(visionDemos[demo].state);
    setQuestions(visionDemos[demo].questions);
  };
  const selectGalleryImage = (item: GalleryImage) => {
    setSelectedImage({ input: `${visionDemo}:${item.id}`, url: queued ? `/gallery/${visionDemo}/images/${item.id}.jpg` : `/api/datasets/${visionDemo}/image/${item.id}`, label: item.label });
    setResult(null); setError(""); setStale(false);
  };
  const updateDrawing = (url: string | null) => {
    setStrokeActive(false);
    setSelectedImage(url ? { input: url, url } : null);
    setResult(null); setError(""); setStale(false);
  };
  const startStroke = () => {
    setStrokeActive(true);
    setResult(null); setError(""); setStale(false);
  };
  const uploadImage = async (file?: File) => {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 6_000_000) {
      setError("Choose a JPEG, PNG or WebP image up to 6 MB."); return;
    }
    const reader = new FileReader();
    reader.onload = async () => {
      let url = String(reader.result);
      if (queued) {
        try {
          const bitmap = await createImageBitmap(file);
          const scale = Math.min(1, 768 / Math.max(bitmap.width, bitmap.height));
          const canvas = document.createElement("canvas");
          canvas.width = Math.max(1, Math.round(bitmap.width * scale));
          canvas.height = Math.max(1, Math.round(bitmap.height * scale));
          const ctx = canvas.getContext("2d");
          if (!ctx) throw new Error("Could not prepare the image.");
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          bitmap.close();
          url = canvas.toDataURL("image/jpeg", 0.9);
        } catch {
          setError("Could not prepare the image. Please try a smaller JPEG or PNG.");
          return;
        }
      }
      setDrawing(false);
      setSelectedImage({ input: url, url }); setResult(null); setError(""); setStale(false);
    };
    reader.onerror = () => setError("Could not read the image.");
    reader.readAsDataURL(file);
  };
  const request = (): Request => ({
    model: "diffusion-jev",
    state: structured ? JSON.parse(state) : state,
    questions,
    ...(mode === "vision" && selectedImage ? { images: [selectedImage.input] } : {}),
  });
  const evaluate = async () => {
    setBusy(true);
    setError("");
    setProgress(queued ? "Starting your prediction…" : "");
    try {
      const response = await fetch(queued ? "/api/jobs" : "/v1/systemone", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Request-ID": crypto.randomUUID() },
        body: JSON.stringify(request()),
      });
      let body = await responseBody(response);
      if (!response.ok)
        throw new Error(
          typeof body.detail === "string" ? body.detail : pretty(body.detail),
        );
      if (queued) {
        const jobId = body.job_id;
        const deadline = Date.now() + 20 * 60 * 1000;
        setProgress("Waiting for your prediction. The first request may take a few minutes while the model starts.");
        while (true) {
          if (Date.now() > deadline) throw new Error("The prediction took too long. Please try again later.");
          await new Promise(resolve => setTimeout(resolve, 2000));
          const check = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
          const job = await responseBody(check);
          if (!check.ok || job.status === "failed") throw new Error(job.detail || "Prediction failed. Please try again later.");
          if (job.status === "completed") { body = job.result; break; }
        }
      }
      setResult(body);
      setTab("results");
      setStale(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setBusy(false);
      setProgress("");
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
          <a href={apiDocs} target="_blank" rel="noreferrer">
            API reference <ArrowUpRight size={14} />
          </a>
          <a href="https://github.com/Hangzhi/diffusion-jev-sglang" target="_blank" rel="noreferrer">
            GitHub <ArrowUpRight size={14} />
          </a>
          <span className="local-tag">
            <span /> Self-hosted
          </span>
        </nav>
      </header>
      <main className={mode === "vision" ? "image-demo" : ""}>
        <div className="eyebrow">
          <span /> SMALL OUTPUT. BIG DECISIONS.
        </div>
        <div className="intro">
          <div>
            {mode === "vision" && visionDemo === "quickdraw" ? <h1>Draw something.<br /><span>See what it thinks.</span></h1> : <h1>
              Less generation.
              <br />
              <span>More decision.</span>
            </h1>}
            <p>
              {mode === "vision" && visionDemo === "quickdraw"
                ? `A quick sketch. ${Object.keys(questions.doodle?.criteria ?? {}).length} possible answers. One ${queued ? "diffusion" : "local"} model.`
                : "Give the model text or an image. Get a choice, a yes/no answer, or a score."}
            </p>
          </div>
          <div className="model-card">
            <div className="model-symbol">
              <Sparkles size={22} />
            </div>
            <div>
              <small>POWERED BY</small>
              <strong>{modelName}</strong>
              <span>SGLang · BF16 · diffusion</span>
            </div>
            <span className={"status " + (ready ? "online" : "")}>
              <i />
              {ready ? queued ? "On demand" : "Ready" : "Connecting"}
            </span>
          </div>
        </div>
        <div className="mode-tabs" role="group" aria-label="Classification mode">
          <button className={mode === "vision" && visionDemo === "quickdraw" ? "selected" : ""}
            aria-pressed={mode === "vision" && visionDemo === "quickdraw"}
            disabled={busy || !supportsImages} onClick={() => openVision("quickdraw")}>
            <Pencil size={15} /> Doodle Detective
          </button>
          <button className={mode === "vision" && visionDemo === "flowers" ? "selected" : ""}
            aria-pressed={mode === "vision" && visionDemo === "flowers"}
            disabled={busy || !supportsImages} onClick={() => openVision("flowers")}>
            <ImageIcon size={15} /> Flowers
          </button>
          <button className={mode === "text" ? "selected" : ""} disabled={busy}
            aria-pressed={mode === "text"}
            onClick={() => { const p = presets.find(x => x.id === active) || presets[0]; if (p) load(p); }}>
            <Braces size={15} /> Text decisions
          </button>
        </div>
        <div className="workspace-top">
          <span className="workspace-label">
            <Terminal size={16} /> {mode === "vision" ? visionDemos[visionDemo].name : "Decision playground"}
          </span>
          {mode === "text" && <div className="preset-tabs">
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
          </div>}
        </div>
        <div className="workspace">
          <section className="input-panel">
            <div className="actions">
              <button
                className={"evaluate" + (strokeActive && selectedImage ? " drawing-stroke" : "")}
                aria-busy={busy}
                disabled={busy || strokeActive || !ready || !state.trim() || (mode === "vision" && !selectedImage)}
                onClick={evaluate}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
                {busy ? "Evaluating…" : mode === "vision" ? visionDemo === "quickdraw" ? "Guess doodle" : "Classify image" : "Evaluate state"}
                <ArrowRight size={16} />
              </button>
            </div>
            {queued && <p className="cloud-note" role="status" aria-live="polite">
              {progress || "The model sleeps between visits to keep this demo affordable. The first prediction may take a few minutes."}
            </p>}
            {mode === "vision" && (
              <div className="vision-panel">
                <div className="panel-heading"><h2><ImageIcon size={18} /> {visionDemo === "quickdraw" ? "Make a doodle" : "Choose an image"}</h2>
                  <label className={"upload-button" + (busy ? " disabled" : "")}>
                    <Upload size={14} /> Upload
                    <input type="file" aria-label="Upload classification image" accept="image/jpeg,image/png,image/webp"
                      disabled={busy} onChange={e => { void uploadImage(e.target.files?.[0]); e.target.value = ""; }} />
                  </label>
                </div>
                {visionDemo === "quickdraw" && <>
                  {questions.doodle?.type === "choice" && questions.doodle.criteria != null &&
                    <div className="doodle-categories"><span>Possible guesses</span><div className="chips">
                      {Object.keys(questions.doodle.criteria).map(label => <span key={label}>{label}</span>)}
                    </div></div>}
                  <div className="doodle-tabs" role="group" aria-label="Doodle source">
                    <button disabled={busy} className={drawing ? "selected" : ""} onClick={() => { if (!drawing) { setDrawing(true); updateDrawing(null); } }}>Draw your own</button>
                    <button disabled={busy} className={!drawing ? "selected" : ""} onClick={() => { if (drawing) { setDrawing(false); updateDrawing(null); } }}>Try a sketch</button>
                  </div>
                </>}
                {drawing ? <DrawPad key={drawingKey} disabled={busy} onChange={updateDrawing} onStrokeStart={startStroke} /> : selectedImage ? (
                  <div className="selected-image">
                    <img src={selectedImage.url} alt="Selected image to classify" />
                    <span>{selectedImage.label && !stale && result?.answers[visionDemos[visionDemo].answerKey]?.choice
                      ? `Dataset label: ${selectedImage.label} · ${result.answers[visionDemos[visionDemo].answerKey]?.choice === selectedImage.label ? "Matched" : "Different guess"}`
                      : "The model receives image pixels, without the dataset label."}</span>
                  </div>
                ) : <div className="image-placeholder"><ImageIcon size={28} /><span>Select {visionDemo === "flowers" ? "a flower" : "a sketch"} below or upload your own image.</span></div>}
                {!drawing && galleryError && <p className="gallery-message">{galleryError}</p>}
                {!drawing && dataset && <>
                  <div className="gallery-heading"><div><strong>{dataset.title}</strong><small>{dataset.count.toLocaleString()} images · {dataset.labels.length} classes · {dataset.benchmark_count ? `${dataset.benchmark_count} benchmark images` : "Demo collection"}</small></div>
                    <select aria-label="Filter image category" disabled={busy} value={galleryLabel} onChange={e => { setGalleryLabel(e.target.value); setGalleryOffset(0); }}>
                      <option value="">All categories</option>{dataset.labels.map(label => <option key={label} value={label}>{label}</option>)}
                    </select>
                  </div>
                  <div className="image-gallery">{gallery.map((item, i) => (
                    <button key={item.id} disabled={busy} aria-label={`Select ${visionDemo === "flowers" ? "flower image" : "doodle"} ${galleryOffset + i + 1}`}
                      className={selectedImage?.input === `${visionDemo}:${item.id}` ? "selected" : ""} onClick={() => selectGalleryImage(item)}>
                      <img loading="lazy" src={queued ? `/gallery/${visionDemo}/images/${item.id}.jpg` : `/api/datasets/${visionDemo}/image/${item.id}`} alt={`Image example ${galleryOffset + i + 1}`} />
                    </button>
                  ))}</div>
                  <div className="gallery-footer"><button disabled={busy || galleryOffset === 0} onClick={() => setGalleryOffset(Math.max(0, galleryOffset - 12))}>Previous</button>
                    <span>{Math.min(galleryOffset + 1, galleryTotal)}–{Math.min(galleryOffset + 12, galleryTotal)} of {galleryTotal} {visionDemo === "flowers" ? "test images" : "sketches"}</span>
                    <button disabled={busy || galleryOffset + 12 >= galleryTotal} onClick={() => setGalleryOffset(galleryOffset + 12)}>Next</button>
                  </div>
                  <a className="dataset-source" href={dataset.source} target="_blank" rel="noreferrer">Dataset source · {dataset.license} <ArrowUpRight size={12} /></a>
                </>}
              </div>
            )}
            <details className={`decision-settings ${mode}`} key={`${mode}-${visionDemo}`} open={mode === "text" ? true : undefined}>
              <summary>Adjust the question and choices</summary>
              <div className="settings-body">
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
                      if (mode === "vision") { openVision(); return; }
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
              </div>
            </details>
            <div className="privacy">
              <span /> {queued ? "Predictions run on Modal. The GPU sleeps between visits." : "Context stays on your inference server."}
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
                <h3>{mode === "vision" ? "Looking at your image…" : "Reading your text…"}</h3>
                <p>Waiting for the model’s answer.</p>
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
                  {mode === "vision" && visionDemo === "quickdraw" ? "What will it see?" : "Your answer goes here."}
                </h3>
                <p>
                  {mode === "vision" && visionDemo === "quickdraw"
                    ? "Draw one of the listed objects, then click Guess doodle."
                    : "Choose an example or bring your own."}
                </p>
                <span className="empty-tip">
                  <ArrowRight size={13} /> {mode === "text" ? "Start with “Evaluate state”" : visionDemo === "quickdraw" ? "A simple sketch is enough" : "Choose an image to classify"}
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
                  <Check size={14} /> {result.meta.probability_source === "self_conditioned_denoiser_logits" ? "Denoiser scores · up to " : "Real model logits · "}{result.meta.passes}{" "}
                  {result.meta.probability_source === "self_conditioned_denoiser_logits" ? "steps" : `pass${result.meta.passes > 1 ? "es" : ""}`} · T=
                  {result.meta.temperature.toFixed(2)}
                </div>
              </div>
            )}
            <div className="output-note">
              <CircleHelp size={14} />
              <span>
                Probabilities are normalized over your candidate options.
                <br />{supportsImages ? "Denoiser scores include self-conditioning and are not calibrated confidence." : "A confident answer can still be wrong."}
              </span>
            </div>
          </section>
        </div>
        <div className="bottom-notes">
          <span>
            <Code2 size={15} /> Three primitives. One endpoint.
          </span>
          <code>{queued ? "POST /api/jobs" : "POST /v1/systemone"}</code>
          <a href={apiDocs}>
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
