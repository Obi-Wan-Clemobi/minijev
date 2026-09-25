"use client";
// The State machine page: build a flow of steps on a canvas, then run it on a request and watch each step decide.
import "@xyflow/react/dist/style.css";
import {
  Background, Controls, MiniMap, ReactFlow, ReactFlowProvider, useReactFlow,
  type Connection, type Edge, type Node, type NodeChange,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { DoneNode, StepNode } from "@/components/flows/Nodes";
import { RunTrace } from "@/components/flows/RunTrace";
import { StepEditor } from "@/components/flows/StepEditor";
import { Tip } from "@/components/Tip";
import { API, api, type FlowCheck } from "@/lib/api";
import {
  addStep, connect, type Decision, DONE, emptyFlow, type Flow, moveStep, parseLines, removeEdge, removeStep,
  type RunEnd, type RunEvent, type StepType, toEdges, toNodes,
} from "@/lib/flow";

const nodeTypes = { step: StepNode, done: DoneNode };
const DRAFT = "mj-flow-draft";
const btn = "h-8 px-3 rounded-md border border-line text-[13px] hover:bg-track disabled:opacity-50";
const PALETTE: [StepType, string, string][] = [
  ["noul", "Yes / No", "answers yes or no"],
  ["choice", "Choice", "picks one option"],
  ["score", "Score", "a level on a scale"],
];

function subscribeTheme(cb: () => void) {
  window.addEventListener("mj-theme", cb);
  return () => window.removeEventListener("mj-theme", cb);
}

function Builder() {
  const dark = useSyncExternalStore(subscribeTheme, () => document.documentElement.classList.contains("dark"), () => true);
  const { screenToFlowPosition } = useReactFlow();
  const [canvasKey, setCanvasKey] = useState(0); // a new key remounts the canvas, which fits the view after measuring
  const [flow, setFlowRaw] = useState<Flow>(emptyFlow());
  const [templates, setTemplates] = useState<Flow[]>([]);
  const [saved, setSaved] = useState<{ id: string; name: string; saved: string }[]>([]);
  const [problems, setProblems] = useState<FlowCheck>({ errors: [], warnings: [] });
  const [selected, setSelected] = useState<string | null>(null);
  const [focusArrow, setFocusArrow] = useState<number | null>(null);
  const [query, setQuery] = useState("find all TODO comments in Python files");
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [end, setEnd] = useState<RunEnd | null>(null);
  const [note, setNote] = useState<string | null>(null);
  // React Flow measures each box and moves it while it is dragged; the flow JSON only changes when the drag ends.
  const [measured, setMeasured] = useState<Record<string, { width: number; height: number }>>({});
  const [dragging, setDragging] = useState<Record<string, { x: number; y: number }>>({});
  const fileRef = useRef<HTMLInputElement>(null);

  const clearRun = () => { setDecisions([]); setEnd(null); setRunning(null); };
  const setFlow = useCallback((f: Flow) => { setFlowRaw(f); setEnd(null); }, []);
  const refit = useCallback(() => { setMeasured({}); setCanvasKey((k) => k + 1); }, []);
  const load = (f: Flow) => { setFlowRaw(f); setSelected(null); clearRun(); refit(); };

  useEffect(() => {
    api.flowTemplates().then((ts) => {
      setTemplates(ts);
      let draft: Flow | null = null;
      try { draft = JSON.parse(localStorage.getItem(DRAFT) ?? "null"); } catch { /* no draft */ }
      setFlowRaw(draft?.steps && Object.keys(draft.steps).length ? draft : ts.find((t) => t.id === "tool-selection") ?? ts[0] ?? emptyFlow());
      refit();
    }).catch((e) => setNote(e.message));
    api.flows().then(setSaved).catch(() => {});
  }, [refit]);

  useEffect(() => { // keep a draft in this browser, and check the flow on the server (errors block a run)
    try { if (Object.keys(flow.steps).length) localStorage.setItem(DRAFT, JSON.stringify(flow)); } catch { /* storage off */ }
    const t = setTimeout(() => { if (Object.keys(flow.steps).length) api.checkFlow(flow).then(setProblems).catch(() => {}); }, 300);
    return () => clearTimeout(t);
  }, [flow]);

  const rfNodes: Node[] = useMemo(() => toNodes(flow, decisions, running, end, selected).map((n) => ({
    ...n, position: dragging[n.id] ?? n.position, measured: measured[n.id],
  })), [flow, decisions, running, end, selected, measured, dragging]);

  const edges: Edge[] = useMemo(() => toEdges(flow, decisions).map((e) => ({
    ...e, labelStyle: { fill: "var(--fg)", fontSize: 11 }, labelBgStyle: { fill: "var(--card)" },
    selected: selected !== null && e.source === selected && focusArrow === Number(e.id.split(":").pop()),
  })), [flow, decisions, selected, focusArrow]);

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    for (const c of changes) {
      if (c.type === "dimensions" && c.dimensions) setMeasured((m) => ({ ...m, [c.id]: c.dimensions! }));
      if (c.type !== "position" || c.id === DONE) continue;
      if (c.dragging && c.position) setDragging((d) => ({ ...d, [c.id]: c.position! }));
      if (!c.dragging) {
        setDragging((d) => {
          const pos = c.position ?? d[c.id];
          if (pos) setFlowRaw((f) => moveStep(f, c.id, pos));
          const { [c.id]: _, ...rest } = d; // eslint-disable-line @typescript-eslint/no-unused-vars
          return rest;
        });
      }
    }
  }, []);

  const addAt = (type: StepType, pos: { x: number; y: number }) => {
    const [f, id] = addStep(flow, type, pos);
    setFlow(f);
    setSelected(id);
    setFocusArrow(null);
  };

  const run = async () => {
    clearRun();
    setSelected(null);
    setRunning(flow.start);
    try {
      const r = await fetch(`${API}/v1/flows/run`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ flow, query }) });
      if (!r.ok || !r.body) throw new Error((await r.json().catch(() => ({})))?.detail ?? `${r.status}`);
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "", ended = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const [events, tail] = parseLines(buf + dec.decode(value, { stream: true }));
        buf = tail;
        events.forEach((e: RunEvent) => {
          if (e.event === "decision") {
            setDecisions((ds) => [...ds, e]);
            setRunning(e.next && e.next !== DONE ? e.next : null);
          } else { setEnd(e); setRunning(null); ended = true; }
        });
      }
      if (!ended) throw new Error("the server stopped before the run ended");
    } catch (e) {
      setEnd({ event: "end", status: "error", step: null, decisions: [], message: e instanceof Error ? e.message : String(e) });
      setRunning(null);
    }
  };

  const exportFlow = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(flow, null, 1)], { type: "application/json" }));
    a.download = `${flow.id}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const importFlow = async (file: File) => {
    try {
      const f = JSON.parse(await file.text()) as Flow;
      const check = await api.checkFlow(f);
      load(f);
      setNote(check.errors.length ? `Loaded ${file.name} with ${check.errors.length} error(s); see the list on the right.` : `Loaded ${file.name}.`);
    } catch (e) { setNote(`Cannot load ${file.name}: ${e instanceof Error ? e.message : e}`); }
  };

  const save = async () => {
    try {
      const r = await api.saveFlow(flow);
      setSaved(await api.flows());
      setNote(`Saved as ${r.id}.`);
    } catch (e) { setNote(e instanceof Error ? e.message : String(e)); }
  };

  const step = selected ? flow.steps[selected] : undefined;
  const nSteps = Object.keys(flow.steps).length;

  return (
    <div className="px-4 md:px-8 py-6 flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <h1 className="m-0 text-[32px] font-semibold tracking-tight flex items-center gap-2">State machine <Tip k="smWhat" /></h1>
        <p className="m-0 text-[15px] text-muted max-w-[900px] leading-normal">
          Chain questions into a flow. Each step asks the model one question about your request; the answer picks the arrow to the next step.
          Later steps read what earlier steps decided.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input className="h-8 px-2 rounded-md border border-line bg-bg text-[13px] w-[180px]" value={flow.name} aria-label="Flow name"
          onChange={(e) => setFlow({ ...flow, name: e.target.value, id: e.target.value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "flow" })} />
        <select className={btn} value="" aria-label="Templates" onChange={(e) => { const t = templates.find((x) => x.id === e.target.value); if (t) load(t); }}>
          <option value="">Templates…</option>
          {templates.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <select className={btn} value="" aria-label="Saved flows" onChange={(e) => e.target.value && api.flow(e.target.value).then(load).catch((x) => setNote(x.message))}>
          <option value="">Saved…{saved.length ? ` (${saved.length})` : ""}</option>
          {saved.map((s) => <option key={s.id} value={s.id}>{s.name} · {s.saved}</option>)}
        </select>
        <button className={btn} onClick={() => load(emptyFlow())}>New</button>
        <button className={btn} onClick={save} disabled={!nSteps}>Save</button>
        <button className={btn} onClick={exportFlow} disabled={!nSteps}>Export</button>
        <button className={btn} onClick={() => fileRef.current?.click()}>Import</button>
        <input ref={fileRef} type="file" accept="application/json,.json" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) importFlow(f); e.target.value = ""; }} />
        <Tip k="smSave" />
        {note && <span className="text-xs text-muted">{note}</span>}
      </div>

      <div className="grid lg:grid-cols-[170px_minmax(0,1fr)_380px] gap-4">
        <aside className="flex lg:flex-col gap-2 flex-wrap">
          <span className="text-xs text-muted w-full">Drag onto the canvas, or click to add</span>
          {PALETTE.map(([type, name, sub]) => (
            <button key={type} draggable className="rounded-lg border border-line bg-card px-3 py-2 text-left hover:border-accent"
              onDragStart={(e) => { e.dataTransfer.setData("application/mj-step", type); e.dataTransfer.effectAllowed = "move"; }}
              onClick={() => addAt(type, { x: 40 * nSteps, y: 420 + 20 * nSteps })}>
              <span className="block text-[13px] font-medium">+ {name}</span>
              <span className="block text-[11px] text-muted">{sub}</span>
            </button>
          ))}
          <div className="hidden lg:flex flex-col gap-1.5 mt-3 text-[11px] text-muted leading-snug">
            <span className="flex items-center gap-1">How it works <Tip k="smArrow" /></span>
            <span>Drag from an answer dot to a box to add an arrow.</span>
            <span>Dashed arrow = only if sure enough.</span>
            <span>Select a box or an arrow and press Backspace to delete it.</span>
          </div>
        </aside>

        <div className="flex flex-col gap-2 min-w-0">
          <div className="flex gap-2">
            <input className="flex-1 h-9 px-3 rounded-md border border-line bg-bg text-sm" value={query} aria-label="Request"
              placeholder="A request for the flow, e.g. find all TODO comments in Python files" onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !running) run(); }} />
            <button className="h-9 px-4 rounded-md bg-accent text-inv-fg text-sm font-medium disabled:opacity-50" onClick={run}
              disabled={!!running || !query.trim() || !nSteps || problems.errors.length > 0}>{running ? "Running…" : "Run"}</button>
            <Tip k="smRun" />
          </div>
          <div className="h-[620px] rounded-xl border border-line bg-surface overflow-hidden"
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }}
            onDrop={(e) => {
              const type = e.dataTransfer.getData("application/mj-step") as StepType;
              if (type) addAt(type, screenToFlowPosition({ x: e.clientX, y: e.clientY }));
            }}>
            <ReactFlow key={canvasKey} nodes={rfNodes} edges={edges} nodeTypes={nodeTypes} colorMode={dark ? "dark" : "light"} fitView fitViewOptions={{ padding: 0.12 }}
              onNodesChange={onNodesChange}
              onConnect={(c: Connection) => c.target && setFlow(connect(flow, c.source, c.sourceHandle, c.target))}
              onNodeClick={(_, n) => { if (n.id !== DONE) { setSelected(n.id); setFocusArrow(null); } }}
              onEdgeClick={(_, e) => { setSelected(e.source); setFocusArrow(Number(e.id.split(":").pop())); }}
              onPaneClick={() => { setSelected(null); setFocusArrow(null); }}
              onNodesDelete={(ns) => { let f = flow; ns.forEach((n) => { if (n.id !== DONE) f = removeStep(f, n.id); }); setFlow(f); setSelected(null); }}
              onEdgesDelete={(es) => { let f = flow; [...es].sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true })).forEach((e) => { f = removeEdge(f, e.id); }); setFlow(f); }}
              proOptions={{ hideAttribution: true }}>
              <Background gap={20} />
              <Controls />
              <MiniMap pannable zoomable className="!hidden md:!block" />
            </ReactFlow>
          </div>
          {(problems.errors.length > 0 || problems.warnings.length > 0) && (
            <div className="flex flex-col gap-0.5 text-xs">
              {problems.errors.map((p, i) => <span key={`e${i}`} className="text-warn">Error{p.step ? ` in ${p.step}` : ""}: {p.message}</span>)}
              {problems.warnings.map((p, i) => <span key={`w${i}`} className="text-muted">Note{p.step ? ` for ${p.step}` : ""}: {p.message}</span>)}
            </div>
          )}
        </div>

        <aside className="rounded-xl border border-line bg-card p-4 lg:max-h-[680px] lg:overflow-y-auto">
          {step
            ? <StepEditor flow={flow} step={step} focus={focusArrow} onChange={setFlow}
              onDelete={() => { setFlow(removeStep(flow, step.id)); setSelected(null); }} onClose={() => setSelected(null)} />
            : <RunTrace flow={flow} decisions={decisions} end={end} running={running} onSelect={(id) => setSelected(id)} />}
        </aside>
      </div>
    </div>
  );
}

export default function FlowsPage() {
  return <ReactFlowProvider><Builder /></ReactFlowProvider>;
}
