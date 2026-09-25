// Flows (the State machine page): the flow JSON of src/minijev/flows.py, and its conversion to React Flow nodes and
// edges. The flow JSON is the only source of truth; the canvas is drawn from it and every edit changes it.

export type Answer = string | boolean | number;
export type Condition = { confidence_gte?: number | null; confidence_lt?: number | null };
// from_answer missing or null = any answer (a fallback arrow)
export type Transition = { from_answer?: Answer | null; target: string; condition?: Condition | null };
export type StepType = "noul" | "choice" | "score";
export type Step = {
  id: string; type: StepType; instructions: string;
  criteria?: Record<string, string | null> | string[] | null;
  position: { x: number; y: number }; transitions: Transition[];
};
export type Flow = { id: string; name: string; description: string; version: string; start: string; steps: Record<string, Step> };

export type Decision = {
  step: string; type: StepType; question: string; answer: Answer; confidence: number;
  probabilities: Record<string, number> | null; transition: number | null; next: string | null; ms: number;
  state: { request: string; "decisions so far": { question: string; answer: Answer; confidence: number }[] };
};
export type RunEnd = {
  event: "end"; status: "completed" | "no_transition" | "max_steps" | "invalid" | "error"; step: string | null;
  decisions: Decision[]; message?: string; path?: string[];
  errors?: { step: string | null; message: string }[]; warnings?: { step: string | null; message: string }[];
};
export type RunEvent = ({ event: "decision" } & Decision) | RunEnd;

export const DONE = "DONE";
export const ANY = "*"; // the handle for "any answer" (from_answer: null)

/** Every answer a step can give, in order: yes/no, the option keys, or the level numbers. */
export function answersOf(s: Step): Answer[] {
  if (s.type === "noul") return [true, false];
  if (s.type === "choice") return Object.keys((s.criteria as Record<string, unknown>) ?? {});
  return ((s.criteria as string[]) ?? []).map((_, i) => i);
}

export const handleId = (a?: Answer | null) => (a == null ? ANY : `a:${String(a)}`);

export function answerFromHandle(s: Step, handle: string | null | undefined): Answer | null {
  if (!handle || handle === ANY) return null;
  return answersOf(s).find((a) => handleId(a) === handle) ?? null;
}

/** How an answer is shown: "yes"/"no", the option key, or "level 2 · frustrated". */
export function answerLabel(s: Step | undefined, a?: Answer | null): string {
  if (a == null) return "any answer";
  if (typeof a === "boolean") return a ? "yes" : "no";
  if (s?.type === "score" && typeof a === "number") {
    const name = (s.criteria as string[] | undefined)?.[a];
    return name ? `${a} · ${name}` : `level ${a}`;
  }
  return String(a);
}

export function conditionLabel(c?: Condition | null): string {
  if (!c) return "";
  const parts = [];
  if (c.confidence_gte != null) parts.push(`sure ≥ ${c.confidence_gte}`);
  if (c.confidence_lt != null) parts.push(`sure < ${c.confidence_lt}`);
  return parts.join(", ");
}

export function edgeLabel(s: Step, t: Transition): string {
  const c = conditionLabel(t.condition);
  return c ? `${answerLabel(s, t.from_answer)} · ${c}` : answerLabel(s, t.from_answer);
}

export type StepStatus = "idle" | "visited" | "current" | "failed";
export type NodeData = { step?: Step; start?: boolean; status: StepStatus; decision?: Decision; problem?: string };

/** React Flow nodes: one per step, plus the DONE node. status comes from the run so far. */
export function toNodes(flow: Flow, decisions: Decision[], running: string | null, end: RunEnd | null, selected: string | null) {
  const last = new Map(decisions.map((d) => [d.step, d]));
  const xs = Object.values(flow.steps).map((s) => s.position.x);
  const nodes: { id: string; type: string; position: { x: number; y: number }; selected: boolean; data: NodeData }[] = Object.values(flow.steps).map((s) => ({
    id: s.id, type: "step", position: s.position, selected: s.id === selected,
    data: {
      step: s, start: s.id === flow.start, decision: last.get(s.id),
      status: (end && end.step === s.id && end.status !== "completed" ? "failed"
        : running === s.id ? "current" : last.has(s.id) ? "visited" : "idle") as StepStatus,
      problem: end && end.step === s.id ? end.message : undefined,
    },
  }));
  const reached = end?.status === "completed";
  nodes.push({
    id: DONE, type: "done", selected: false,
    position: { x: (xs.length ? Math.max(...xs) : 0) + 360, y: Object.values(flow.steps)[0]?.position.y ?? 0 },
    data: { status: reached ? "visited" : "idle" },
  });
  return nodes;
}

/** React Flow edges: one per transition, from the answer's handle. Taken edges are highlighted. */
export function toEdges(flow: Flow, decisions: Decision[]) {
  const taken = new Set(decisions.filter((d) => d.transition !== null).map((d) => `${d.step}:${d.transition}`));
  return Object.values(flow.steps).flatMap((s) => s.transitions.map((t, i) => {
    const id = `${s.id}:${i}`, on = taken.has(id);
    return {
      id, source: s.id, sourceHandle: handleId(t.from_answer), target: t.target, label: edgeLabel(s, t),
      animated: on, data: { taken: on, conditional: !!conditionLabel(t.condition) },
      style: { strokeWidth: on ? 2.5 : 1.5, stroke: on ? "var(--accent)" : "var(--ghost)", strokeDasharray: conditionLabel(t.condition) ? "6 4" : undefined },
    };
  }));
}

// ---- edits: each returns a new flow

export function connect(flow: Flow, source: string, handle: string | null | undefined, target: string): Flow {
  const s = flow.steps[source];
  if (!s || source === target) return flow;
  const t: Transition = { from_answer: answerFromHandle(s, handle), target };
  return { ...flow, steps: { ...flow.steps, [source]: { ...s, transitions: [...s.transitions, t] } } };
}

export function removeEdge(flow: Flow, edgeId: string): Flow {
  const [sid, i] = [edgeId.slice(0, edgeId.lastIndexOf(":")), Number(edgeId.slice(edgeId.lastIndexOf(":") + 1))];
  const s = flow.steps[sid];
  if (!s) return flow;
  return { ...flow, steps: { ...flow.steps, [sid]: { ...s, transitions: s.transitions.filter((_, j) => j !== i) } } };
}

export function updateEdge(flow: Flow, edgeId: string, patch: Partial<Transition>): Flow {
  const sid = edgeId.slice(0, edgeId.lastIndexOf(":")), i = Number(edgeId.slice(edgeId.lastIndexOf(":") + 1));
  const s = flow.steps[sid];
  if (!s?.transitions[i]) return flow;
  const transitions = s.transitions.map((t, j) => (j === i ? { ...t, ...patch } : t));
  return { ...flow, steps: { ...flow.steps, [sid]: { ...s, transitions } } };
}

export function moveStep(flow: Flow, id: string, position: { x: number; y: number }): Flow {
  const s = flow.steps[id];
  return s ? { ...flow, steps: { ...flow.steps, [id]: { ...s, position } } } : flow;
}

export function updateStep(flow: Flow, id: string, patch: Partial<Step>): Flow {
  const s = flow.steps[id];
  if (!s) return flow;
  const next = { ...s, ...patch };
  if (patch.criteria !== undefined) { // arrows from answers that no longer exist would be invalid
    const ok = new Set(answersOf(next).map(String));
    next.transitions = next.transitions.filter((t) => t.from_answer == null || ok.has(String(t.from_answer)));
  }
  return { ...flow, steps: { ...flow.steps, [id]: next } };
}

const STARTERS: Record<StepType, Pick<Step, "instructions" | "criteria">> = {
  noul: { instructions: "Is this true?", criteria: null },
  choice: { instructions: "Which option fits best?", criteria: { first: "The first option", second: "The second option" } },
  score: { instructions: "How strong is it?", criteria: ["low", "medium", "high"] },
};

export function addStep(flow: Flow, type: StepType, position: { x: number; y: number }): [Flow, string] {
  let n = Object.keys(flow.steps).length + 1;
  while (flow.steps[`step_${n}`]) n++;
  const id = `step_${n}`;
  const s: Step = { id, type, position, transitions: [], ...STARTERS[type] };
  return [{ ...flow, start: flow.start in flow.steps ? flow.start : id, steps: { ...flow.steps, [id]: s } }, id];
}

export function removeStep(flow: Flow, id: string): Flow {
  const steps = Object.fromEntries(Object.entries(flow.steps).filter(([k]) => k !== id).map(([k, s]) =>
    [k, { ...s, transitions: s.transitions.filter((t) => t.target !== id) }]));
  return { ...flow, steps, start: flow.start === id ? Object.keys(steps)[0] ?? "" : flow.start };
}

export function emptyFlow(): Flow {
  return { id: "my-flow", name: "My flow", description: "", version: "1.0", start: "", steps: {} };
}

/** Parse one NDJSON chunk stream: returns the complete events and the unfinished tail. */
export function parseLines(buffer: string): [RunEvent[], string] {
  const parts = buffer.split("\n");
  const tail = parts.pop() ?? "";
  return [parts.filter((l) => l.trim()).map((l) => JSON.parse(l) as RunEvent), tail];
}
