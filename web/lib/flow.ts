// Flows (the State machine page): the flow JSON of src/minijev/flows.py, and its conversion to React Flow nodes and
// edges. The flow JSON is the only source of truth; the canvas is drawn from it and every edit changes it.

export type Answer = string | boolean | number;
export type Condition = { confidence_gte?: number | null; confidence_lt?: number | null };
// from_answer missing or null = any answer (a fallback arrow). from_question names a fan-out question of the step;
// missing or null tests the step's own question.
export type Transition = { from_answer?: Answer | null; from_question?: string | null; target: string; condition?: Condition | null };
export type StepType = "noul" | "choice" | "score";
export type Criteria = Record<string, string | null> | string[] | null;
// A fan-out question: asked in the same request (one forward pass) as its step's own question.
export type SubQuestion = { id: string; type: StepType; instructions: string; criteria?: Criteria };
// A Choice step's options from an earlier step's answer: map[String(answer)] is the options of that answer.
export type OptionsFrom = { step: string; map: Record<string, Record<string, string | null>> };
export type Step = {
  id: string; type: StepType; instructions: string;
  criteria?: Criteria; options_from?: OptionsFrom | null; fanout?: SubQuestion[];
  position: { x: number; y: number }; transitions: Transition[];
};
export type Flow = { id: string; name: string; description: string; version: string; start: string; steps: Record<string, Step> };

// One answered question. question_id is the step id for the step's own question, or a fan-out question id; only
// the own question's decision has transition and next. state is the request (the same for every step of a run).
export type Decision = {
  step: string; question_id?: string; type: StepType; question: string; answer: Answer; confidence: number;
  probabilities: Record<string, number> | null; ranked?: [string, number][] | null;
  transition?: number | null; next?: string | null; ms: number; state: unknown;
};
export type Usage = { input_tokens: number; requests: number };
export type RunEnd = {
  event: "end"; status: "completed" | "escalated" | "no_transition" | "max_steps" | "invalid" | "error"; step: string | null;
  decisions: Decision[]; message?: string; path?: string[]; usage?: Usage; confidence?: number; condition?: Condition | null;
  errors?: { step: string | null; message: string }[]; warnings?: { step: string | null; message: string }[];
};
export type RunEvent = ({ event: "decision" } & Decision) | RunEnd;

export const DONE = "DONE";
export const ESCALATE = "ESCALATE"; // the hand-off: the large model makes the decision
export const TERMINALS = [DONE, ESCALATE];
export const ANY = "*"; // the handle for "any answer" (from_answer: null)

/** The options a Choice can offer: its criteria, or for options_from every mapped option. */
export function optionsOf(s: Step | SubQuestion): Record<string, string | null> {
  if ("options_from" in s && s.options_from) return Object.assign({}, ...Object.values(s.options_from.map));
  return (s.criteria as Record<string, string | null>) ?? {};
}

/** Every answer a step (or fan-out question) can give, in order: yes/no, the option keys, or the level numbers. */
export function answersOf(s: Step | SubQuestion): Answer[] {
  if (s.type === "noul") return [true, false];
  if (s.type === "choice") return Object.keys(optionsOf(s));
  return ((s.criteria as string[]) ?? []).map((_, i) => i);
}

/** An output handle: "a:<answer>" (own question), "q:<fan-out id>:<answer>", or "*" / "q:<id>:*" for any answer. */
export const handleId = (a?: Answer | null, question?: string | null) =>
  question ? `q:${question}:${a == null ? ANY : String(a)}` : a == null ? ANY : `a:${String(a)}`;

/** The (fan-out question, answer) a handle stands for. */
export function fromHandle(s: Step, handle: string | null | undefined): { from_question: string | null; from_answer: Answer | null } {
  if (!handle || handle === ANY) return { from_question: null, from_answer: null };
  if (handle.startsWith("q:")) {
    const rest = handle.slice(2), qid = rest.slice(0, rest.indexOf(":")), a = rest.slice(rest.indexOf(":") + 1);
    const q = (s.fanout ?? []).find((f) => f.id === qid);
    return { from_question: qid, from_answer: a === ANY || !q ? null : answersOf(q).find((x) => String(x) === a) ?? null };
  }
  return { from_question: null, from_answer: answersOf(s).find((a) => handleId(a) === handle) ?? null };
}

export function answerFromHandle(s: Step, handle: string | null | undefined): Answer | null {
  return fromHandle(s, handle).from_answer;
}

/** The decision of a step's own question (fan-out decisions have another question_id). */
export const isOwn = (d: Decision) => d.question_id == null || d.question_id === d.step;

/** How an answer is shown: "yes"/"no", the option key, or "level 2 · frustrated". */
export function answerLabel(s: Step | SubQuestion | undefined, a?: Answer | null): string {
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
  const q = t.from_question ? (s.fanout ?? []).find((f) => f.id === t.from_question) : undefined;
  const a = t.from_question ? `${t.from_question}: ${answerLabel(q, t.from_answer)}` : answerLabel(s, t.from_answer);
  return c ? `${a} · ${c}` : a;
}

export type StepStatus = "idle" | "visited" | "current" | "failed";
export type NodeData = { step?: Step; start?: boolean; status: StepStatus; decision?: Decision; problem?: string };

/** React Flow nodes: one per step, plus DONE, plus ESCALATE when an arrow goes there. status comes from the run. */
export function toNodes(flow: Flow, decisions: Decision[], running: string | null, end: RunEnd | null, selected: string | null) {
  const last = new Map(decisions.filter(isOwn).map((d) => [d.step, d]));
  const xs = Object.values(flow.steps).map((s) => s.position.x);
  const nodes: { id: string; type: string; position: { x: number; y: number }; selected: boolean; data: NodeData }[] = Object.values(flow.steps).map((s) => ({
    id: s.id, type: "step", position: s.position, selected: s.id === selected,
    data: {
      step: s, start: s.id === flow.start, decision: last.get(s.id),
      status: (end && end.step === s.id && end.status !== "completed" && end.status !== "escalated" ? "failed"
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
  if (Object.values(flow.steps).some((s) => s.transitions.some((t) => t.target === ESCALATE))) nodes.push({
    id: ESCALATE, type: "escalate", selected: false,
    position: { x: (xs.length ? Math.max(...xs) : 0) + 360, y: (Object.values(flow.steps)[0]?.position.y ?? 0) + 120 },
    data: { status: end?.status === "escalated" ? "visited" : "idle" },
  });
  return nodes;
}

/** React Flow edges: one per transition, from the answer's handle. Taken edges are highlighted. */
export function toEdges(flow: Flow, decisions: Decision[]) {
  const taken = new Set(decisions.filter((d) => isOwn(d) && d.transition != null).map((d) => `${d.step}:${d.transition}`));
  return Object.values(flow.steps).flatMap((s) => s.transitions.map((t, i) => {
    const id = `${s.id}:${i}`, on = taken.has(id);
    return {
      id, source: s.id, sourceHandle: handleId(t.from_answer, t.from_question), target: t.target, label: edgeLabel(s, t),
      animated: on, data: { taken: on, conditional: !!conditionLabel(t.condition) },
      style: { strokeWidth: on ? 2.5 : 1.5, stroke: on ? "var(--accent)" : "var(--ghost)", strokeDasharray: conditionLabel(t.condition) ? "6 4" : undefined },
    };
  }));
}

// ---- edits: each returns a new flow

export function connect(flow: Flow, source: string, handle: string | null | undefined, target: string): Flow {
  const s = flow.steps[source];
  if (!s || source === target) return flow;
  const { from_question, from_answer } = fromHandle(s, handle);
  const t: Transition = from_question ? { from_answer, from_question, target } : { from_answer, target };
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
  if (patch.criteria !== undefined || patch.options_from !== undefined || patch.fanout !== undefined) {
    // arrows from questions or answers that no longer exist would be invalid
    const ok = new Set(answersOf(next).map(String));
    const fan = new Map((next.fanout ?? []).map((q) => [q.id, new Set(answersOf(q).map(String))]));
    next.transitions = next.transitions.filter((t) => t.from_question
      ? fan.has(t.from_question) && (t.from_answer == null || fan.get(t.from_question)!.has(String(t.from_answer)))
      : t.from_answer == null || ok.has(String(t.from_answer)));
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
