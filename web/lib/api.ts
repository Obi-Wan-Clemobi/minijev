import type { AskResponse, Preset, Req, Settings } from "./types";

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(API + path, { ...init, headers: { "content-type": "application/json", ...init?.headers } });
  } catch {
    throw new Error(`Cannot reach the minijev API at ${API}. Start it: cd poc && uv run uvicorn server:app --port 8000`);
  }
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    const detail = (body as { detail?: unknown }).detail;
    throw new Error(typeof detail === "string" ? detail : `${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<T>;
}

const post = (body: unknown) => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  ask: (req: Req, settings: Settings, mode = "packed") => call<AskResponse>("/v1/ask", post({ ...req, settings, mode })),
  tree: (req: Req, settings: Settings) => call<TreeResponse>("/v1/tree", post({ ...req, settings })),
  compare: (req: Req, methods: string[]) => call<CompareResponse>("/v1/compare", post({ ...req, methods })),
  sameFormat: (req: Req, settings: Settings) => call<SameFormatResponse>("/v1/same_format", post({ ...req, settings })),
  presets: () => call<Preset[]>("/v1/presets"),
  results: () => call<Record<string, any>>("/v1/results"), // eslint-disable-line @typescript-eslint/no-explicit-any
  model: () => call<{ model: string; available: string[] }>("/v1/model"),
  setModel: (name: string) => call<{ model: string }>("/v1/model", post({ name })),
};

export type TreeBranch = {
  question: string; label: string; type: string; pointwise: boolean;
  length: number; start: number; positions: [number, number]; tokens: string[];
};
export type TreeResponse = { prefix: { length: number; tokens: string[]; state_span: [number, number] }; branches: TreeBranch[]; total: number; max_position: number };

export type MethodResult = {
  seconds: number; output_tokens: number; answers: Record<string, string | null>;
  agrees_with_readout: number; parse_failures: number;
};
export type CompareResponse = { model: string; questions: Record<string, unknown>; methods: Record<string, MethodResult> };

export type Written = { ok: true; noul?: number; choice?: string; score?: number; probabilities?: Record<string, number>; sums_to_1?: boolean }
  | { ok: false; problem: string };
export type SameFormatResponse = {
  model: string;
  readout: { seconds: number; output_tokens: number; input_tokens: number; response: Record<string, any> }; // eslint-disable-line @typescript-eslint/no-explicit-any
  written: { seconds: number; output_tokens: number; prompt_tokens: number; token_budget: number; text: string;
    valid_json: boolean; answers: Record<string, Written>; usable: number };
  structured: { seconds: number; output_tokens: number; forced_tokens: number; prompt_tokens: number; text: string;
    valid_json: boolean; answers: Record<string, Written>; usable: number };
  compare: Record<string, { agree: boolean; note: string }>;
  compare_structured: Record<string, { agree: boolean; note: string }>;
  prompt: string;
};
