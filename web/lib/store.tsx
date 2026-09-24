"use client";
// The request being edited, the settings, and the last response. Shared by all pages and kept in
// localStorage, so Compare and Under the hood always show the request from the Playground.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { AskResponse, Preset, Question, Req, Settings } from "./types";
import { DEFAULT_SETTINGS } from "./types";

// opts: a Choice's options as ordered [name, description] rows. The editor edits these, so a name that
// briefly repeats another one never merges two rows; q.criteria is derived from them.
export type QItem = { key: string; id: string; q: Question; opts?: [string, string][] };

let counter = 0;
export const newKey = () => `k${Date.now().toString(36)}${(counter++).toString(36)}`;

export function stateToText(state: unknown): string {
  return typeof state === "string" ? state : JSON.stringify(state, null, 2);
}

export function textToState(text: string): unknown {
  const t = text.trim();
  if (t.startsWith("{") || t.startsWith("[")) {
    try { return JSON.parse(t); } catch { /* plain text that starts with a brace */ }
  }
  return text;
}

export function toReq(stateText: string, items: QItem[]): Req {
  return { state: textToState(stateText), questions: Object.fromEntries(items.map((i) => [i.id, i.q])) };
}

export function fromQuestions(qs: Record<string, Question>): QItem[] {
  return Object.entries(qs).map(([id, q]) => ({ key: newKey(), id, q }));
}

const SUPPORT: Req = {
  state: "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
  questions: {
    urgency: { type: "noul", instructions: "Does this message express urgency?" },
    team: { type: "choice", instructions: "Which team should handle this?", criteria: {
      billing: "Payments, invoices, refunds", technical: "Integrations, bugs, outages", sales: "Pricing, new plans" } },
    tone: { type: "score", instructions: "How upset is the customer?", criteria: ["calm", "mildly annoyed", "frustrated", "angry"] },
  },
};

type Store = {
  stateText: string; setStateText: (s: string) => void;
  items: QItem[]; setItems: (f: (items: QItem[]) => QItem[]) => void;
  settings: Settings; setSettings: (f: (s: Settings) => Settings) => void;
  mode: string; setMode: (m: string) => void;
  presets: Preset[]; presetId: string; loadPreset: (p: Preset) => void;
  response: AskResponse | null; running: boolean; error: string | null;
  run: () => Promise<void>;
  stale: boolean; // the request, evaluation or default modes changed since the last run
  canUndo: boolean; undoPreset: () => void;
  model: string; models: string[]; switchModel: (name: string) => Promise<void>; switching: boolean;
  req: Req;
};

const Ctx = createContext<Store | null>(null);

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [stateText, setStateText] = useState(SUPPORT.state as string);
  const [items, setItemsRaw] = useState<QItem[]>(() => fromQuestions(SUPPORT.questions));
  const [settings, setSettingsRaw] = useState<Settings>(DEFAULT_SETTINGS);
  const [mode, setMode] = useState("packed");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetId, setPresetId] = useState("support");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [switching, setSwitching] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [undo, setUndo] = useState<{ stateText: string; items: QItem[]; presetId: string } | null>(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("mj-request") ?? "null");
      if (saved) {
        // Restoring the saved request after hydration is the point of this effect; the server render has no storage.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setStateText(saved.stateText); setItemsRaw(saved.items); setSettingsRaw({ ...DEFAULT_SETTINGS, ...saved.settings });
        setPresetId(saved.presetId ?? ""); setMode(saved.mode ?? "packed");
      }
    } catch { /* storage unavailable: start from the default request */ }
    setLoaded(true);
  }, []);

  // Ask the API for its model and presets; if it is offline or busy (starting, or loading a model), ask again every 3 s.
  useEffect(() => {
    if (model) return;
    let stop = false;
    const connect = () => api.model()
      .then((m) => { if (stop) return; setModel(m.model); setModels(m.available); setError(null); api.presets().then(setPresets).catch(() => {}); })
      .catch(() => { if (!stop) timer = setTimeout(connect, 3000); });
    let timer = setTimeout(connect, 0);
    return () => { stop = true; clearTimeout(timer); };
  }, [model]);

  useEffect(() => {
    if (!loaded) return;
    try { localStorage.setItem("mj-request", JSON.stringify({ stateText, items, settings, presetId, mode })); } catch { /* ignore */ }
  }, [loaded, stateText, items, settings, presetId, mode]);

  const req = useMemo(() => toReq(stateText, items), [stateText, items]);
  // Everything that changes what the model computes. The temperatures and the bias are not in it: they act instantly.
  const runKey = JSON.stringify({ req, mode, c: settings.choice_mode, s: settings.score_mode, model });

  const run = useCallback(async () => {
    setRunning(true); setError(null);
    const key = runKey;
    try { setResponse(await api.ask(req, settings, mode)); setLastRun(key); }
    catch (e) { setError((e as Error).message); }
    finally { setRunning(false); }
  }, [req, settings, mode, runKey]);

  const store: Store = {
    stateText, setStateText: (s) => { setStateText(s); setPresetId(""); setUndo(null); },
    items, setItems: (f) => { setItemsRaw(f); setPresetId(""); setUndo(null); },
    settings, setSettings: setSettingsRaw,
    mode, setMode,
    presets, presetId,
    loadPreset: (p) => {
      setUndo((u) => u ?? { stateText, items, presetId });
      setStateText(stateToText(p.state)); setItemsRaw(fromQuestions(p.questions)); setPresetId(p.id);
    },
    canUndo: undo !== null,
    undoPreset: () => { if (undo) { setStateText(undo.stateText); setItemsRaw(undo.items); setPresetId(undo.presetId); setUndo(null); } },
    response, running, error, run,
    stale: response !== null && lastRun !== runKey,
    model, models, switching,
    switchModel: async (name) => {
      setSwitching(true); setError(null);
      try { setModel((await api.setModel(name)).model); setResponse(null); }
      catch (e) { setError((e as Error).message); }
      finally { setSwitching(false); }
    },
    req,
  };
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside StoreProvider");
  return s;
}
