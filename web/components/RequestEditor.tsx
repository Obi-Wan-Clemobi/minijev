"use client";
// The request editor: presets, the state, and the questions (form or raw JSON). It edits the shared store,
// so a change here shows on every page.
import { useState } from "react";
import { fromQuestions, newKey, stateToText, textToState, useStore } from "@/lib/store";
import type { QType } from "@/lib/types";
import { Icon } from "./Icon";
import { Tip } from "./Tip";
import { blankQuestion, problems, QuestionCard } from "./QuestionEditor";

export function PresetBar() {
  const s = useStore();
  return (
    <div className="flex border border-line rounded-lg p-[3px] gap-0.5 flex-wrap">
      {s.presets.map((p) => (
        <button key={p.id} onClick={() => s.loadPreset(p)}
          className={`h-[30px] px-3 rounded-[5px] text-[13px] transition-colors ${s.presetId === p.id ? "bg-track text-fg font-medium" : "text-muted hover:text-fg"}`}>{p.name}</button>
      ))}
      {!s.presets.length && <span className="h-[30px] px-3 text-[13px] text-muted grid place-items-center">start the API to load presets</span>}
      {s.canUndo && (
        <Tip k="undo">
          <button onClick={s.undoPreset} className="h-[30px] px-3 rounded-[5px] text-[13px] text-accent hover:bg-track">Undo</button>
        </Tip>
      )}
    </div>
  );
}

export function RequestEditor({ presets = false, children }: { presets?: boolean; children?: React.ReactNode }) {
  const s = useStore();
  const [view, setView] = useState<"form" | "json">("form");
  const [jsonText, setJsonText] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const issues = problems(s.items);
  const stateIsJson = typeof textToState(s.stateText) !== "string";

  const openJson = () => { setJsonText(JSON.stringify(s.req, null, 2)); setJsonError(null); setView("json"); };
  const applyJson = () => {
    try {
      const v = JSON.parse(jsonText);
      if (!v || typeof v !== "object" || !("state" in v) || typeof v.questions !== "object") throw new Error('Expected {"state": …, "questions": {…}}');
      s.setStateText(stateToText(v.state));
      s.setItems(() => fromQuestions(v.questions));
      setView("form");
    } catch (e) { setJsonError((e as Error).message); }
  };
  const add = (type: QType) => s.setItems((xs) => [...xs, { key: newKey(), id: `q${xs.length + 1}`, q: blankQuestion(type) }]);

  return (
    <div className="flex flex-col gap-5">
      {presets && <div className="flex items-center gap-2 flex-wrap"><span className="text-[13px] text-muted">Examples</span><Tip k="presets" /><PresetBar /></div>}
      <div className="flex items-center justify-between">
        <div className="flex border border-line rounded-lg p-[3px] gap-0.5">
          <button onClick={() => setView("form")} className={`h-7 px-3 rounded-[5px] text-xs ${view === "form" ? "bg-track text-fg" : "text-muted"}`}>Form</button>
          <button onClick={openJson} className={`h-7 px-3 rounded-[5px] text-xs flex items-center gap-1.5 ${view === "json" ? "bg-track text-fg" : "text-muted"}`}><Icon name="code" size={12} />JSON</button>
        </div>
        <Tip k="formJson" className="mr-auto ml-1.5" />
        {s.response && <span className="text-xs text-muted font-mono">{s.response.usage.input_tokens} tokens in the last run</span>}
      </div>

      {view === "json" ? (
        <div className="flex flex-col gap-2">
          <label htmlFor="req-json" className="text-xs text-muted">The whole request: {'{"state": …, "questions": {…}}'}</label>
          <textarea id="req-json" value={jsonText} onChange={(e) => setJsonText(e.target.value)} spellCheck={false}
            className="min-h-[520px] p-3.5 rounded-lg border border-line bg-card font-mono text-[13px] leading-relaxed outline-none focus:border-fg" />
          {jsonError && <p role="alert" className="text-xs text-warn">{jsonError}</p>}
          <div className="flex gap-2">
            <button onClick={applyJson} className="h-9 px-3 rounded-md bg-inv-bg text-inv-fg text-sm">Apply</button>
            <button onClick={() => setView("form")} className="h-9 px-3 rounded-md border border-line text-sm">Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1"><label htmlFor="state" className="text-xs font-medium text-muted uppercase tracking-wider">State · your text</label><Tip k="state" /></span>
            <textarea id="state" rows={stateIsJson ? 8 : 3} value={s.stateText} onChange={(e) => s.setStateText(e.target.value)}
              placeholder="Type or paste any text: a message, a review, a paragraph…"
              className="resize-y w-full px-3.5 py-3 rounded-lg border border-line bg-card font-mono text-[13px] leading-relaxed outline-none focus:border-fg" />
            <span className="text-xs text-muted font-mono">{stateIsJson ? "JSON state · sent as structure" : "text state"} · prefilled once, shared by every branch</span>
          </div>

          <div className="flex items-center justify-between gap-2 flex-wrap">
            <h2 className="text-xs font-medium text-muted uppercase tracking-wider">Questions · {s.items.length}</h2>
            <div className="flex gap-1.5">
              {(["noul", "choice", "score"] as QType[]).map((t) => (
                <Tip key={t} k={t === "noul" ? "addNoul" : t === "choice" ? "addChoice" : "addScore"}>
                  <button onClick={() => add(t)} className="h-8 px-2.5 rounded-md border border-line text-[13px] flex items-center gap-1.5 hover:bg-track transition-colors">
                    <Icon name="plus" size={14} />{t === "noul" ? "Yes/no" : t === "choice" ? "Choice" : "Scale"}
                  </button>
                </Tip>
              ))}
            </div>
          </div>

          {s.items.map((it, i) => (
            <QuestionCard key={it.key} item={it} index={i} count={s.items.length} issues={issues[it.key]}
              onChange={(q, id, opts) => s.setItems((xs) => xs.map((x) => (x.key === it.key ? { ...x, q, id, opts } : x)))}
              onRemove={() => s.setItems((xs) => xs.filter((x) => x.key !== it.key))}
              onMove={(d) => s.setItems((xs) => {
                const j = i + d; if (j < 0 || j >= xs.length) return xs;
                const out = xs.slice(); [out[i], out[j]] = [out[j], out[i]]; return out;
              })} />
          ))}
          {!s.items.length && <p className="text-sm text-muted">Add a question: a Noul (yes/no), a Choice (pick one option), or a Score (a level on a scale).</p>}
          {children}
        </>
      )}
    </div>
  );
}

export function useReady() {
  const s = useStore();
  const issues = problems(s.items);
  return Object.keys(issues).length === 0 && s.items.length > 0 && s.stateText.trim().length > 0 && !s.switching;
}
