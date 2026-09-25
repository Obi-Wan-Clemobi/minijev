"use client";
// The side panel for one step: its question, options or levels, and its arrows (answer, target, condition, order).
import { Tip } from "@/components/Tip";
import { answerLabel, answersOf, DONE, type Answer, type Flow, type Step, type Transition, updateStep } from "@/lib/flow";

const input = "w-full h-8 px-2 rounded-md border border-line bg-bg text-[13px]";
const small = "h-7 px-2 rounded-md border border-line text-xs text-muted hover:text-fg";

function num(v: string): number | null {
  const x = parseFloat(v);
  return Number.isFinite(x) ? Math.min(Math.max(x, 0), 1) : null;
}

export function StepEditor({ flow, step, focus, onChange, onDelete, onClose }: {
  flow: Flow; step: Step; focus: number | null;
  onChange: (f: Flow) => void; onDelete: () => void; onClose: () => void;
}) {
  const set = (patch: Partial<Step>) => onChange(updateStep(flow, step.id, patch));
  const setT = (ts: Transition[]) => set({ transitions: ts });
  const crit = step.criteria;
  const answers = answersOf(step);
  const targets = [...Object.keys(flow.steps).filter((k) => k !== step.id), DONE];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <h2 className="text-[15px] font-semibold m-0 flex items-center gap-1">Step <span className="font-mono text-muted text-sm">{step.id}</span> <Tip k="smStep" /></h2>
        <button className={`${small} ml-auto`} onClick={onClose}>Close</button>
      </div>
      <div className="flex gap-2">
        {flow.start === step.id
          ? <span className="text-xs text-ok">Every run starts here.</span>
          : <button className={small} onClick={() => onChange({ ...flow, start: step.id })}>Start here</button>}
        <button className={`${small} ml-auto !text-warn`} onClick={onDelete}>Delete step</button>
      </div>

      <label className="flex flex-col gap-1 text-xs text-muted">Question
        <textarea className="px-2 py-1.5 rounded-md border border-line bg-bg text-[13px] text-fg min-h-[64px]" value={step.instructions}
          onChange={(e) => set({ instructions: e.target.value })} />
      </label>

      {step.type === "noul" && (
        <div className="flex flex-col gap-2 text-xs text-muted">
          <span>What yes and no mean (optional; it helps a vague question)</span>
          {(["true", "false"] as const).map((k) => (
            <label key={k} className="flex items-center gap-2">
              <span className="w-8">{k === "true" ? "Yes" : "No"}</span>
              <input className={input} value={(crit as Record<string, string> | null)?.[k] ?? ""}
                onChange={(e) => {
                  const c = { ...((crit as Record<string, string>) ?? {}), [k]: e.target.value };
                  set({ criteria: c.true || c.false ? c : null });
                }} />
            </label>
          ))}
        </div>
      )}

      {step.type === "choice" && (
        <div className="flex flex-col gap-2 text-xs text-muted">
          <span>Options (name: what it means)</span>
          {Object.entries((crit as Record<string, string | null>) ?? {}).map(([k, v], i, all) => (
            <div key={i} className="flex gap-1.5">
              <input className={`${input} w-[110px] font-mono`} value={k} onChange={(e) => {
                const name = e.target.value.trim();
                if (!name || (name !== k && all.some(([o]) => o === name))) return;
                set({ criteria: Object.fromEntries(all.map(([o, d]) => [o === k ? name : o, d])),
                  transitions: step.transitions.map((t) => (t.from_answer === k ? { ...t, from_answer: name } : t)) });
              }} />
              <input className={input} value={v ?? ""} onChange={(e) => set({ criteria: Object.fromEntries(all.map(([o, d]) => [o, o === k ? e.target.value : d])) })} />
              <button className={small} disabled={all.length <= 2} onClick={() => set({ criteria: Object.fromEntries(all.filter(([o]) => o !== k)) })}>✕</button>
            </div>
          ))}
          <button className={small} onClick={() => {
            const c = { ...((crit as Record<string, string | null>) ?? {}) };
            let n = Object.keys(c).length + 1;
            while (c[`option_${n}`] !== undefined) n++;
            set({ criteria: { ...c, [`option_${n}`]: "" } });
          }}>+ option</button>
        </div>
      )}

      {step.type === "score" && (
        <div className="flex flex-col gap-2 text-xs text-muted">
          <span>Levels, from lowest (0) to highest</span>
          {((crit as string[]) ?? []).map((lv, i, all) => (
            <div key={i} className="flex gap-1.5 items-center">
              <span className="w-4 font-mono">{i}</span>
              <input className={input} value={lv} onChange={(e) => set({ criteria: all.map((x, j) => (j === i ? e.target.value : x)) })} />
              <button className={small} disabled={all.length <= 2} onClick={() => set({ criteria: all.filter((_, j) => j !== i) })}>✕</button>
            </div>
          ))}
          <button className={small} disabled={((crit as string[]) ?? []).length >= 10} onClick={() => set({ criteria: [...((crit as string[]) ?? []), "new level"] })}>+ level</button>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <span className="text-[13px] font-medium flex items-center gap-1">Arrows, checked top to bottom <Tip k="smArrow" /></span>
        {step.transitions.length === 0 && <p className="m-0 text-xs text-muted">No arrows yet. Drag from a dot on the right of the box to another box, or add one here.</p>}
        {step.transitions.map((t, i) => {
          const upd = (patch: Partial<Transition>) => setT(step.transitions.map((x, j) => (j === i ? { ...x, ...patch } : x)));
          const c = t.condition ?? {};
          return (
            <div key={i} className={`rounded-lg border p-2 flex flex-col gap-1.5 text-xs ${focus === i ? "border-accent" : "border-line"}`}>
              <div className="flex items-center gap-1.5">
                <span className="text-muted w-5">{i + 1}.</span>
                <span className="text-muted">if</span>
                <select className={`${input} !w-auto`} value={t.from_answer == null ? "*" : String(t.from_answer)}
                  onChange={(e) => upd({ from_answer: e.target.value === "*" ? null : answers.find((a) => String(a) === e.target.value) as Answer })}>
                  <option value="*">any answer</option>
                  {answers.map((a) => <option key={String(a)} value={String(a)}>{answerLabel(step, a)}</option>)}
                </select>
                <span className="text-muted">→</span>
                <select className={`${input} !w-auto flex-1`} value={t.target} onChange={(e) => upd({ target: e.target.value })}>
                  {targets.map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </div>
              <div className="flex items-center gap-1.5 pl-6">
                <span className="text-muted flex items-center gap-1 whitespace-nowrap">if sure <Tip k="smSure" /></span>
                <span className="text-muted">≥</span>
                <input type="number" min={0} max={1} step={0.05} className={`${input} !w-16 font-mono`} placeholder="–" value={c.confidence_gte ?? ""}
                  onChange={(e) => upd({ condition: { ...c, confidence_gte: num(e.target.value) } })} />
                <span className="text-muted">and &lt;</span>
                <input type="number" min={0} max={1} step={0.05} className={`${input} !w-16 font-mono`} placeholder="–" value={c.confidence_lt ?? ""}
                  onChange={(e) => upd({ condition: { ...c, confidence_lt: num(e.target.value) } })} />
                <span className="ml-auto flex gap-1">
                  <button className={small} disabled={i === 0} aria-label="Move up" onClick={() => { const ts = [...step.transitions]; [ts[i - 1], ts[i]] = [ts[i], ts[i - 1]]; setT(ts); }}>↑</button>
                  <button className={small} aria-label="Delete arrow" onClick={() => setT(step.transitions.filter((_, j) => j !== i))}>✕</button>
                </span>
              </div>
            </div>
          );
        })}
        <button className={small} onClick={() => setT([...step.transitions, { from_answer: null, target: DONE }])}>+ arrow</button>
      </div>
    </div>
  );
}
