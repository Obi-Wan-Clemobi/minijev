"use client";
import { useId } from "react";
import type { QItem } from "@/lib/store";
import type { QType, Question } from "@/lib/types";
import { levelText } from "@/lib/scoring";
import { Icon } from "./Icon";

const TYPE_LABEL: Record<QType, string> = { noul: "Noul", choice: "Choice", score: "Score" };
const input = "w-full h-9 px-2.5 rounded-md border border-line bg-card text-sm outline-none focus:border-fg transition-colors";
const iconBtn = "w-8 h-8 shrink-0 rounded-md grid place-items-center text-muted hover:text-fg hover:bg-track disabled:opacity-30 disabled:hover:bg-transparent transition-colors";

// Criteria helpers: a Choice keeps [name, description] pairs, a Score keeps ordered levels.
const optionsOf = (q: Question): [string, string][] =>
  q.type === "choice" ? Object.entries(q.criteria as Record<string, string | null>).map(([k, d]) => [k, d ?? ""]) : [];
const levelsOf = (q: Question): string[] => (q.type === "score" ? (q.criteria as unknown[]).map(levelText) : []);
const toOptions = (rows: [string, string][]) => Object.fromEntries(rows.map(([k, d]) => [k, d.trim() ? d : null]));

const rowsOf = (it: QItem): [string, string][] => it.opts ?? optionsOf(it.q);

export function problems(items: QItem[]): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  const ids = items.map((i) => i.id.trim());
  items.forEach((it) => {
    const p: string[] = [];
    if (!it.id.trim()) p.push("Give the question an id.");
    else if (ids.filter((x) => x === it.id.trim()).length > 1) p.push("Ids must be unique.");
    if (!it.q.instructions.trim()) p.push("Write the question.");
    if (it.q.type === "choice") {
      const names = rowsOf(it).map(([k]) => k.trim());
      if (names.length < 2 || names.length > 25) p.push("A Choice needs 2–25 options.");
      if (names.some((n) => !n)) p.push("Every option needs a name.");
      if (new Set(names).size !== names.length) p.push("Option names must be unique.");
    }
    if (it.q.type === "score") {
      const lv = levelsOf(it.q);
      if (lv.length < 2 || lv.length > 10) p.push("A Score needs 2–10 levels.");
      if (lv.some((l) => !l.trim())) p.push("Every level needs text.");
    }
    if (p.length) out[it.key] = p;
  });
  return out;
}

function convert(q: Question, type: QType): Question {
  if (type === q.type) return q;
  const base = { type, instructions: q.instructions };
  const names = q.type === "choice" ? Object.keys(q.criteria as object) : q.type === "score" ? levelsOf(q) : ["yes", "no"];
  if (type === "noul") return base;
  if (type === "choice") return { ...base, criteria: Object.fromEntries(names.map((n) => [n, null])) };
  return { ...base, criteria: q.type === "noul" ? ["no", "somewhat", "yes"] : names };
}

function move<T>(xs: T[], i: number, d: number): T[] {
  const j = i + d;
  if (j < 0 || j >= xs.length) return xs;
  const out = xs.slice();
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

type Props = {
  item: QItem; index: number; count: number; issues?: string[];
  onChange: (q: Question, id: string, opts: [string, string][] | undefined) => void; onRemove: () => void; onMove: (d: number) => void;
};

export function QuestionCard({ item, index, count, issues, onChange, onRemove, onMove }: Props) {
  const { q, id } = item;
  const uid = useId(); // stable across the server render and hydration, unlike item.key
  const set = (patch: Partial<Question>) => onChange({ ...q, ...patch }, id, item.opts);
  const rows = rowsOf(item);
  const setRows = (next: [string, string][]) => onChange({ ...q, criteria: toOptions(next) }, id, next);
  const nBranches = q.type === "score" && (q.score_mode ?? "pointwise") === "pointwise" ? levelsOf(q).length
    : q.type === "choice" && q.choice_mode === "pointwise" ? rows.length : 1;
  const modeKey = q.type === "choice" ? "choice_mode" : "score_mode";
  const mode = q.type === "choice" ? q.choice_mode ?? "default" : q.score_mode ?? "default";

  return (
    <section aria-label={`Question ${id}`} className={`rounded-[10px] border bg-card p-4 flex flex-col gap-3 ${issues ? "border-warn" : "border-line"}`}>
      <div className="flex items-center gap-2">
        <label className="sr-only" htmlFor={`${uid}-id`}>Question id</label>
        <input id={`${uid}-id`} value={id} onChange={(e) => onChange(q, e.target.value.replace(/\s/g, "_"), item.opts)}
          className="h-8 flex-1 min-w-0 px-2 rounded-md border border-transparent hover:border-line focus:border-fg bg-transparent font-mono text-[13px] font-medium outline-none" />
        <button className={iconBtn} aria-label="Move question up" disabled={index === 0} onClick={() => onMove(-1)}><Icon name="up" size={14} /></button>
        <button className={iconBtn} aria-label="Move question down" disabled={index === count - 1} onClick={() => onMove(1)}><Icon name="down" size={14} /></button>
        <button className={iconBtn} aria-label={`Remove question ${id}`} onClick={onRemove}><Icon name="x" size={14} /></button>
      </div>
      <div className="flex items-center gap-2 flex-wrap -mt-1">
        <label className="sr-only" htmlFor={`${uid}-type`}>Type</label>
        <select id={`${uid}-type`} value={q.type} onChange={(e) => onChange(convert(q, e.target.value as QType), id, undefined)}
          className="h-7 px-2 rounded-full border border-line bg-card text-xs font-medium text-muted outline-none">
          {(["noul", "choice", "score"] as QType[]).map((t) => <option key={t} value={t}>{TYPE_LABEL[t]}</option>)}
        </select>
        {q.type !== "noul" && (
          <select aria-label="Readout mode" value={mode}
            onChange={(e) => set({ [modeKey]: e.target.value === "default" ? undefined : e.target.value } as Partial<Question>)}
            className="h-7 px-2 rounded-full border border-line bg-card text-xs text-muted outline-none">
            <option value="default">default mode</option>
            <option value="listwise">listwise</option>
            <option value="pointwise">pointwise</option>
          </select>
        )}
        <span className="text-xs text-muted font-mono whitespace-nowrap">{nBranches} branch{nBranches === 1 ? "" : "es"}</span>
      </div>

      <label className="sr-only" htmlFor={`${uid}-text`}>Question</label>
      <textarea id={`${uid}-text`} rows={1} value={q.instructions} placeholder="Ask one question about the state"
        onChange={(e) => set({ instructions: e.target.value })}
        className="w-full resize-y min-h-9 px-2.5 py-2 rounded-md border border-line bg-card text-sm outline-none focus:border-fg" />

      {q.type === "noul" && (
        <div className="grid grid-cols-2 gap-2">
          {(["true", "false"] as const).map((k) => {
            const crit = (q.criteria ?? {}) as { true?: string; false?: string };
            return (
              <label key={k} className="flex flex-col gap-1 text-xs text-muted">
                {k === "true" ? "Yes means (optional)" : "No means (optional)"}
                <input className={input} value={crit[k] ?? ""} onChange={(e) => {
                  const next = { ...crit, [k]: e.target.value || undefined };
                  set({ criteria: next.true || next.false ? next : undefined });
                }} />
              </label>
            );
          })}
        </div>
      )}

      {q.type === "choice" && (
        <div className="flex flex-col gap-1.5">
          <div className="grid grid-cols-[20px_minmax(0,0.8fr)_minmax(0,1.4fr)_32px] gap-2 text-[11px] text-muted uppercase tracking-wider">
            <span /><span>Option</span><span>Description (optional)</span><span />
          </div>
          {rows.map(([k, d], i) => (
            <div key={i} className="grid grid-cols-[20px_minmax(0,0.8fr)_minmax(0,1.4fr)_32px] gap-2 items-center">
              <span className="font-mono text-xs text-muted text-center">{String.fromCharCode(65 + i + (i >= 8 ? 1 : 0))}</span>
              <input aria-label={`Option ${i + 1} name`} className={`${input} font-mono`} value={k}
                onChange={(e) => setRows(rows.map((r, j) => (j === i ? [e.target.value, r[1]] : r)))} />
              <input aria-label={`Option ${i + 1} description`} className={input} value={d}
                onChange={(e) => setRows(rows.map((r, j) => (j === i ? [r[0], e.target.value] : r)))} />
              <button className={iconBtn} aria-label={`Remove option ${k}`} disabled={rows.length <= 2}
                onClick={() => setRows(rows.filter((_, j) => j !== i))}><Icon name="x" size={14} /></button>
            </div>
          ))}
          <AddRow label="Add option" disabled={rows.length >= 25}
            onClick={() => setRows([...rows, [`option_${rows.length + 1}`, ""]])} />
        </div>
      )}

      {q.type === "score" && (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] text-muted uppercase tracking-wider">Levels, lowest first</span>
          {levelsOf(q).map((lv, i, levels) => (
            <div key={i} className="flex gap-2 items-center">
              <span className="w-5 font-mono text-xs text-muted text-center">{i}</span>
              <input aria-label={`Level ${i}`} className={input} value={lv}
                onChange={(e) => set({ criteria: levels.map((l, j) => (j === i ? e.target.value : l)) })} />
              <button className={iconBtn} aria-label={`Move level ${i} up`} disabled={i === 0} onClick={() => set({ criteria: move(levels, i, -1) })}><Icon name="up" size={14} /></button>
              <button className={iconBtn} aria-label={`Move level ${i} down`} disabled={i === levels.length - 1} onClick={() => set({ criteria: move(levels, i, 1) })}><Icon name="down" size={14} /></button>
              <button className={iconBtn} aria-label={`Remove level ${i}`} disabled={levels.length <= 2}
                onClick={() => set({ criteria: levels.filter((_, j) => j !== i) })}><Icon name="x" size={14} /></button>
            </div>
          ))}
          <AddRow label="Add level" disabled={levelsOf(q).length >= 10}
            onClick={() => set({ criteria: [...levelsOf(q), ""] })} />
        </div>
      )}

      {issues && (
        <ul className="text-xs text-warn flex flex-col gap-0.5">
          {issues.map((p) => <li key={p} className="flex items-center gap-1.5"><Icon name="alert" size={12} />{p}</li>)}
        </ul>
      )}
    </section>
  );
}

function AddRow({ label, onClick, disabled }: { label: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className="self-start h-8 px-2 rounded-md text-[13px] text-muted hover:text-fg hover:bg-track flex items-center gap-1.5 disabled:opacity-40 transition-colors">
      <Icon name="plus" size={14} />{label}
    </button>
  );
}

export function blankQuestion(type: QType): Question {
  if (type === "noul") return { type, instructions: "" };
  if (type === "choice") return { type, instructions: "", criteria: { option_1: null, option_2: null } };
  return { type, instructions: "", criteria: ["low", "medium", "high"] };
}
