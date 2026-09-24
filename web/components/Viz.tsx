"use client";
import type { HelpKey } from "@/lib/help";
import type { ChoiceAnswer, NoulAnswer, ScoreAnswer } from "@/lib/scoring";
import { Tip } from "./Tip";

const f2 = (x: number) => x.toFixed(2);

export function Ring({ value, label, help }: { value: number; label: string; help: HelpKey }) {
  const c = 2 * Math.PI * 22;
  return (
    <div className="w-[120px] shrink-0 flex flex-col items-center justify-center gap-1.5 border-l border-line">
      <svg width="72" height="72" viewBox="0 0 60 60" role="img" aria-label={`${label} ${f2(value)}`}>
        <circle cx="30" cy="30" r="22" fill="none" stroke="var(--track)" strokeWidth="6" />
        <circle className="anim" cx="30" cy="30" r="22" fill="none" stroke="var(--accent)" strokeWidth="6" strokeLinecap="round"
          strokeDasharray={`${(value * c).toFixed(1)} ${c.toFixed(1)}`} transform="rotate(-90 30 30)" />
      </svg>
      <span className="text-xl font-semibold tabular-nums">{f2(value)}</span>
      <span className="text-[11px] text-muted text-center flex items-center gap-0.5">{label}<Tip k={help} /></span>
    </div>
  );
}

export function NoulMeter({ a, raw }: { a: NoulAnswer; raw: NoulAnswer }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="relative h-3 rounded-full bg-track">
        <div className="anim absolute inset-y-0 left-0 rounded-full border border-dashed border-ghost" style={{ width: `${raw.noul * 100}%` }} />
        <div className="anim absolute inset-y-[2px] left-0 rounded-full bg-accent" style={{ width: `${a.noul * 100}%` }} />
        <div className="absolute left-1/2 -inset-y-1 w-px bg-muted" />
      </div>
      <div className="flex justify-between text-[11px] font-mono text-muted"><span>0 · No</span><span>0.5</span><span>Yes · 1</span></div>
    </div>
  );
}

export function ChoiceBars({ a, raw, descriptions }: { a: ChoiceAnswer; raw: ChoiceAnswer; descriptions: Record<string, string | null> }) {
  const rows = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]);
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map(([k, p]) => {
        const win = k === a.choice;
        return (
          <div key={k} className="grid grid-cols-[minmax(80px,140px)_minmax(0,1fr)_52px] items-center gap-3" title={descriptions[k] ?? undefined}>
            <span className={`text-sm truncate ${win ? "font-semibold" : ""}`}>{k}</span>
            <div className="relative h-[22px] rounded bg-track">
              <div className="anim absolute inset-y-0 left-0 rounded border border-dashed border-ghost" style={{ width: `${raw.probabilities[k] * 100}%` }} />
              <div className={`anim absolute inset-y-[3px] left-0 rounded-sm ${win ? "bg-accent" : "bg-ghost"}`} style={{ width: `${p * 100}%` }} />
            </div>
            <span className="font-mono text-[13px] text-right tabular-nums">{f2(p)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function ScoreColumns({ a, raw }: { a: ScoreAnswer; raw: ScoreAnswer }) {
  const levels = Object.keys(a.probabilities);
  const n = levels.length;
  const ps = levels.map((k) => a.probabilities[k]);
  const mode = ps.indexOf(Math.max(...ps));
  return (
    <div className="flex flex-col gap-2">
      <div className="grid items-end gap-3 h-[150px] border-b border-line" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
        {levels.map((k, i) => (
          <div key={k} className="flex flex-col items-center justify-end gap-1.5 h-full">
            <span className="font-mono text-xs tabular-nums">{f2(ps[i])}</span>
            <div className="relative w-full max-w-14" style={{ height: `${Math.max(ps[i], raw.probabilities[k]) * 120}px` }}>
              <div className="anim absolute inset-x-0 bottom-0 rounded-t border border-dashed border-ghost" style={{ height: `${raw.probabilities[k] * 120}px` }} />
              <div className={`anim absolute inset-x-[3px] bottom-0 rounded-t-sm ${i === mode ? "bg-accent" : "bg-ghost"}`} style={{ height: `${ps[i] * 120}px` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="relative h-5">
        <div className="anim absolute top-0 -translate-x-1/2 text-xs font-mono text-accent whitespace-nowrap"
          style={{ left: `${(a.score / (n - 1)) * (100 - 100 / n) + 50 / n}%` }}><Tip k="expected">▲ E[level] = {f2(a.score)}</Tip></div>
      </div>
      <div className="grid gap-3 text-xs text-muted text-center" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
        {levels.map((k) => <span key={k} className="truncate" title={a.legend[k]}>{k} · {a.legend[k]}</span>)}
      </div>
    </div>
  );
}
