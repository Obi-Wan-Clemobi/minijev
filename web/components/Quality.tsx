"use client";
// E11 on the Compare page: the same small model answers labelled questions four ways.
import { useState } from "react";
import type { HelpKey } from "@/lib/help";
import { Tip } from "./Tip";

type M = {
  accuracy: number; accuracy_ci95: [number, number]; ece: number | null; ece_ci95: [number, number] | null;
  parse_failures: number; mean_s: number; output_tokens: number; n: number; has_probs: boolean;
};
type Task = { n: number; chance: number; methods: Record<string, M> };
export type QualityData = Record<string, Record<string, Task> | null>; // model -> task -> result

const METHODS: { key: string; name: string; sub: string; help: HelpKey; color: string }[] = [
  { key: "readout", name: "Reads out (minijev)", sub: "writes nothing", help: "qReadout", color: "var(--accent)" },
  { key: "readout, calibrated", name: "Reads out, calibrated", sub: "plus the temperature dial", help: "qReadoutCal", color: "var(--accent2)" },
  { key: "writes the answer", name: "Writes the answer", sub: "e.g. \"Yes\" or \"Sports\"", help: "qWritesAnswer", color: "var(--gen)" },
  { key: "writes a probability", name: "Writes a probability", sub: "e.g. \"0.8\" or a JSON list", help: "qWritesProb", color: "var(--warn)" },
];
const TASKS = [["boolq", "Yes/no · BoolQ"], ["ag_news", "Multiple choice · AG News"]] as const;

function Interval({ v, lo, hi, min, max, color, mark }: { v: number; lo: number; hi: number; min: number; max: number; color: string; mark?: number }) {
  const pct = (x: number) => `${Math.min(100, Math.max(0, ((x - min) / (max - min)) * 100))}%`;
  return (
    <div className="relative h-6" role="img" aria-label={`${v.toFixed(3)}, interval ${lo.toFixed(3)} to ${hi.toFixed(3)}`}>
      <div className="absolute inset-x-0 top-3 h-px bg-line" />
      <div className="absolute top-[11px] h-0.5" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})`, background: color }} />
      <div className="absolute top-1.5 w-3 h-3 -ml-1.5 rounded-full" style={{ left: pct(v), background: color }} />
      {mark !== undefined && <div className="absolute -top-0.5 bottom-0 w-0.5 bg-gen opacity-70" style={{ left: pct(mark) }} />}
    </div>
  );
}

const overlap = (a: [number, number], b: [number, number]) => a[0] <= b[1] && b[0] <= a[1];

export function Quality({ data }: { data: QualityData | null }) {
  const [task, setTask] = useState<string>("boolq");
  const [model, setModel] = useState("0.5B");
  const t = data?.[model]?.[task];

  const toggle = (items: readonly (readonly [string, string])[], value: string, set: (v: string) => void) => (
    <div className="flex border border-line rounded-lg p-[3px] gap-0.5">
      {items.map(([k, l]) => (
        <button key={k} onClick={() => set(k)} aria-pressed={value === k}
          className={`h-8 px-3 rounded-[5px] text-[13px] ${value === k ? "bg-track text-fg" : "text-muted hover:text-fg"}`}>{l}</button>
      ))}
    </div>
  );

  let takeaways: string[] = [];
  if (t) {
    const r = t.methods["readout"], rc = t.methods["readout, calibrated"], w = t.methods["writes the answer"], p = t.methods["writes a probability"];
    takeaways = [
      `Reading the answer out took ${r.mean_s.toFixed(2)} s per question; writing it took ${w.mean_s.toFixed(2)} s (${(w.mean_s / r.mean_s).toFixed(1)}×), and writing a probability ${p.mean_s.toFixed(2)} s (${(p.mean_s / r.mean_s).toFixed(1)}×).`,
      overlap(r.accuracy_ci95, w.accuracy_ci95)
        ? `Accuracy: readout ${r.accuracy.toFixed(2)} vs written ${w.accuracy.toFixed(2)}. The intervals overlap, so the two are equally right: it is the same model.`
        : `Accuracy: readout ${r.accuracy.toFixed(2)} vs written ${w.accuracy.toFixed(2)}. The intervals do not overlap, so the way of asking changed the answers.`,
      rc.ece !== null && p.ece !== null
        ? `Honesty: the calibrated readout's error is ${rc.ece.toFixed(3)}; the written probabilities' error is ${p.ece.toFixed(3)}${p.parse_failures ? `, and ${p.parse_failures} of ${p.n} replies could not be read` : ""}.`
        : "",
    ].filter(Boolean);
  }

  const all = data ? (["0.5B", "1.5B"] as const).flatMap((m) => METHODS.map((me) => ({ m, me, r: data[m]?.[task]?.methods[me.key] })).filter((x) => x.r)) : [];
  const maxS = Math.max(0.1, ...all.map((x) => x.r!.mean_s));
  const accMin = t ? Math.max(0, Math.min(t.chance, ...all.map((x) => x.r!.accuracy_ci95[0])) - 0.05) : 0;

  return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-5">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-[15px] font-semibold flex items-center gap-1">2 · Which way gives better answers? Questions with known answers <Tip k="quality" /></h2>
        <div className="flex-1" />
        {toggle(TASKS, task, setTask)}<Tip k="qTask" />
        {toggle([["0.5B", "Qwen2.5-0.5B"], ["1.5B", "Qwen2.5-1.5B"]], model, setModel)}
      </div>

      {!t ? (
        <p className="text-sm text-muted">No results for this model yet. Run <code className="font-mono">cd poc && uv run python experiments.py quality{model === "1.5B" ? " --model Qwen/Qwen2.5-1.5B-Instruct" : ""}</code>, then reload.</p>
      ) : (
        <div className="grid xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] gap-6">
          <div className="flex flex-col gap-1 overflow-x-auto">
            <div className="grid grid-cols-[190px_minmax(150px,1fr)_minmax(150px,1fr)_110px_90px] gap-3 text-[11px] text-muted uppercase tracking-wider pb-1 min-w-[720px]">
              <span>{t.n} questions</span>
              <span className="flex items-center gap-1">Accuracy <Tip k="qAccuracy" /></span>
              <span className="flex items-center gap-1">Calibration error <Tip k="qHonesty" /></span>
              <span className="flex items-center gap-1">Time / question <Tip k="qTime" /></span>
              <span className="flex items-center gap-1">Written <Tip k="qWords" /></span>
            </div>
            {METHODS.map((me) => {
              const r = t.methods[me.key];
              if (!r) return null;
              return (
                <div key={me.key} className="grid grid-cols-[190px_minmax(150px,1fr)_minmax(150px,1fr)_110px_90px] gap-3 items-center py-2 border-t border-line min-w-[720px]">
                  <div className="flex flex-col">
                    <span className="text-sm flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: me.color }} />{me.name}<Tip k={me.help} /></span>
                    <span className="text-xs text-muted pl-3.5">{me.sub}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <Interval v={r.accuracy} lo={r.accuracy_ci95[0]} hi={r.accuracy_ci95[1]} min={accMin} max={1} color={me.color} mark={t.chance} />
                    <span className="text-[11px] font-mono text-muted">{r.accuracy.toFixed(3)} [{r.accuracy_ci95[0].toFixed(2)}–{r.accuracy_ci95[1].toFixed(2)}]</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    {r.ece !== null && r.ece_ci95 ? (
                      <>
                        <Interval v={r.ece} lo={r.ece_ci95[0]} hi={r.ece_ci95[1]} min={0} max={0.45} color={me.color} />
                        <span className="text-[11px] font-mono text-muted">{r.ece.toFixed(3)} [{r.ece_ci95[0].toFixed(2)}–{r.ece_ci95[1].toFixed(2)}]</span>
                      </>
                    ) : <span className="text-xs text-muted">no probability to check</span>}
                  </div>
                  <div className="flex flex-col gap-1">
                    <div className="h-2.5 rounded-sm bg-track relative"><div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${(r.mean_s / maxS) * 100}%`, background: me.color }} /></div>
                    <span className="text-[11px] font-mono">{r.mean_s.toFixed(2)} s</span>
                  </div>
                  <div className="flex flex-col text-[11px] font-mono">
                    <span>{r.output_tokens.toFixed(1)} tokens</span>
                    <span className={r.parse_failures ? "text-warn" : "text-muted"}>{r.parse_failures} unreadable</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex flex-col gap-2">
            <span className="text-[13px] font-medium flex items-center gap-1">Speed vs accuracy · both models <Tip k="qScatter" /></span>
            <svg viewBox="0 0 400 260" className="w-full max-w-[460px]" role="img" aria-label="Seconds per question against accuracy, for every method and both models">
              <line x1="44" y1="220" x2="390" y2="220" stroke="var(--line)" />
              <line x1="44" y1="220" x2="44" y2="10" stroke="var(--line)" />
              {[0, 0.5, 1].map((f) => <text key={f} x={44 + f * 346} y="236" textAnchor="middle" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">{(f * maxS).toFixed(1)}s</text>)}
              {[accMin, (accMin + 1) / 2, 1].map((a) => <text key={a} x="38" y={220 - ((a - accMin) / (1 - accMin)) * 210 + 3} textAnchor="end" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">{a.toFixed(2)}</text>)}
              <line x1="44" x2="390" y1={220 - ((t.chance - accMin) / (1 - accMin)) * 210} y2={220 - ((t.chance - accMin) / (1 - accMin)) * 210} stroke="var(--gen)" strokeDasharray="3 3" opacity="0.7" />
              <text x="215" y="254" textAnchor="middle" fontSize="10" fill="var(--muted)">seconds per question → (left is faster)</text>
              {all.map(({ m, me, r }) => {
                const x = 44 + (r!.mean_s / maxS) * 346, y = 220 - ((r!.accuracy - accMin) / (1 - accMin)) * 210;
                return m === "0.5B"
                  ? <circle key={m + me.key} cx={x} cy={y} r="6" fill={me.color} opacity={m === model ? 1 : 0.45}><title>{`${me.name}, 0.5B: ${r!.accuracy.toFixed(2)} in ${r!.mean_s.toFixed(2)} s`}</title></circle>
                  : <rect key={m + me.key} x={x - 6} y={y - 6} width="12" height="12" fill={me.color} opacity={m === model ? 1 : 0.45}><title>{`${me.name}, 1.5B: ${r!.accuracy.toFixed(2)} in ${r!.mean_s.toFixed(2)} s`}</title></rect>;
              })}
            </svg>
            <span className="text-xs text-muted">● 0.5B · ■ 1.5B · dashed orange = always guessing the most common answer</span>
          </div>

          <ul className="xl:col-span-2 flex flex-col gap-1.5 text-sm leading-relaxed list-disc pl-5">
            {takeaways.map((x) => <li key={x}>{x}</li>)}
          </ul>
        </div>
      )}
    </section>
  );
}
