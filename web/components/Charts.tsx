"use client";
// Small shared chart pieces: horizontal race bars with whiskers, a savings waterfall, an interval row.

export type RaceRow = { name: string; sub: string; sec: number; lo?: number; hi?: number; note: string; tone: "accent" | "accent2" | "gen" };
const TONE = { accent: "bg-accent", accent2: "bg-accent2", gen: "bg-gen" };

export function RaceBars({ rows }: { rows: RaceRow[] }) {
  const max = Math.max(...rows.map((r) => r.hi ?? r.sec));
  return (
    <div className="flex flex-col gap-1">
      {rows.map((r) => (
        <div key={r.name} className="grid grid-cols-[minmax(150px,230px)_minmax(0,1fr)_120px] items-center gap-3.5 min-h-[52px]">
          <div className="flex flex-col gap-0.5 min-w-0">
            <span className={`text-sm truncate ${r.tone === "accent" ? "font-semibold" : ""}`}>{r.name}</span>
            <span className="text-xs text-muted truncate">{r.sub}</span>
          </div>
          <div className="relative h-[26px]" role="img" aria-label={`${r.name}: ${r.sec.toFixed(2)} seconds`}>
            <div className={`anim absolute inset-y-0 left-0 rounded ${TONE[r.tone]}`} style={{ width: `${(r.sec / max) * 100}%` }} />
            {r.lo !== undefined && r.hi !== undefined && (
              <div className="anim absolute top-3 h-0.5 bg-fg opacity-55" style={{ left: `${(r.lo / max) * 100}%`, width: `${((r.hi - r.lo) / max) * 100}%` }} />
            )}
          </div>
          <div className="flex flex-col items-end gap-0.5 font-mono">
            <span className="text-sm tabular-nums">{r.sec.toFixed(2)} s</span>
            <span className="text-[11px] text-muted text-right">{r.note}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function Waterfall({ steps, labels }: { steps: number[]; labels: string[] }) {
  const top = steps[0];
  const bars = steps.map((v, i) => {
    if (i === 0) return { bottom: 0, height: 100, tone: "bg-gen", text: `${v.toFixed(1)} s`, at: 100 };
    if (i === steps.length - 1) return { bottom: 0, height: (v / top) * 100, tone: "bg-accent", text: `${v.toFixed(1)} s`, at: (v / top) * 100 };
    const prev = steps[i - 1];
    return { bottom: (v / top) * 100, height: ((prev - v) / top) * 100, tone: i === 1 ? "bg-gen" : "bg-accent2", text: `−${(prev - v).toFixed(1)} s`, at: (prev / top) * 100 };
  });
  // The last bar is the result; the middle bars are the savings.
  bars[steps.length - 1] = { ...bars[steps.length - 1] };
  return (
    <div className="flex flex-col gap-2">
      <div className="grid gap-3.5 h-[230px] items-end border-b border-line" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}>
        {bars.map((b, i) => (
          <div key={i} className="relative h-full">
            <div className={`anim absolute inset-x-0 rounded ${b.tone}`} style={{ bottom: `${b.bottom}%`, height: `${b.height}%` }} />
            <div className="anim absolute inset-x-0 text-center font-mono text-xs pb-1" style={{ bottom: `${Math.min(b.at, 92)}%` }}>{b.text}</div>
          </div>
        ))}
      </div>
      <div className="grid gap-3.5 text-xs text-muted text-center leading-snug" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}>
        {labels.map((l) => <span key={l}>{l}</span>)}
      </div>
    </div>
  );
}

export function IntervalRow({ name, value, lo, hi, max, min = 0, tone = "bg-accent", mark }: {
  name: string; value: number; lo: number; hi: number; max: number; min?: number; tone?: string; mark?: number;
}) {
  const pct = (x: number) => `${((x - min) / (max - min)) * 100}%`;
  return (
    <div className="grid grid-cols-[minmax(140px,220px)_minmax(0,1fr)_170px] items-center gap-3.5 h-[30px]">
      <span className="text-[13px] truncate">{name}</span>
      <div className="relative h-[30px] border-l border-line" role="img" aria-label={`${name}: ${value.toFixed(3)}, interval ${lo.toFixed(3)} to ${hi.toFixed(3)}`}>
        <div className={`absolute top-3.5 h-0.5 ${tone}`} style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
        <div className={`absolute top-[9px] w-3 h-3 -ml-1.5 rounded-full ${tone}`} style={{ left: pct(value) }} />
        {mark !== undefined && <div className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-gen" style={{ left: pct(mark) }} />}
      </div>
      <span className="text-xs font-mono text-right">{value.toFixed(3)} [{lo.toFixed(3)}–{hi.toFixed(3)}]</span>
    </div>
  );
}
