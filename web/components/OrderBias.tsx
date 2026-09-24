"use client";
// E13 on the Findings page: the listwise order flaw, and three fixes compared on AG News.
import { Tip } from "./Tip";

type M = { branches: number; flip_rate: number; flip_rate_ci95: [number, number]; mean_max_dp: number;
  accuracy: number; accuracy_ci95: [number, number]; ece: number; ece_ci95: [number, number] };
export type OrderData = { n: number; orders_per_item: number;
  position: { mean_probability: number[]; model_picks: number[]; right_answer_at: number[] }; methods: Record<string, M> };

const NAMES: Record<string, [string, string, string]> = {
  listwise: ["As-is (listwise)", "all options at once; the model picks a letter", "var(--gen)"],
  debiased: ["Debiased", "divide out the model's liking for each letter", "var(--warn)"],
  averaged: ["All orders averaged", "every option in every position once", "var(--accent)"],
  pointwise: ["Pointwise", "each option judged on its own", "var(--accent2)"],
};

export function OrderBias({ data, size }: { data: OrderData | null; size: string }) {
  if (!data) return (
    <section className="rounded-xl border border-line bg-card p-6 text-sm text-muted">
      No order-bias results for {size} yet. Run <code className="font-mono">cd poc && uv run python experiments.py order_bias</code>.
    </section>
  );
  const pos = data.position, k = pos.model_picks.length;
  const maxPick = Math.max(...pos.model_picks, ...pos.right_answer_at, 0.5);
  return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2 className="text-[15px] font-semibold flex items-center gap-1">The order flaw, and three fixes · {size} <Tip k="orderFlaw" /></h2>
        <p className="text-sm text-muted max-w-[900px] leading-relaxed">
          {data.n} news articles, each asked {data.orders_per_item} times with the options in a different order. The right answer is known.
          If the model only read the content, the order would never change its answer.
        </p>
      </div>
      <div className="grid xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.4fr)] gap-6">
        <div className="flex flex-col gap-3">
          <span className="text-[13px] font-medium flex items-center gap-1">Which letter does the model pick? <Tip k="orderLetters" /></span>
          <div className="grid gap-4 items-end h-[180px] border-b border-line" style={{ gridTemplateColumns: `repeat(${k}, minmax(0, 1fr))` }}>
            {pos.model_picks.map((p, j) => (
              <div key={j} className="flex items-end justify-center gap-1 h-full">
                <div className="flex flex-col items-center justify-end gap-1 h-full">
                  <span className="text-[11px] font-mono">{Math.round(p * 100)}%</span>
                  <div className="w-6 rounded-t-sm bg-gen" style={{ height: `${(p / maxPick) * 140}px` }} />
                </div>
                <div className="flex flex-col items-center justify-end gap-1 h-full">
                  <span className="text-[11px] font-mono text-muted">{Math.round(pos.right_answer_at[j] * 100)}%</span>
                  <div className="w-6 rounded-t-sm bg-ghost" style={{ height: `${(pos.right_answer_at[j] / maxPick) * 140}px` }} />
                </div>
              </div>
            ))}
          </div>
          <div className="grid gap-4 text-center font-mono text-sm" style={{ gridTemplateColumns: `repeat(${k}, minmax(0, 1fr))` }}>
            {pos.model_picks.map((_, j) => <span key={j}>{"ABCDEFGH"[j]}</span>)}
          </div>
          <span className="text-xs text-muted flex gap-4">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-gen rounded-sm" />model picks this letter</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-ghost rounded-sm" />right answer is at this letter</span>
          </span>
        </div>
        <div className="flex flex-col gap-1 overflow-x-auto">
          <div className="grid grid-cols-[200px_70px_minmax(140px,1fr)_minmax(140px,1fr)_80px] gap-3 text-[11px] text-muted uppercase tracking-wider pb-1 min-w-[660px]">
            <span>Way of asking</span>
            <span className="flex items-center gap-1">Cost <Tip k="orderCost" /></span>
            <span className="flex items-center gap-1">Answer flips <Tip k="orderFlips" /></span>
            <span className="flex items-center gap-1">Accuracy <Tip k="qAccuracy" /></span>
            <span className="flex items-center gap-1">ECE <Tip k="fEce" /></span>
          </div>
          {Object.keys(NAMES).filter((m) => data.methods[m]).map((m) => {
            const r = data.methods[m], [name, sub, color] = NAMES[m];
            const bar = (v: number, lo: number, hi: number, max: number) => (
              <div className="flex flex-col gap-0.5">
                <div className="relative h-5">
                  <div className="absolute inset-x-0 top-2.5 h-px bg-line" />
                  <div className="absolute top-2 h-1 rounded-sm" style={{ left: `${(lo / max) * 100}%`, width: `${((hi - lo) / max) * 100}%`, background: color, opacity: 0.5 }} />
                  <div className="absolute top-1 w-3 h-3 -ml-1.5 rounded-full" style={{ left: `${(v / max) * 100}%`, background: color }} />
                </div>
                <span className="text-[11px] font-mono text-muted">{v.toFixed(3)} [{lo.toFixed(2)}–{hi.toFixed(2)}]</span>
              </div>
            );
            return (
              <div key={m} className="grid grid-cols-[200px_70px_minmax(140px,1fr)_minmax(140px,1fr)_80px] gap-3 items-center py-2 border-t border-line min-w-[660px]">
                <div className="flex flex-col"><span className="text-sm flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />{name}</span><span className="text-xs text-muted pl-4">{sub}</span></div>
                <span className="text-xs font-mono">{r.branches} branch{r.branches > 1 ? "es" : ""}</span>
                {bar(r.flip_rate, r.flip_rate_ci95[0], r.flip_rate_ci95[1], 1)}
                {bar(r.accuracy, r.accuracy_ci95[0], r.accuracy_ci95[1], 1)}
                <span className="text-xs font-mono">{r.ece.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
