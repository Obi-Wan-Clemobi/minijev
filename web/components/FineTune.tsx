"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */
// E20 on the Findings page: a LoRA fine-tune with option shuffles (poc/train_lora.py), base vs fine-tuned on test.
import { Tip } from "./Tip";

type Delta = { mean: number; ci95: [number, number] };

function LossCurve({ loss, perEpoch }: { loss: number[]; perEpoch: number }) {
  const w = 460, h = 170, max = 2; // a few single steps reach ~3; the axis stops at 2 so the trend stays readable
  const x = (i: number) => 36 + (i / Math.max(loss.length - 1, 1)) * (w - 46);
  const y = (v: number) => 10 + (1 - Math.min(v, max) / max) * (h - 34);
  const smooth = loss.map((_, i) => { const s = loss.slice(Math.max(0, i - 9), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
  const path = (vs: number[]) => vs.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-[460px]" role="img" aria-label="Training loss per step">
      <line x1="36" y1={h - 24} x2={w - 10} y2={h - 24} stroke="var(--line)" />
      {[0, 0.5, 1, 1.5, 2].map((v) => <text key={v} x="30" y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--muted)" fontFamily="var(--font-geist-mono)">{v.toFixed(1)}</text>)}
      {[0.69, 1.39].filter((v) => v <= max).map((v) => <line key={v} x1="36" x2={w - 10} y1={y(v)} y2={y(v)} stroke="var(--muted)" strokeDasharray="3 4" opacity="0.5" />)}
      {perEpoch > 0 && perEpoch < loss.length && <line x1={x(perEpoch)} x2={x(perEpoch)} y1="10" y2={h - 24} stroke="var(--line)" strokeDasharray="2 3" />}
      <path d={path(loss)} fill="none" stroke="var(--gen)" strokeWidth="1" opacity="0.35" />
      <path d={path(smooth)} fill="none" stroke="var(--accent)" strokeWidth="2" />
      <text x={(36 + w) / 2} y={h - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">optimizer step (8 examples each){perEpoch > 0 ? " · dotted line = second pass starts" : ""}</text>
    </svg>
  );
}

export function FineTune({ data, size }: { data: any; size: string }) {
  const intro = (
    <div className="grid md:grid-cols-3 gap-4">
      {[
        ["What changed", "ftLora", "The model itself. We trained a small add-on (LoRA) on labelled examples. Everything else on this page only changes how we ask."],
        ["What it saw", "ftShuffle", "500 yes/no questions (BoolQ) and 600 news articles (AG News) from the train split. Each article came back in new option orders."],
        ["How we test it", "ftSplits", "The base and the fine-tuned model answer the same held-out test items: 300 questions, 400 articles. Nothing was chosen on test."],
      ].map(([t, k, body]) => (
        <div key={t} className="rounded-lg border border-line p-4 flex flex-col gap-1.5">
          <span className="text-[13px] font-medium flex items-center gap-1">{t} <Tip k={k as any} /></span>
          <p className="m-0 text-xs text-muted leading-relaxed">{body}</p>
        </div>
      ))}
    </div>
  );
  const head = (
    <div className="flex flex-col gap-1">
      <h2 className="text-[15px] font-semibold flex items-center gap-1">Teaching the model: a LoRA fine-tune · {size} <Tip k="ftWhat" /></h2>
      <p className="text-sm text-muted max-w-[900px] leading-relaxed">
        Can a little training fix what prompting could not? We trained the model on the right answers, with the options shuffled,
        and compared it with the untrained model on questions it never saw.
      </p>
    </div>
  );
  if (!data) return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-5">
      {head}{intro}
      <p className="text-sm text-muted">No fine-tune results for {size} yet. Run <code className="font-mono">cd poc && uv run --group train python train_lora.py train</code>, then <code className="font-mono">uv run --group train python experiments.py lora --adapter adapters/&lt;model&gt;/epoch-&lt;n&gt;</code>.</p>
    </section>
  );

  const b = data.base, l = data.lora, d: Record<string, Delta> = data.delta_lora_minus_base;
  const rows: { name: string; base: number; tuned: number; delta?: Delta; better: "up" | "down"; tip?: string }[] = [
    { name: "Yes/no questions · accuracy", base: b.noul.raw.accuracy, tuned: l.noul.raw.accuracy, delta: d.noul_correct, better: "up" },
    { name: "Yes/no questions · log loss", base: b.noul.raw.nll, tuned: l.noul.raw.nll, delta: d.noul_nll, better: "down" },
    { name: "Yes/no questions · ECE", base: b.noul.raw.ece, tuned: l.noul.raw.ece, better: "down" },
    { name: "News topic · accuracy", base: b.choice.listwise.raw.accuracy, tuned: l.choice.listwise.raw.accuracy, delta: d.listwise_correct, better: "up" },
    { name: "News topic · ECE", base: b.choice.listwise.raw.ece, tuned: l.choice.listwise.raw.ece, better: "down" },
    { name: "Answer flips when the order changes", base: b.order_flips.flip_rate, tuned: l.order_flips.flip_rate, delta: d.flip, better: "down" },
  ];
  const tone = (r: typeof rows[number]) => {
    if (!r.delta) return "text-muted";
    const [lo, hi] = r.delta.ci95;
    if (lo <= 0 && hi >= 0) return "text-muted";
    return (r.delta.mean > 0) === (r.better === "up") ? "text-accent" : "text-warn";
  };
  const picks = (m: any) => Object.values(m.order_flips.position_picks) as number[];
  const tr = data.training;
  const perEpoch = tr ? Math.round(tr.loss_per_step.length / tr.hyper.epochs) : 0;

  return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-5">
      {head}{intro}
      <div className="grid xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-6">
        <div className="flex flex-col gap-1 overflow-x-auto">
          <div className="grid grid-cols-[minmax(200px,1fr)_80px_80px_170px] gap-3 text-[11px] text-muted uppercase tracking-wider pb-1 min-w-[560px]">
            <span>Test result (raw, no temperature)</span><span className="text-right">Before</span><span className="text-right">After</span>
            <span className="flex items-center gap-1 justify-end">Change [95%] <Tip k="ftSplits" /></span>
          </div>
          {rows.map((r) => (
            <div key={r.name} className="grid grid-cols-[minmax(200px,1fr)_80px_80px_170px] gap-3 items-center py-2 border-t border-line text-[13px] min-w-[560px]">
              <span>{r.name} <span className="text-xs text-muted">({r.better === "up" ? "higher" : "lower"} is better)</span></span>
              <span className="font-mono text-right">{r.base.toFixed(3)}</span>
              <span className="font-mono text-right">{r.tuned.toFixed(3)}</span>
              <span className={`font-mono text-right text-xs ${tone(r)}`}>
                {r.delta ? `${r.delta.mean >= 0 ? "+" : ""}${r.delta.mean.toFixed(3)} [${r.delta.ci95[0].toFixed(2)}, ${r.delta.ci95[1].toFixed(2)}]` : `${r.tuned - r.base >= 0 ? "+" : ""}${(r.tuned - r.base).toFixed(3)}`}
              </span>
            </div>
          ))}
          <p className="text-xs text-muted leading-normal pt-2">
            Green: a clear improvement. Orange: clearly worse. Grey: the interval contains 0, so the change is not clear (or no interval).
            Flips: each test article asked with its 4 options in 4 rotated orders.
            Temperature that val says the answers need (1.0 = already honest): yes/no {b.noul.T.toFixed(2)} → {l.noul.T.toFixed(2)}, news {b.choice.listwise.T.toFixed(2)} → {l.choice.listwise.T.toFixed(2)}. Pointwise was never trained; after training its news accuracy is {l.choice.pointwise.raw.accuracy.toFixed(3)} (before {b.choice.pointwise.raw.accuracy.toFixed(3)}).
          </p>
        </div>
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <span className="text-[13px] font-medium flex items-center gap-1">Which letter does it pick? (right answer: 25% each) <Tip k="ftShuffle" /></span>
            {[["before", b, "bg-gen"], ["after", l, "bg-accent"]].map(([name, m, c]) => (
              <div key={name as string} className="grid grid-cols-[50px_repeat(4,minmax(0,1fr))] gap-x-5 items-center text-xs">
                <span className="text-muted">{name as string}</span>
                {picks(m).map((p, j) => (
                  <div key={j} className="flex items-center gap-1.5">
                    <span className="font-mono w-3">{"ABCD"[j]}</span>
                    <div className="flex-1 h-3 rounded-sm bg-track relative"><div className={`absolute inset-y-0 left-0 rounded-sm ${c}`} style={{ width: `${Math.min(p / 0.5, 1) * 100}%` }} /><div className="absolute inset-y-0 w-px bg-fg opacity-40" style={{ left: "50%" }} /></div>
                    <span className="font-mono w-8 text-right">{Math.round(p * 100)}%</span>
                  </div>
                ))}
              </div>
            ))}
            <span className="text-[11px] text-muted">thin line = 25%, where a model that ignores the letter would be</span>
          </div>
          {tr && (
            <div className="flex flex-col gap-1">
              <span className="text-[13px] font-medium flex items-center gap-1">Training loss <Tip k="ftLoss" /></span>
              <LossCurve loss={tr.loss_per_step} perEpoch={perEpoch} />
              <span className="text-[11px] text-muted">
                {`${tr.examples_per_epoch} examples per pass · ${tr.hyper.epochs} passes · ${tr.hours.toFixed(1)} h on a laptop CPU · kept pass ${tr.selected_epoch + 1} (lowest loss on val) · pale line: each step, blue: average of 10 steps · dashed: 0.69 = coin flip, 1.39 = guess among 4`}
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
