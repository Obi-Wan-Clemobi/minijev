"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { IntervalRow } from "@/components/Charts";
import { api } from "@/lib/api";

type Bin = { conf_bin: string; n: number; mean_conf: number | null; accuracy: number | null };

function Reliability({ raw, temp, view }: { raw: Bin[]; temp: Bin[]; view: string }) {
  const px = (c: number) => 50 + ((c - 0.5) / 0.5) * 370;
  const py = (a: number) => 360 - ((Math.max(a, 0.3) - 0.3) / 0.7) * 350;
  const dots = (bins: Bin[], color: string, on: boolean) => bins.filter((b) => b.n && b.mean_conf !== null).map((b, i) => (
    <circle key={color + i} className="anim" cx={px(b.mean_conf!)} cy={py(b.accuracy!)} r={3 + Math.sqrt(b.n) * 1.1} fill={color} opacity={on ? 0.85 : 0.08}>
      <title>{`${b.conf_bin}: ${b.n} items, confidence ${b.mean_conf!.toFixed(2)}, accuracy ${b.accuracy!.toFixed(2)}`}</title>
    </circle>
  ));
  return (
    <svg viewBox="0 0 430 400" className="w-full max-w-[430px]" role="img" aria-label="Reliability diagram: accuracy against stated confidence">
      <line x1="50" y1="360" x2="420" y2="360" stroke="var(--line)" />
      <line x1="50" y1="360" x2="50" y2="10" stroke="var(--line)" />
      <line x1="50" y1={py(0.5)} x2="420" y2="10" stroke="var(--muted)" strokeDasharray="4 4" />
      {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((c) => <text key={c} x={px(c)} y="378" textAnchor="middle" fontSize="11" fill="var(--muted)" fontFamily="var(--font-geist-mono)">{c.toFixed(1)}</text>)}
      {[0.4, 0.6, 0.8, 1].map((a) => <text key={a} x="42" y={py(a) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" fontFamily="var(--font-geist-mono)">{a.toFixed(1)}</text>)}
      <text x="235" y="396" textAnchor="middle" fontSize="11" fill="var(--muted)">stated confidence</text>
      {dots(raw, "var(--gen)", view !== "temp")}
      {dots(temp, "var(--accent)", view !== "raw")}
    </svg>
  );
}

function Bars({ rows }: { rows: { name: string; v: number; label: string; tone: string }[] }) {
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map((r) => (
        <div key={r.name} className="grid grid-cols-[130px_minmax(0,1fr)_56px] items-center gap-2.5 text-[13px]">
          <span>{r.name}</span>
          <div className="h-4 rounded-sm bg-track relative"><div className={`absolute inset-y-0 left-0 rounded-sm ${r.tone}`} style={{ width: `${Math.max(r.v * 100, 0.6)}%` }} /></div>
          <span className="font-mono text-right">{r.label}</span>
        </div>
      ))}
    </div>
  );
}

export default function Findings() {
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState("both");
  const [size, setSize] = useState("0.5B");
  useEffect(() => { api.results().then(setRes).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="p-8"><p role="alert" className="text-warn text-sm">{err}</p></div>;
  if (!res) return <div className="p-8 text-sm text-muted">Loading the results…</div>;

  const cal = (s: string) => res[s].calibration.metrics;
  const m = cal(size);
  const ece = (s: string, k: string, name: string, tone: string) => {
    const x = cal(s)[k];
    return <IntervalRow key={s + k} name={name} value={x.ece} lo={x.ece_ci95[0]} hi={x.ece_ci95[1]} max={0.4} tone={tone} />;
  };
  const perm = (s: string) => res[s].permutation;
  const jev = (s: string) => res[s].jevdocs.raw;
  const n = res["0.5B"].jevdocs.n;
  const sd = res.single_decision.generation_costs;

  return (
    <div className="px-4 md:px-8 py-8 flex flex-col gap-6">
      <div className="flex flex-col gap-1.5">
        <h1 className="m-0 text-[32px] font-semibold tracking-tight">What we measured</h1>
        <p className="m-0 text-[15px] text-muted max-w-[820px] leading-normal">Every chart reads poc/results/*.json. Brackets are 95% bootstrap intervals. Qwen2.5 0.5B and 1.5B on a laptop CPU.</p>
      </div>

      <div className="grid xl:grid-cols-[480px_minmax(0,1fr)] gap-6">
        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <h2 className="text-[15px] font-semibold">Reliability · BoolQ, n = {res[size].calibration.n}</h2>
            <div className="flex gap-2">
              <div className="flex border border-line rounded-lg p-[3px] gap-0.5">
                {["0.5B", "1.5B"].map((s) => <button key={s} onClick={() => setSize(s)} aria-pressed={size === s} className={`h-[30px] px-2.5 rounded-[5px] text-xs font-mono ${size === s ? "bg-track text-fg" : "text-muted"}`}>{s}</button>)}
              </div>
              <div className="flex border border-line rounded-lg p-[3px] gap-0.5">
                {[["raw", "Raw"], ["temp", "Temperature"], ["both", "Both"]].map(([k, l]) => <button key={k} onClick={() => setView(k)} aria-pressed={view === k} className={`h-[30px] px-2.5 rounded-[5px] text-xs ${view === k ? "bg-track text-fg" : "text-muted"}`}>{l}</button>)}
              </div>
            </div>
          </div>
          <Reliability raw={m.raw.reliability} temp={m["temperature (2-fold)"].reliability} view={view} />
          <div className="flex gap-4 flex-wrap text-xs text-muted">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-gen" />raw · ECE {m.raw.ece.toFixed(3)}</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent" />T = {res[size].calibration.temperature_all.toFixed(2)} · ECE {m["temperature (2-fold)"].ece.toFixed(3)}</span>
            <span>dot size = items in the bin · dashed = perfect calibration</span>
          </div>
        </section>

        <div className="flex flex-col gap-6">
          <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
            <div className="flex items-baseline justify-between"><h2 className="text-[15px] font-semibold">Calibration error (ECE) · lower is better</h2><span className="text-xs text-muted font-mono">axis 0–0.40</span></div>
            {ece("0.5B", "raw", "0.5B · raw", "bg-gen")}
            {ece("0.5B", "temperature (2-fold)", "0.5B · temperature", "bg-accent")}
            {ece("0.5B", "contextual", '0.5B · contextual "N/A"', "bg-ghost")}
            {ece("1.5B", "raw", "1.5B · raw", "bg-gen")}
            {ece("1.5B", "temperature (2-fold)", "1.5B · temperature", "bg-accent")}
            {ece("1.5B", "Platt a*z+b (2-fold)", "1.5B · Platt", "bg-accent")}
            <p className="text-xs text-muted leading-normal">Temperature scaling helps at both sizes. The calibrated 0.5B and 1.5B intervals overlap. Contextual calibration with an &quot;N/A&quot; state makes it worse.</p>
          </section>

          <div className="grid md:grid-cols-2 gap-6">
            <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
              <h2 className="text-[15px] font-semibold">Accuracy vs always &quot;yes&quot;</h2>
              {["0.5B", "1.5B"].map((s) => {
                const x = cal(s).raw;
                return <IntervalRow key={s} name={`Qwen2.5-${s}`} value={x.accuracy} lo={x.accuracy_ci95[0]} hi={x.accuracy_ci95[1]} min={0.5} max={0.9} mark={res[s].calibration.base_rate_yes} />;
              })}
              <p className="text-xs text-muted leading-normal">Orange line: base rate {res["0.5B"].calibration.base_rate_yes.toFixed(3)}. The 0.5B interval crosses it.</p>
            </section>
            <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
              <h2 className="text-[15px] font-semibold">Option order changes the winner</h2>
              <Bars rows={["0.5B", "1.5B"].flatMap((s) => [
                { name: `${s} listwise`, v: perm(s).listwise.mean_argmax_change_rate, label: `${Math.round(perm(s).listwise.mean_argmax_change_rate * 100)}%`, tone: "bg-gen" },
                { name: `${s} pointwise`, v: perm(s).pointwise.mean_argmax_change_rate, label: `${Math.round(perm(s).pointwise.mean_argmax_change_rate * 100)}%`, tone: "bg-accent" },
              ])} />
              <p className="text-xs text-muted leading-normal">8 Choices, every rotation. Pointwise is 0% by construction.</p>
            </section>
          </div>
        </div>
      </div>

      <div className="grid xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-6">
        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
          <div className="flex items-baseline justify-between flex-wrap gap-2"><h2 className="text-[15px] font-semibold">Agreement with Jev&apos;s published answers</h2><span className="text-xs text-muted">not accuracy · small samples · not held out</span></div>
          {[["Noul · same side of 0.5", "noul_same_side_of_0.5", n.noul], ["Score · same rounded level", "score_pointwise_same_rounded_level", n.score], ["Choice · same pick (listwise)", "choice_listwise_same_argmax", n.choice]].map(([name, key, count]) => (
            <div key={key as string} className="grid grid-cols-[minmax(150px,220px)_repeat(2,minmax(0,1fr))] items-center gap-4 text-[13px]">
              <span>{name}</span>
              {["0.5B", "1.5B"].map((s) => {
                const v = jev(s)[key as string];
                return (
                  <div key={s} className="flex items-center gap-2.5">
                    <div className="flex-1 h-3 rounded-sm bg-track relative"><div className={`absolute inset-y-0 left-0 rounded-sm ${s === "1.5B" ? "bg-accent" : "bg-accent2"}`} style={{ width: `${v * 100}%` }} /></div>
                    <span className="font-mono w-[72px]">{s} {Math.round(v * (count as number))}/{count}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </section>
        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3.5">
          <h2 className="text-[15px] font-semibold">Cost per token · 0.5B, {sd.context_tokens}-token context</h2>
          <Bars rows={[
            { name: "prefill", v: sd.ms_per_prefill_token / sd.ms_per_decode_token, label: `${sd.ms_per_prefill_token.toFixed(1)} ms`, tone: "bg-accent" },
            { name: "decode", v: 1, label: `${sd.ms_per_decode_token.toFixed(0)} ms`, tone: "bg-gen" },
          ]} />
          <p className="text-xs text-muted leading-normal">A generated token costs about {Math.round(sd.decode_over_prefill)}× a prefilled one on this CPU. A readout generates none.</p>
        </section>
      </div>
    </div>
  );
}
