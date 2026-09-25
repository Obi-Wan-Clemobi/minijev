"use client";
// The live half of the Data page: every data.py check re-run now, the split balance, and the held-out (E14) results.
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { API } from "@/lib/api";
import { Icon } from "./Icon";
import { Tip } from "./Tip";

const SPLITS = ["train", "val", "test"] as const;
const COLORS = ["var(--accent)", "var(--gen)", "var(--accent2)", "var(--ok)"];

function Interval({ v, lo, hi }: { v: number; lo: number; hi: number }) {
  return <span className="font-mono">{v.toFixed(3)} <span className="text-muted">[{lo.toFixed(2)}–{hi.toFixed(2)}]</span></span>;
}

export function DataFacts() {
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showClaims, setShowClaims] = useState(false);
  useEffect(() => {
    fetch(`${API}/v1/data`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))).then(setD)
      .catch(() => setErr(`Cannot reach the minijev API at ${API}. Start it to see the live checks.`));
  }, []);
  if (err) return <p className="text-sm text-warn">{err}</p>;
  if (!d) return <p className="text-sm text-muted">Checking the data files…</p>;
  const passed = d.claims.filter((c: any) => c.passed).length;

  return (
    <div className="flex flex-col gap-6">
      <section className={`rounded-xl border p-5 flex flex-col gap-3 ${d.all_pass ? "border-ok" : "border-warn"} bg-card`}>
        <div className="flex items-center gap-2 flex-wrap">
          <Icon name={d.all_pass ? "check" : "alert"} className={d.all_pass ? "text-ok" : "text-warn"} />
          <h2 className="text-[15px] font-semibold">{passed} of {d.claims.length} claims verified just now, from the files</h2>
          <Tip k="dataClaims" />
          <div className="flex-1" />
          <button onClick={() => setShowClaims((v) => !v)} className="h-8 px-3 rounded-md border border-line text-[13px] hover:bg-track">
            {showClaims ? "Hide the checks" : "Show the checks"}
          </button>
        </div>
        <p className="text-xs text-muted">Frozen splits file <span className="font-mono">datasets/splits_v2.json</span> · SHA-256 <span className="font-mono">{d.summary.splits_sha256.slice(0, 16)}…</span> · seed {d.summary.seed} · created {d.summary.created}</p>
        {showClaims && (
          <ul className="flex flex-col gap-1 text-[13px]">
            {d.claims.map((c: any, i: number) => (
              <li key={i} className="flex gap-2"><span className={`font-mono text-xs w-10 shrink-0 ${c.passed ? "text-ok" : "text-warn"}`}>{c.passed ? "PASS" : "FAIL"}</span><span className="text-muted w-16 shrink-0">{c.dataset}</span><span>{c.claim}</span></li>
            ))}
          </ul>
        )}
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        {Object.entries(d.summary.datasets).map(([name, ds]: [string, any]) => {
          const labels = Object.values(ds.labels) as string[];
          const max = Math.max(...SPLITS.map((s) => ds.splits[s].n));
          return (
            <div key={name} className="rounded-xl border border-line bg-card p-5 flex flex-col gap-3">
              <div className="flex items-baseline justify-between gap-2 flex-wrap">
                <h3 className="text-[15px] font-semibold">{name === "boolq" ? "BoolQ · yes/no" : "AG News · topics"}</h3>
                <span className="text-xs text-muted font-mono">{ds.source.hf_id} · {ds.source.split} · {ds.source.rows} rows</span>
              </div>
              {SPLITS.map((s) => {
                const sp = ds.splits[s];
                return (
                  <div key={s} className="grid grid-cols-[48px_minmax(0,1fr)_56px] items-center gap-3 text-[13px]">
                    <span className="font-mono">{s}</span>
                    <div className="h-5 flex rounded overflow-hidden" style={{ width: `${(sp.n / max) * 100}%` }}
                      role="img" aria-label={`${s}: ${Object.entries(sp.labels).map(([k, v]) => `${v} ${k}`).join(", ")}`}>
                      {labels.map((l, i) => (
                        <div key={l} title={`${l}: ${sp.labels[l] ?? 0}`} style={{ width: `${((sp.labels[l] ?? 0) / sp.n) * 100}%`, background: COLORS[i] }} />
                      ))}
                    </div>
                    <span className="font-mono text-right">{sp.n}</span>
                  </div>
                );
              })}
              <div className="flex gap-3 flex-wrap text-xs text-muted">
                {labels.map((l, i) => <span key={l} className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: COLORS[i] }} />{l}</span>)}
              </div>
              <p className="text-xs text-muted leading-relaxed">{ds.sampling}</p>
            </div>
          );
        })}
      </section>

      <section className="rounded-xl border border-line bg-card p-5 flex flex-col gap-4">
        <div className="flex items-center gap-1">
          <h2 className="text-[15px] font-semibold">Held-out results: fitted on train, chosen on val, reported on test</h2>
          <Tip k="dataHeldout" />
        </div>
        {Object.keys(d.heldout).length === 0 && <p className="text-sm text-muted">Not run yet. Run <code className="font-mono">uv run python experiments.py heldout</code> in poc/ (both models).</p>}
        {Object.entries(d.heldout).map(([size, h]: [string, any]) => {
          const cal = h.calibration, nt = h.test.noul, ct = h.test.choice;
          return (
            <div key={size} className="flex flex-col gap-2">
              <h3 className="text-[13px] font-semibold">Qwen2.5-{size}</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px] min-w-[640px]">
                  <thead><tr className="text-left text-xs text-muted"><th className="font-normal py-1.5 pr-4">Test split</th><th className="font-normal py-1.5 pr-4">Fitted on train</th><th className="font-normal py-1.5 pr-4">Accuracy</th><th className="font-normal py-1.5 pr-4">ECE raw</th><th className="font-normal py-1.5">ECE calibrated</th></tr></thead>
                  <tbody>
                    <tr className="border-t border-line align-top">
                      <td className="py-2 pr-4">BoolQ yes/no · n = {nt.raw.n}</td>
                      <td className="py-2 pr-4 font-mono text-xs">{cal.noul.selected === "platt" ? `Platt a = ${cal.noul.platt.a.toFixed(2)}, b = ${cal.noul.platt.b.toFixed(2)}` : `T = ${cal.noul.temperature.toFixed(2)}`}<span className="block text-muted font-sans">{cal.noul.selected} chosen on val</span></td>
                      <td className="py-2 pr-4"><Interval v={nt[cal.noul.selected].accuracy} lo={nt[cal.noul.selected].accuracy_ci95[0]} hi={nt[cal.noul.selected].accuracy_ci95[1]} /></td>
                      <td className="py-2 pr-4"><Interval v={nt.raw.ece} lo={nt.raw.ece_ci95[0]} hi={nt.raw.ece_ci95[1]} /></td>
                      <td className="py-2"><Interval v={nt[cal.noul.selected].ece} lo={nt[cal.noul.selected].ece_ci95[0]} hi={nt[cal.noul.selected].ece_ci95[1]} /></td>
                    </tr>
                    {Object.entries(ct).map(([mode, r]: [string, any]) => (
                      <tr key={mode} className={`border-t border-line align-top ${mode === cal.choice.selected_mode ? "" : "text-muted"}`}>
                        <td className="py-2 pr-4">AG News {mode} · n = {r.calibrated.n ?? 400}{mode === cal.choice.selected_mode && <span className="ml-2 text-[11px] px-1.5 rounded-full border border-accent text-accent">chosen on val</span>}</td>
                        <td className="py-2 pr-4 font-mono text-xs">T = {cal.choice.temperature[mode].toFixed(2)}</td>
                        <td className="py-2 pr-4"><Interval v={r.calibrated.accuracy} lo={r.calibrated.accuracy_ci95[0]} hi={r.calibrated.accuracy_ci95[1]} /></td>
                        <td className="py-2 pr-4"><Interval v={r.raw.ece} lo={r.raw.ece_ci95[0]} hi={r.raw.ece_ci95[1]} /></td>
                        <td className="py-2"><Interval v={r.calibrated.ece} lo={r.calibrated.ece_ci95[0]} hi={r.calibrated.ece_ci95[1]} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted">Splits read: {h.splits_accessed.map((a: any) => `${a.dataset}/${a.split} (${a.n})`).join(" · ")}. Test was only reported: {h.use_of_splits.test}.</p>
            </div>
          );
        })}
      </section>
    </div>
  );
}
