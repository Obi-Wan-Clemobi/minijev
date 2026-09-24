"use client";
// The same Qwen model, the same text and questions, the same answer format. Two ways to get it:
// minijev reads the answers out (writes nothing); or the model is told to write that same JSON itself.
import { useState } from "react";
import { api, type SameFormatResponse, type Written } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Icon } from "./Icon";
import { Tip } from "./Tip";
import { useReady } from "./RequestEditor";

const f2 = (x: number | undefined) => (x === undefined ? "—" : x.toFixed(2));

function valueOf(type: string, a: Record<string, unknown> | Written | undefined): string {
  if (!a) return "—";
  if ("ok" in a && !a.ok) return "unusable";
  const x = a as Record<string, unknown>;
  if (type === "noul") return `P(yes) ${f2(x.noul as number)}`;
  if (type === "choice") return `${x.choice}`;
  return `score ${f2(x.score as number)}`;
}

export function SameFormat() {
  const { req, settings, model } = useStore();
  const ready = useReady();
  const [res, setRes] = useState<SameFormatResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setBusy(true); setErr(null);
    try { setRes(await api.sameFormat(req, settings)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const max = res ? Math.max(res.readout.seconds, res.written.seconds) : 1;
  const qtype = (id: string) => req.questions[id]?.type ?? "noul";

  return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-4">
      <div className="flex items-start gap-3 flex-wrap">
        <div className="flex flex-col gap-1 max-w-[760px]">
          <h2 className="text-[15px] font-semibold flex items-center gap-1">1 · Same model, same questions, same answer format: which is faster? <Tip k="sameFormat" /></h2>
          <p className="text-sm text-muted leading-relaxed">
            Both sides use <span className="text-fg font-mono">{model.split("/")[1] ?? "the same model"}</span> on your text and questions, and both return the same JSON.
            <span className="text-accent"> minijev</span> reads every probability out of one pass and writes nothing.
            The <span className="text-gen">model writing it</span> is told to type that exact JSON itself, one token at a time:
            once <span className="text-gen">freely</span>, and once <span className="text-warn">with the format enforced</span>, as AI APIs with structured output do.
          </p>
        </div>
        <div className="flex-1" />
        <Tip k="sameFormatRun">
          <button onClick={run} disabled={busy || !ready} className="h-10 px-4 rounded-lg bg-inv-bg text-inv-fg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
            <Icon name="play" size={14} fill />{busy ? "Running both… (the writing side takes a while)" : "Run both"}
          </button>
        </Tip>
      </div>
      {err && <p role="alert" className="text-sm text-warn">{err}</p>}

      {res && (
        <>
          <div className="flex flex-col gap-2">
            {[
              { name: "minijev reads it out", sec: res.readout.seconds, color: "bg-accent", note: `0 tokens written · ${Object.keys(res.compare).length}/${Object.keys(res.compare).length} usable`, help: undefined },
              { name: "the model writes it freely", sec: res.written.seconds, color: "bg-gen", note: `${res.written.output_tokens} tokens written · ${res.written.usable}/${Object.keys(res.compare).length} usable`, help: undefined },
              { name: "…with the format enforced", sec: res.structured.seconds, color: "bg-warn", note: `${res.structured.output_tokens} tokens chosen + ${res.structured.forced_tokens} typed for it · ${res.structured.usable}/${Object.keys(res.compare).length} usable`, help: "structured" as const },
            ].map((r) => (
              <div key={r.name} className="grid grid-cols-[210px_minmax(0,1fr)_230px] items-center gap-3">
                <span className="text-sm flex items-center gap-1">{r.name}{r.help && <Tip k={r.help} />}</span>
                <div className="h-7 rounded bg-track relative"><div className={`anim absolute inset-y-0 left-0 rounded ${r.color}`} style={{ width: `${(r.sec / max) * 100}%` }} /></div>
                <span className="font-mono text-sm text-right">{r.sec.toFixed(2)} s <span className="text-[11px] text-muted block">{r.note}</span></span>
              </div>
            ))}
            <p className="text-sm">
              Reading it out was <span className="font-semibold">{(res.written.seconds / res.readout.seconds).toFixed(1)}×</span> faster than free writing,
              and <span className="font-semibold">{(res.structured.seconds / res.readout.seconds).toFixed(1)}×</span> faster than writing with the format enforced.
              {" "}Every written token needs its own pass of the model; the readout needs one pass in total.
            </p>
          </div>

          <div className="overflow-x-auto">
            <span className="text-xs text-muted flex items-center gap-1 pb-2">Answer by answer <Tip k="sameFormatDiffer" /></span>
            <table className="w-full text-[13px]">
              <thead><tr className="text-left text-muted text-xs"><th className="font-normal py-1.5 pr-4">question</th><th className="font-normal py-1.5 pr-4 text-accent">minijev reads it out</th><th className="font-normal py-1.5 pr-4 text-gen">writes freely</th><th className="font-normal py-1.5 pr-4">same?</th><th className="font-normal py-1.5 pr-4 text-warn">format enforced</th><th className="font-normal py-1.5 pr-4">same?</th><th className="font-normal py-1.5">minijev&apos;s probability for the enforced answer</th></tr></thead>
              <tbody className="font-mono">
                {Object.keys(res.compare).map((id) => {
                  const a = res.readout.response[id], t = qtype(id);
                  const cell = (w: Written, c: { agree: boolean; note: string }, tone: string) => (
                    <>
                      <td className={`py-2 pr-4 max-w-[260px] ${!w.ok ? "text-warn" : tone}`}>{valueOf(t, w)}{!w.ok && <span className="flex items-start gap-1 text-[11px] font-sans leading-snug">{w.problem}<Tip k="unusable" /></span>}</td>
                      <td className="py-2 pr-4">{!w.ok ? <span className="text-muted">can&apos;t compare</span> : c.agree ? <span className="text-ok">yes</span> : <span className="text-gen">no</span>}<span className="block text-[11px] text-muted font-sans">{w.ok ? c.note : "unusable answer"}</span></td>
                    </>
                  );
                  const s2 = res.structured.answers[id];
                  let cross = "—";
                  if (s2.ok && t === "choice" && s2.choice && a.probabilities) cross = `${f2(a.probabilities[s2.choice])} for "${s2.choice}"`;
                  if (s2.ok && t === "noul" && s2.noul !== undefined) cross = s2.noul > 0.5 ? `${f2(a.noul)} for yes` : `${f2(1 - a.noul)} for no`;
                  if (s2.ok && t === "score" && s2.score !== undefined && a.probabilities) cross = `${f2(a.probabilities[String(Math.round(s2.score))])} for level ${Math.round(s2.score)}`;
                  return (
                    <tr key={id} className="border-t border-line align-top">
                      <td className="py-2 pr-4">{id}</td>
                      <td className="py-2 pr-4">{valueOf(t, a)}</td>
                      {cell(res.written.answers[id], res.compare[id], "")}
                      {cell(s2, res.compare_structured[id], "")}
                      <td className="py-2 text-muted">{cross}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {Object.values(res.written.answers).some((w) => !w.ok) && (
            <p className="text-xs text-muted leading-relaxed">
              &quot;Unusable&quot; means the model did not follow the format for that question (a missing key, a wrong option name, or numbers that are not probabilities).
              Small models often copy the template instead of filling it in. Enforcing the format fixes that: every answer becomes usable.
              It does not make the numbers good: a small model is poor at writing probabilities, even when the probabilities inside it (the ones minijev reads) are sensible.
            </p>
          )}
          {Object.keys(res.compare).some((id) => (res.written.answers[id].ok && !res.compare[id].agree) || (res.structured.answers[id].ok && !res.compare_structured[id].agree)) && (
            <p className="text-xs text-muted leading-relaxed">
              Where the answers differ, the only thing that changed is how the question was asked, so it changed the model&apos;s answer.
              Here there is no answer key, so neither side is known to be right. Section 2 below uses questions with known answers to show which way is right more often.
            </p>
          )}

          <details className="text-[13px]" open={Object.values(res.written.answers).some((w) => !w.ok)}>
            <summary className="cursor-pointer text-muted">What each side got and returned: compare the written reply with the shape it was asked for</summary>
            <div className="grid lg:grid-cols-2 gap-4 mt-3">
              <div className="flex flex-col gap-1">
                <span className="text-xs text-accent">minijev: JSON built from the read-out probabilities</span>
                <pre className="m-0 p-3 rounded-lg border border-line bg-surface font-mono text-xs whitespace-pre-wrap overflow-auto max-h-80">{JSON.stringify(res.readout.response, (k, v) => (typeof v === "number" ? Math.round(v * 100) / 100 : v), 2)}</pre>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-gen">written freely, as it came out ({res.written.valid_json ? "valid JSON" : "not valid JSON"}, {res.written.usable}/{Object.keys(res.compare).length} answers usable)</span>
                <pre className="m-0 p-3 rounded-lg border border-line bg-surface font-mono text-xs whitespace-pre-wrap overflow-auto max-h-80">{res.written.text}</pre>
              </div>
              <div className="lg:col-span-2 flex flex-col gap-1">
                <span className="text-xs text-warn">with the format enforced: the program typed the fixed parts, the model chose only the values</span>
                <pre className="m-0 p-3 rounded-lg border border-line bg-surface font-mono text-xs whitespace-pre-wrap overflow-auto max-h-60">{res.structured.text}</pre>
              </div>
              <div className="lg:col-span-2 flex flex-col gap-1">
                <span className="text-xs text-muted">The prompt the model got for writing ({res.written.prompt_tokens} tokens)</span>
                <pre className="m-0 p-3 rounded-lg border border-line bg-surface font-mono text-xs whitespace-pre-wrap overflow-auto max-h-60">{res.prompt}</pre>
              </div>
            </div>
          </details>
        </>
      )}
      {!res && !err && <p className="text-sm text-muted">Uses the request from the Playground. Edit it above, then press Run both.</p>}
    </section>
  );
}
