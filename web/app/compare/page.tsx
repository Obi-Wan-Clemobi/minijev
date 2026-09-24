"use client";
import { useEffect, useState } from "react";
import { RaceBars, Waterfall, type RaceRow } from "@/components/Charts";
import { Quality, type QualityData } from "@/components/Quality";
import { SameFormat } from "@/components/SameFormat";
import { Icon } from "@/components/Icon";
import { Tip } from "@/components/Tip";
import type { HelpKey } from "@/lib/help";
import { RequestEditor, useReady } from "@/components/RequestEditor";
import { api, type CompareResponse } from "@/lib/api";
import { useStore } from "@/lib/store";

const METHODS: { id: string; name: string; sub: string; tone: RaceRow["tone"]; slow?: boolean; help?: HelpKey }[] = [
  { id: "readout", name: "Reads out (minijev)", sub: "all questions in one pass; writes nothing", tone: "accent" },
  { id: "logprobs_cached", name: "Reads out, one question per request", sub: "like an AI API that returns probabilities", tone: "accent2", help: "cmpLogprobs" },
  { id: "generate_cached", name: "Writes each answer", sub: "one request per question; text remembered", tone: "gen", help: "cmpGenCached" },
  { id: "generate_json", name: "Writes all answers as one JSON", sub: "one request with every question", tone: "gen", help: "cmpJson" },
  { id: "generate_uncached", name: "Writes each answer, resends the text", sub: "one request per question; whole text every time", tone: "gen", slow: true, help: "cmpUncached" },
];

type Rec = { questions: number; [k: string]: unknown };

export default function Compare() {
  const { req, model } = useStore();
  const ready = useReady();
  const [picked, setPicked] = useState<string[]>(METHODS.filter((m) => !m.slow).map((m) => m.id));
  const [res, setRes] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<Record<string, Rec[]> | null>(null);
  const [quality, setQuality] = useState<QualityData | null>(null);
  const [recModel, setRecModel] = useState("0.5B");

  useEffect(() => {
    api.results().then((r) => {
      setRecorded({ "0.5B": r["0.5B"].fanout, "1.5B": r["1.5B"].fanout });
      setQuality({ "0.5B": r["0.5B"].quality, "1.5B": r["1.5B"].quality });
    }).catch(() => {});
  }, []);

  const run = async () => {
    setBusy(true); setErr(null);
    try { setRes(await api.compare(req, picked)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  const live: RaceRow[] = res ? METHODS.filter((m) => res.methods[m.id]).map((m) => {
    const r = res.methods[m.id];
    return { name: m.name, sub: m.sub, sec: r.seconds, tone: m.tone,
      note: `${r.output_tokens} out · ${m.id === "readout" ? "reference" : `agree ${Math.round(r.agrees_with_readout * 100)}%`}` };
  }).sort((a, b) => b.sec - a.sec) : [];

  const rec = recorded?.[recModel]?.find((r) => r.questions === 13) as Record<string, { seconds: number; seconds_all: number[]; output_tokens: number; agrees_with_readout: number }> | undefined;
  const recRows: RaceRow[] = rec ? ([
    ["generate_per_question", "Writes each answer, resends the text", "the text is read 13 times", "gen"],
    ["generate_one_json", "Writes all answers as one JSON", "one request, 124 tokens written", "gen"],
    ["generate_batched_cached", "Writes all answers in parallel", "13 requests side by side", "gen"],
    ["generate_per_question_cached", "Writes each answer", "text remembered between requests", "gen"],
    ["generate_1_token_logprobs_cached", "Reads out, one question per request", "like an AI API with probabilities", "accent2"],
    ["readout_packed", "Reads out (minijev)", "all questions in one pass", "accent"],
  ] as const).map(([k, name, sub, tone]) => ({
    name, sub, tone, sec: rec[k].seconds, lo: Math.min(...rec[k].seconds_all), hi: Math.max(...rec[k].seconds_all),
    note: `${rec[k].output_tokens} out · ${k === "readout_packed" ? "reference" : `agree ${Math.round(rec[k].agrees_with_readout * 100)}%`}`,
  })) : [];
  const fall = rec ? ["generate_per_question", "generate_per_question_cached", "generate_1_token_logprobs_cached", "readout_packed"].map((k) => rec[k].seconds) : [];
  const saving = fall.length ? fall[0] - fall[3] : 1;
  const shares = fall.length ? [fall[0] - fall[1], fall[1] - fall[2], fall[2] - fall[3]].map((x) => Math.round((x / saving) * 100)) : [];

  return (
    <div className="px-4 md:px-8 py-8 flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <h1 className="m-0 text-[32px] font-semibold tracking-tight">Read the answer out, or let the model write it?</h1>

        {/* Color Legend */}
        <div className="rounded-xl border-2 border-accent bg-surface p-4 flex items-center gap-6 flex-wrap">
          <span className="text-sm font-semibold text-muted uppercase tracking-wide">Chart Colors:</span>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-accent shrink-0" />
            <span className="text-sm font-medium">Blue = minijev (readout)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-gen shrink-0" />
            <span className="text-sm font-medium">Orange = LLM generation</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-accent2 shrink-0" />
            <span className="text-sm font-medium">Cyan = Readout per request</span>
          </div>
        </div>

        <div className="rounded-xl border border-line bg-surface p-4 flex flex-col gap-3 max-w-[980px]">
          <p className="m-0 text-[15px] leading-relaxed">
            <span className="font-semibold">Everything on this page is the same model</span> ({model.split("/")[1] ?? "Qwen"}) on this laptop. There is no second or bigger AI here.
            What changes is only <span className="font-semibold">how we get the answer out of it</span>:
          </p>
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div className="flex gap-2"><span className="mt-1 w-3 h-3 rounded-sm bg-accent shrink-0" /><span><span className="font-medium">Reads the answer out</span> (minijev): the model reads the text and questions once, and we look at how likely it finds each allowed answer. It writes nothing.</span></div>
            <div className="flex gap-2"><span className="mt-1 w-3 h-3 rounded-sm bg-accent2 shrink-0" /><span><span className="font-medium">Reads out, one question per request</span>: the same trick done through an AI API that returns probabilities, one question at a time.</span></div>
            <div className="flex gap-2"><span className="mt-1 w-3 h-3 rounded-sm bg-gen shrink-0" /><span><span className="font-medium">Writes the answer</span> (the usual chatbot way): the model types its answer as text, word piece by word piece, and code reads the text back.</span></div>
          </div>
        </div>
      </div>

      <details className="group rounded-xl border border-line bg-card">
        <summary className="list-none cursor-pointer px-6 py-4 flex items-center gap-2 text-[15px] font-semibold">
          <Icon name="down" className="transition-transform group-open:rotate-180" />
          Edit the request · your own state and questions
          <span className="text-xs font-normal text-muted ml-2 truncate max-w-[480px]">{typeof req.state === "string" ? req.state : "JSON state"}</span>
        </summary>
        <div className="px-6 pb-6 max-w-[720px]"><RequestEditor presets /></div>
      </details>

      <SameFormat />

      <Quality data={quality} />


      <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-4">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="text-[15px] font-semibold">3 · Other ways to ask several questions · your request, {Object.keys(req.questions).length} questions</h2>
          <div className="flex-1" />
          <fieldset className="flex gap-3 flex-wrap text-[13px]">
            <legend className="sr-only">Methods</legend>
            <Tip k="cmpMethods" />
            {METHODS.filter((m) => m.id !== "readout").map((m) => (
              <span key={m.id} className="flex items-center gap-0.5">
                <label className="flex items-center gap-1.5">
                  <input type="checkbox" checked={picked.includes(m.id)} className="accent-[var(--accent)]"
                    onChange={(e) => setPicked((p) => (e.target.checked ? [...p, m.id] : p.filter((x) => x !== m.id)))} />
                  {m.name}{m.slow ? " (slow)" : ""}
                </label>
                {m.help && <Tip k={m.help} />}
              </span>
            ))}
          </fieldset>
          <Tip k="cmpRun"><button onClick={run} disabled={busy || !ready} className="h-10 px-4 rounded-lg bg-inv-bg text-inv-fg text-sm font-medium flex items-center gap-2 disabled:opacity-40">
            <Icon name="play" size={14} fill />{busy ? "Running… (this takes seconds on a CPU)" : "Run comparison"}
          </button></Tip>
        </div>
        {err && <p role="alert" className="text-sm text-warn">{err}</p>}
        {res ? <RaceBars rows={live} /> : <p className="text-sm text-muted">Not run yet. The readout always runs as the reference.</p>}
        {res && <p className="text-xs text-muted">One run each, in this order: readout first. Single runs on a laptop CPU vary by about ±10–20%; the recorded results below are medians of 3.</p>}
        {res && (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px] font-mono">
              <caption className="text-left text-xs text-muted pb-2">Each method&apos;s answer. Orange = differs from the readout; &quot;unparsed&quot; = the written text could not be matched to an option. <Tip k="agreement" /></caption>
              <thead><tr className="text-muted text-left"><th className="font-normal py-1.5 pr-4">question</th>{METHODS.filter((m) => res.methods[m.id]).map((m) => <th key={m.id} className="font-normal py-1.5 pr-4">{m.name}</th>)}</tr></thead>
              <tbody>
                {Object.keys(res.questions).map((q) => (
                  <tr key={q} className="border-t border-line">
                    <td className="py-1.5 pr-4">{q}</td>
                    {METHODS.filter((m) => res.methods[m.id]).map((m) => {
                      const v = res.methods[m.id].answers[q], same = v === res.methods.readout.answers[q];
                      return <td key={m.id} className={`py-1.5 pr-4 ${v === null ? "text-warn" : same ? "" : "text-gen"}`}>{v ?? "unparsed"}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="grid xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)] gap-6">
        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3.5">
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-[15px] font-semibold">Measured earlier: 13 questions about one 500-token article</h2>
            <Tip k="cmpRecorded" />
            <div className="flex-1" />
            <div className="flex border border-line rounded-lg p-[3px] gap-0.5 font-mono">
              {["0.5B", "1.5B"].map((m) => (
                <button key={m} onClick={() => setRecModel(m)} aria-pressed={recModel === m}
                  className={`h-8 px-3 rounded-[5px] text-[13px] ${recModel === m ? "bg-track text-fg" : "text-muted"}`}>Qwen2.5-{m}</button>
              ))}
            </div>
          </div>
          <p className="text-xs text-muted">Median of 3 runs in rotating order; whiskers show the full spread. poc/results/fanout*.json.</p>
          {recRows.length ? <RaceBars rows={recRows} /> : <p className="text-sm text-muted">Start the API to load the recorded results.</p>}
        </section>
        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-4">
          <h2 className="text-[15px] font-semibold flex items-center gap-1">Where the time goes · {recModel} <Tip k="cmpWaterfall" /></h2>
          {fall.length > 0 && (
            <>
              <Waterfall steps={fall} labels={["writes each answer, resends the text", "remember the text", "read out instead of writing", "all questions in one pass"]} />
              <div className="flex h-2.5 rounded-full overflow-hidden">
                <div className="anim bg-gen" style={{ width: `${shares[0]}%` }} />
                <div className="anim bg-accent2" style={{ width: `${shares[1]}%` }} />
                <div className="anim bg-accent" style={{ width: `${shares[2]}%` }} />
              </div>
              <div className="flex justify-between text-xs font-mono"><span>remember the text {shares[0]}%</span><span>read out {shares[1]}%</span><span>one pass {shares[2]}%</span></div>
              <p className="text-xs text-muted leading-normal">Most of the time is saved by reading the text only once. The next saving is reading the answer out instead of writing it. An AI API that remembers the text and returns probabilities can copy both.</p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
