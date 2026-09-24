"use client";
import { useMemo, useState } from "react";
import { API } from "@/lib/api";
import { rescore, type ChoiceAnswer, type NoulAnswer, type ScoreAnswer } from "@/lib/scoring";
import { useStore } from "@/lib/store";
import { ChoiceBars, NoulMeter, Ring, ScoreColumns } from "./Viz";
import { Icon } from "./Icon";

const TABS = ["Answers", "JSON", "cURL", "Raw logits"] as const;
const r2 = (x: unknown): unknown =>
  typeof x === "number" ? Math.round(x * 100) / 100 : x && typeof x === "object"
    ? Array.isArray(x) ? x.map(r2) : Object.fromEntries(Object.entries(x).map(([k, v]) => [k, k === "legend" ? v : r2(v)])) : x;

export function Response() {
  const { response, settings, running, error, req } = useStore();
  const [tab, setTab] = useState<(typeof TABS)[number]>("Answers");
  const answers = useMemo(() => response && rescore(response.questions, response.debug.raw, settings), [response, settings]);
  const raw = useMemo(() => response && rescore(response.questions, response.debug.raw,
    { ...settings, temp_noul: 1, temp_choice: 1, temp_score: 1, bias_noul: 0 }), [response, settings]);
  const stale = response && JSON.stringify(Object.keys(response.questions)) !== JSON.stringify(Object.keys(req.questions));
  const lowMass = response ? Object.entries(response.debug.raw).filter(([, r]) => Math.min(...r.mass) < settings.min_label_mass) : [];
  const minMass = response ? Math.min(...Object.values(response.debug.raw).flatMap((r) => r.mass)) : 1;

  return (
    <div className="flex flex-col gap-4 min-w-0">
      <div className="flex items-center gap-3">
        <div role="tablist" aria-label="Response views" className="flex gap-0.5 border-b border-line flex-wrap">
          {TABS.map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
              className={`h-9 px-3 text-[13px] -mb-px border-b-2 whitespace-nowrap transition-colors ${tab === t ? "border-fg text-fg font-medium" : "border-transparent text-muted hover:text-fg"}`}>{t}</button>
          ))}
        </div>
        <div className="flex-1" />
        {response && <span className="text-xs font-mono text-ok">● 200</span>}
        {response && <span className="text-xs font-mono text-muted">{response.usage.latency_ms} ms</span>}
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-warn p-4 text-sm flex gap-2 items-start"><Icon name="alert" className="text-warn mt-0.5 shrink-0" /><span>{error}</span></div>
      )}
      {stale && <div className="text-xs text-warn">The questions changed since the last run. Press Run to update the answers.</div>}
      {!response && !error && (
        <div className="rounded-xl border border-dashed border-line p-10 text-center text-sm text-muted">
          {running ? "Running one forward pass on the CPU…" : "Press Run (⌘↵) to read the answers out of one forward pass."}
        </div>
      )}

      {response && answers && raw && tab === "Answers" && (
        <div className={`flex flex-col gap-4 transition-opacity ${running ? "opacity-50" : ""}`}>
          <div className="flex items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5"><span className="w-[18px] h-2 rounded-sm bg-accent" />calibrated</span>
            <span className="flex items-center gap-1.5"><span className="w-[18px] h-2 rounded-sm border border-dashed border-ghost" />raw (T = 1)</span>
          </div>
          {Object.entries(response.questions).map(([id, q]) => {
            const a = answers[id], b = raw[id];
            const mode = q.type === "choice" ? q.choice_mode : q.type === "score" ? q.score_mode : null;
            return (
              <article key={id} className="rounded-xl border border-line bg-card p-5 flex gap-6">
                <div className="flex-1 min-w-0 flex flex-col gap-3">
                  <div className="flex items-baseline gap-2.5 flex-wrap">
                    <span className="font-mono text-[13px] font-medium">{id}</span>
                    <span className="text-xs text-muted">{q.type === "noul" ? "Noul" : q.type === "choice" ? "Choice" : "Score"}{mode ? ` · ${mode}` : ""} · {q.instructions}</span>
                    <div className="flex-1" />
                    {a.type === "noul" && <><span className="text-3xl font-semibold tracking-tight tabular-nums">{a.noul.toFixed(2)}</span><span className="text-xs text-muted">P(yes)</span></>}
                    {a.type === "score" && <><span className="text-3xl font-semibold tracking-tight tabular-nums">{a.score.toFixed(2)}</span><span className="text-xs text-muted">of {Object.keys(a.probabilities).length - 1}</span></>}
                  </div>
                  {a.type === "noul" && <NoulMeter a={a} raw={b as NoulAnswer} />}
                  {a.type === "choice" && <ChoiceBars a={a} raw={b as ChoiceAnswer} descriptions={(q.criteria ?? {}) as Record<string, string | null>} />}
                  {a.type === "score" && <ScoreColumns a={a} raw={b as ScoreAnswer} />}
                </div>
                {a.type === "choice" && <Ring value={a.confidence} label="confidence" />}
                {a.type === "score" && <Ring value={a.confidence} label="ordinal confidence" />}
              </article>
            );
          })}
        </div>
      )}

      {response && answers && tab === "JSON" && (
        <pre className="m-0 p-5 rounded-xl border border-line bg-card font-mono text-[13px] leading-relaxed whitespace-pre-wrap overflow-auto">
          {JSON.stringify(r2({ model: response.model, answers, usage: response.usage }), null, 2)}
        </pre>
      )}
      {tab === "cURL" && (
        <pre className="m-0 p-5 rounded-xl border border-line bg-card font-mono text-[13px] leading-relaxed whitespace-pre-wrap overflow-auto">
          {`curl -s ${API}/v1/ask \\\n  -H 'content-type: application/json' \\\n  -d '${JSON.stringify({ ...req, settings }, null, 2).replace(/'/g, "'\\''")}'`}
        </pre>
      )}
      {response && tab === "Raw logits" && (
        <div className="rounded-xl border border-line bg-card px-5 py-2 font-mono text-[13px] overflow-x-auto">
          {Object.entries(response.debug.raw).map(([id, r]) => (
            <div key={id} className="grid grid-cols-[120px_minmax(0,1fr)_140px] gap-3 py-2.5 border-b border-line last:border-0">
              <span>{id}</span>
              <span>{r.logits.map((v) => v.toFixed(3)).join(", ")}</span>
              <span className="text-muted">mass {Math.min(...r.mass).toFixed(4)}</span>
            </div>
          ))}
          <p className="py-2.5 text-xs text-muted font-sans">Noul and listwise: log-probabilities of each label. Pointwise: the yes/no log-odds of each item.</p>
        </div>
      )}

      {response && (
        <div className="flex items-center gap-6 flex-wrap px-4 py-3.5 rounded-[10px] border border-line bg-card text-[13px]">
          <span><span className="text-muted">input</span> <span className="font-mono">{response.usage.input_tokens}</span></span>
          <span><span className="text-muted">output</span> <span className="font-mono font-semibold">0</span></span>
          <span><span className="text-muted">latency</span> <span className="font-mono">{response.usage.latency_ms} ms</span></span>
          <span className="text-muted font-mono text-xs">{response.model.replace("minijev-poc (", "").replace(")", "")}</span>
          <div className="flex-1" />
          {lowMass.length ? (
            <span className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border border-warn text-warn">
              <Icon name="alert" size={12} />low label mass: {lowMass.map(([id]) => id).join(", ")}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border border-line">
              <Icon name="check" size={12} className="text-ok" />label mass ≥ {minMass.toFixed(3)} on every branch
            </span>
          )}
        </div>
      )}
    </div>
  );
}

