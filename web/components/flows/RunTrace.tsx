"use client";
// The side panel after (or during) a run: every decision in order, with probabilities and what the model read.
import { Tip } from "@/components/Tip";
import { answerLabel, type Decision, type Flow, type RunEnd } from "@/lib/flow";

const STATUS: Record<RunEnd["status"], [string, string]> = {
  completed: ["Reached DONE", "text-ok"],
  no_transition: ["Stopped: no arrow matched", "text-warn"],
  max_steps: ["Stopped: too many steps (a loop?)", "text-warn"],
  invalid: ["Not run: the flow has errors", "text-warn"],
  error: ["Not run", "text-warn"],
};

export function RunTrace({ flow, decisions, end, running, onSelect }: {
  flow: Flow; decisions: Decision[]; end: RunEnd | null; running: string | null; onSelect: (id: string) => void;
}) {
  if (!decisions.length && !end && !running) return (
    <p className="m-0 text-sm text-muted leading-relaxed">
      Type a request above the canvas and press Run. Each step will light up when the model decides it, and its decision appears here.
    </p>
  );
  const total = decisions.reduce((s, d) => s + d.ms, 0);
  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-[15px] font-semibold m-0 flex items-center gap-1">Decisions <Tip k="smAnswer" /></h2>
      {decisions.map((d, i) => {
        const s = flow.steps[d.step];
        return (
          <div key={i} className="rounded-lg border border-line p-3 flex flex-col gap-2">
            <button className="text-left flex items-baseline gap-2" onClick={() => onSelect(d.step)}>
              <span className="text-xs text-muted font-mono">{i + 1}. {d.step}</span>
              <span className="ml-auto text-xs text-muted font-mono">{(d.ms / 1000).toFixed(2)} s</span>
            </button>
            <p className="m-0 text-[13px]">{d.question}</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-1.5 py-0.5 rounded bg-accent text-inv-fg font-medium">{answerLabel(s, d.answer)}</span>
              <span className="text-muted">sure</span>
              <div className="flex-1 h-2 rounded-sm bg-track relative"><div className="absolute inset-y-0 left-0 rounded-sm bg-accent" style={{ width: `${d.confidence * 100}%` }} /></div>
              <span className="font-mono w-9 text-right">{d.confidence.toFixed(2)}</span>
            </div>
            {d.probabilities && (
              <div className="flex flex-col gap-1">
                {Object.entries(d.probabilities).map(([k, p]) => (
                  <div key={k} className="grid grid-cols-[110px_minmax(0,1fr)_40px] items-center gap-2 text-[11px]">
                    <span className="truncate text-muted">{s?.type === "score" ? answerLabel(s, Number(k)) : k}</span>
                    <div className="h-1.5 rounded-sm bg-track relative"><div className="absolute inset-y-0 left-0 rounded-sm bg-accent2" style={{ width: `${p * 100}%` }} /></div>
                    <span className="font-mono text-right">{Math.round(p * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
            <p className="m-0 text-xs text-muted">
              {d.next ? <>→ arrow {d.transition! + 1} to <span className="font-mono text-fg">{d.next}</span></> : "no arrow matched"}
            </p>
            <details className="text-xs">
              <summary className="text-muted cursor-pointer flex items-center gap-1">What the model read <Tip k="smHistory" /></summary>
              <pre className="mt-1.5 p-2 rounded-md bg-surface border border-line overflow-x-auto text-[11px] whitespace-pre-wrap">{JSON.stringify(d.state, null, 2)}</pre>
            </details>
          </div>
        );
      })}
      {running && <p className="m-0 text-sm text-accent animate-pulse">Deciding <span className="font-mono">{running}</span>…</p>}
      {end && (
        <div className="flex flex-col gap-1 text-sm">
          <span className={`font-medium ${STATUS[end.status][1]}`}>{STATUS[end.status][0]}</span>
          {end.message && <span className="text-xs text-muted">{end.message}</span>}
          {end.errors?.map((e, i) => <span key={i} className="text-xs text-warn">{e.step ? `${e.step}: ` : ""}{e.message}</span>)}
          {end.status === "completed" && <span className="text-xs text-muted">Path: {end.path?.join(" → ")} → DONE · {decisions.length} step{decisions.length === 1 ? "" : "s"} · {(total / 1000).toFixed(1)} s</span>}
        </div>
      )}
    </div>
  );
}
