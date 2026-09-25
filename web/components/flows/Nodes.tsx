"use client";
// The boxes on the State machine canvas: one per step (with one output dot per answer), and DONE.
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { ANY, answerLabel, answersOf, handleId, type NodeData } from "@/lib/flow";

const TYPE = { noul: "Yes / No", choice: "Choice", score: "Score" } as const;

const ring: Record<NodeData["status"], string> = {
  idle: "border-line",
  visited: "border-accent",
  current: "border-accent ring-4 ring-soft animate-pulse",
  failed: "border-warn ring-4 ring-warn/20",
};

export function StepNode({ data, selected }: NodeProps) {
  const { step, start, status, decision, problem } = data as NodeData;
  if (!step) return null;
  const outs = [...answersOf(step).map((a) => ({ id: handleId(a), label: answerLabel(step, a) })), { id: ANY, label: "any answer" }];
  const usedHandles = new Set(step.transitions.map((t) => handleId(t.from_answer)));
  return (
    <div className={`w-[250px] rounded-xl border-2 bg-card text-fg shadow-sm ${ring[status]} ${selected ? "outline outline-2 outline-offset-2 outline-accent" : ""}`}>
      <Handle type="target" position={Position.Left} className="!w-3 !h-3 !bg-fg !border-2 !border-card" />
      <div className="px-3 pt-2.5 pb-2 flex flex-col gap-1">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
          <span className="px-1.5 py-0.5 rounded bg-track text-muted">{TYPE[step.type]}</span>
          {start && <span className="px-1.5 py-0.5 rounded bg-ok/15 text-ok font-semibold">start</span>}
          <span className="ml-auto font-mono normal-case tracking-normal text-muted">{step.id}</span>
        </div>
        <p className="m-0 text-[13px] leading-snug line-clamp-3">{step.instructions || <span className="text-warn">(no question)</span>}</p>
        {decision && (
          <div className="mt-1 flex items-center gap-2 text-xs">
            <span className="px-1.5 py-0.5 rounded bg-accent text-inv-fg font-medium">{answerLabel(step, decision.answer)}</span>
            <span className="text-muted font-mono">sure {decision.confidence.toFixed(2)}</span>
            <span className="ml-auto text-muted font-mono">{(decision.ms / 1000).toFixed(1)} s</span>
          </div>
        )}
        {problem && <p className="m-0 text-[11px] text-warn leading-snug">{problem}</p>}
      </div>
      <div className="border-t border-line py-1">
        {outs.map((o) => (
          <div key={o.id} className="relative flex justify-end items-center h-6 pr-4 text-[11px]">
            <span className={usedHandles.has(o.id) ? "text-fg" : "text-muted"}>{o.label}</span>
            <Handle type="source" id={o.id} position={Position.Right}
              className={`!w-3 !h-3 !border-2 !border-card ${o.id === ANY ? "!bg-ghost" : "!bg-accent"}`} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function DoneNode({ data }: NodeProps) {
  const { status } = data as NodeData;
  return (
    <div className={`w-[110px] h-[56px] rounded-full border-2 grid place-items-center font-mono text-sm font-semibold bg-card text-fg ${status === "visited" ? "border-ok text-ok" : "border-line"}`}>
      <Handle type="target" position={Position.Left} className="!w-3 !h-3 !bg-fg !border-2 !border-card" />
      DONE
    </div>
  );
}
