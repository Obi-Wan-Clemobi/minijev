"use client";
import { useEffect } from "react";
import { Calibration } from "@/components/Calibration";
import { Icon } from "@/components/Icon";
import { PresetBar, RequestEditor, useReady } from "@/components/RequestEditor";
import { Response } from "@/components/Response";
import { Tip } from "@/components/Tip";
import { useStore } from "@/lib/store";

export default function Playground() {
  const s = useStore();
  const ready = useReady();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && ready && !s.running) { e.preventDefault(); s.run(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ready, s]);

  return (
    <div className="flex-1 flex flex-col">
      <div className="min-h-16 px-4 md:px-8 py-3 border-b border-line flex items-center gap-3 flex-wrap">
        <span className="text-[13px] text-muted">Examples</span>
        <Tip k="presets" />
        <PresetBar />
        <div className="flex-1" />
        <span className="text-[13px] text-muted">Evaluation</span>
        <Tip k="evaluation" />
        <div className="flex border border-line rounded-lg p-[3px] gap-0.5 font-mono">
          {["naive", "kv", "packed"].map((m) => (
            <button key={m} onClick={() => s.setMode(m)} aria-pressed={s.mode === m}
              className={`h-[30px] px-2.5 rounded-[5px] text-xs transition-colors ${s.mode === m ? "bg-track text-fg" : "text-muted hover:text-fg"}`}>{m}</button>
          ))}
        </div>
        <Tip k="run">
          <button onClick={s.run} disabled={!ready || s.running}
            className="h-10 px-4 rounded-lg bg-inv-bg text-inv-fg text-sm font-medium flex items-center gap-2.5 disabled:opacity-40 transition-opacity">
            <Icon name="play" size={14} fill />{s.switching ? "Loading model…" : s.running ? "Running…" : "Run"}<span className="font-mono text-[11px] opacity-60">⌘↵</span>
          </button>
        </Tip>
      </div>

      <div className="flex-1 grid lg:grid-cols-[minmax(460px,560px)_minmax(0,1fr)]">
        <div className="lg:border-r border-line px-4 md:px-8 py-6"><RequestEditor><Calibration /></RequestEditor></div>
        <div className="px-4 md:px-8 py-6 bg-surface min-w-0"><Response /></div>
      </div>
    </div>
  );
}
