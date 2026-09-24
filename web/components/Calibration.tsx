"use client";
import { useStore } from "@/lib/store";
import type { Settings } from "@/lib/types";
import { Icon } from "./Icon";

// Fitted on BoolQ (docs/RESEARCH.md §7.3 R6). Only the Noul dials have fitted values so far.
const PRESETS: { label: string; s: Partial<Settings> }[] = [
  { label: "Raw · T = 1", s: { temp_noul: 1, temp_choice: 1, temp_score: 1, bias_noul: 0 } },
  { label: "0.5B BoolQ · T = 2.72", s: { temp_noul: 2.72, bias_noul: 0 } },
  { label: "0.5B Platt · T = 2.92, b = 0.21", s: { temp_noul: 2.92, bias_noul: 0.208 } },
  { label: "1.5B BoolQ · T = 1.93", s: { temp_noul: 1.93, bias_noul: 0 } },
];

const DIALS: { key: keyof Settings; label: string; min: number; max: number }[] = [
  { key: "temp_noul", label: "Temperature · Noul", min: 0.25, max: 4 },
  { key: "bias_noul", label: "Bias · Noul (Platt b)", min: -2, max: 2 },
  { key: "temp_choice", label: "Temperature · Choice", min: 0.25, max: 4 },
  { key: "temp_score", label: "Temperature · Score", min: 0.25, max: 4 },
];

export function Calibration() {
  const { settings, setSettings } = useStore();
  const matches = (p: Partial<Settings>) => Object.entries(p).every(([k, v]) => Math.abs((settings[k as keyof Settings] as number) - (v as number)) < 1e-9);
  return (
    <section aria-label="Calibration" className="rounded-[10px] border border-line bg-surface p-4 flex flex-col gap-3.5">
      <div className="flex items-center gap-2 flex-wrap">
        <Icon name="sliders" />
        <h2 className="text-sm font-semibold">Calibration</h2>
        <span className="text-xs text-muted">re-scores the returned logits instantly · no model re-run</span>
      </div>
      <div className="flex gap-1.5 flex-wrap">
        {PRESETS.map((p) => (
          <button key={p.label} onClick={() => setSettings((s) => ({ ...s, ...p.s }))}
            className={`h-8 px-2.5 rounded-md border font-mono text-xs transition-colors ${matches(p.s) ? "border-fg bg-track text-fg" : "border-line text-muted hover:text-fg"}`}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="grid sm:grid-cols-2 gap-x-5 gap-y-3">
        {DIALS.map((d) => (
          <div key={d.key} className="flex flex-col gap-1">
            <div className="flex justify-between text-[13px]">
              <label htmlFor={d.key}>{d.label}</label>
              <span className="font-mono tabular-nums">{(settings[d.key] as number).toFixed(2)}</span>
            </div>
            <input id={d.key} type="range" min={d.min} max={d.max} step={0.01} value={settings[d.key] as number}
              onChange={(e) => setSettings((s) => ({ ...s, [d.key]: parseFloat(e.target.value) }))} className="w-full h-6" />
          </div>
        ))}
      </div>
      <div className="grid sm:grid-cols-3 gap-3 text-[13px]">
        <label className="flex flex-col gap-1">Choice default
          <select value={settings.choice_mode} onChange={(e) => setSettings((s) => ({ ...s, choice_mode: e.target.value as Settings["choice_mode"] }))}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs"><option>listwise</option><option>pointwise</option></select>
        </label>
        <label className="flex flex-col gap-1">Score default
          <select value={settings.score_mode} onChange={(e) => setSettings((s) => ({ ...s, score_mode: e.target.value as Settings["score_mode"] }))}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs"><option>pointwise</option><option>listwise</option></select>
        </label>
        <label className="flex flex-col gap-1">Warn below label mass
          <input type="number" min={0} max={1} step={0.05} value={settings.min_label_mass}
            onChange={(e) => setSettings((s) => ({ ...s, min_label_mass: parseFloat(e.target.value) || 0 }))}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs" />
        </label>
      </div>
      <p className="text-xs text-muted font-mono break-all">
        poc/minijev.env: MINIJEV_TEMP_NOUL={settings.temp_noul.toFixed(2)} MINIJEV_BIAS_NOUL={settings.bias_noul.toFixed(3)} MINIJEV_TEMP_CHOICE={settings.temp_choice.toFixed(2)} MINIJEV_TEMP_SCORE={settings.temp_score.toFixed(2)}
      </p>
      <p className="text-xs text-muted">Readout modes change the branches, so they apply on the next Run.</p>
    </section>
  );
}
