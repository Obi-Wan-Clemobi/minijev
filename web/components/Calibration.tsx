"use client";
import { useStore } from "@/lib/store";
import type { HelpKey } from "@/lib/help";
import { DEFAULT_SETTINGS, type Settings } from "@/lib/types";
import { Icon } from "./Icon";
import { Tip } from "./Tip";
import { useState } from "react";

// Fitted calibration data from docs/RESEARCH.md §7.3 R6 (E11)
type CalibrationData = { temp: number; ece: number; fitted: boolean };
const FITTED_CALIBRATION: Record<string, Record<string, CalibrationData>> = {
  "0.5B": {
    noul: { temp: 2.72, ece: 0.053, fitted: true },
    // Choice modes not yet fitted (Task 2.2 needs to run fit-choice-temps)
    choice_listwise: { temp: 1.0, ece: 0, fitted: false },
    choice_pointwise: { temp: 1.0, ece: 0, fitted: false },
    choice_averaged: { temp: 1.0, ece: 0, fitted: false },
    // Score modes need ordinal dataset
    score_pointwise: { temp: 1.0, ece: 0, fitted: false },
    score_listwise: { temp: 1.0, ece: 0, fitted: false },
  },
  "1.5B": {
    noul: { temp: 1.93, ece: 0.060, fitted: true },
    choice_listwise: { temp: 1.0, ece: 0, fitted: false },
    choice_pointwise: { temp: 1.0, ece: 0, fitted: false },
    choice_averaged: { temp: 1.0, ece: 0, fitted: false },
    score_pointwise: { temp: 1.0, ece: 0, fitted: false },
    score_listwise: { temp: 1.0, ece: 0, fitted: false },
  },
};

// Fitted on BoolQ (docs/RESEARCH.md §7.3 R6). Only the Noul dials have fitted values so far.
const PRESETS: { label: string; s: Partial<Settings> }[] = [
  { label: "Raw · T = 1", s: { temp_noul: 1, temp_choice: 1, temp_score: 1, bias_noul: 0 } },
  { label: "0.5B BoolQ · T = 2.72", s: { temp_noul: 2.72, bias_noul: 0 } },
  { label: "0.5B Platt · T = 2.92, b = 0.21", s: { temp_noul: 2.92, bias_noul: 0.208 } },
  { label: "1.5B BoolQ · T = 1.93", s: { temp_noul: 1.93, bias_noul: 0 } },
];

type NumKey = "temp_noul" | "bias_noul" | "temp_choice" | "temp_score";
const DIALS: { key: NumKey; label: string; min: number; max: number; help: HelpKey }[] = [
  { key: "temp_noul", label: "Temperature · yes/no (Noul)", min: 0.25, max: 4, help: "tempNoul" },
  { key: "bias_noul", label: "Bias · yes/no (Noul)", min: -2, max: 2, help: "biasNoul" },
  { key: "temp_choice", label: "Temperature · multiple choice", min: 0.25, max: 4, help: "tempChoice" },
  { key: "temp_score", label: "Temperature · scale (Score)", min: 0.25, max: 4, help: "tempScore" },
];

type CalProfile = { label: string; factor: number };
const PROFILES: CalProfile[] = [
  { label: "Conservative", factor: 0.8 },
  { label: "Balanced", factor: 1.0 },
  { label: "Aggressive", factor: 1.2 },
];

export function Calibration() {
  const { settings, setSettings, stale } = useStore();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const matches = (p: Partial<Settings>) => Object.entries(p).every(([k, v]) => Math.abs((settings[k as keyof Settings] as number) - (v as number)) < 1e-9);

  // Determine current model from settings (would need to come from API in real implementation)
  // For now, assume 0.5B as default
  const currentModelSize = "0.5B"; // TODO: Get from model state

  const applyProfile = (factor: number) => {
    setSettings((s) => ({
      ...s,
      temp_noul: s.temp_noul * factor,
      temp_choice: s.temp_choice * factor,
      temp_score: s.temp_score * factor,
      temp_choice_listwise: s.temp_choice_listwise ? s.temp_choice_listwise * factor : undefined,
      temp_choice_pointwise: s.temp_choice_pointwise ? s.temp_choice_pointwise * factor : undefined,
      temp_choice_averaged: s.temp_choice_averaged ? s.temp_choice_averaged * factor : undefined,
      temp_score_pointwise: s.temp_score_pointwise ? s.temp_score_pointwise * factor : undefined,
      temp_score_listwise: s.temp_score_listwise ? s.temp_score_listwise * factor : undefined,
    }));
  };

  const resetToFitted = () => {
    const fitted = FITTED_CALIBRATION[currentModelSize];
    setSettings((s) => ({
      ...s,
      temp_noul: fitted.noul.temp,
      bias_noul: 0,
      temp_choice: 1.0,
      temp_score: 1.0,
      temp_choice_listwise: undefined,
      temp_choice_pointwise: undefined,
      temp_choice_averaged: undefined,
      temp_score_pointwise: undefined,
      temp_score_listwise: undefined,
    }));
  };

  const getFittedData = (primitive: string, mode?: string): CalibrationData | null => {
    const fitted = FITTED_CALIBRATION[currentModelSize];
    if (!fitted) return null;
    const key = mode ? `${primitive}_${mode}` : primitive;
    return fitted[key] || null;
  };

  return (
    <section aria-label="Calibration" className="rounded-[10px] border border-line bg-surface p-4 flex flex-col gap-3.5">
      <div className="flex items-center gap-2 flex-wrap">
        <Icon name="sliders" />
        <h2 className="text-sm font-semibold">Calibration</h2>
        <span className="text-xs text-muted">the dials change the answers instantly · no need to run the model</span>
      </div>

      {/* Calibration Profiles */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">Calibration profiles</span>
          <Tip k="calProfiles" />
        </div>
        <div className="flex gap-1.5 flex-wrap items-center">
          {PROFILES.map((profile) => (
            <button
              key={profile.label}
              onClick={() => applyProfile(profile.factor)}
              className="h-8 px-2.5 rounded-md border border-line text-muted hover:text-fg hover:border-fg font-mono text-xs transition-colors"
            >
              {profile.label} (×{profile.factor})
            </button>
          ))}
          <Tip k="resetAllCal">
            <button
              onClick={resetToFitted}
              className="h-8 px-2.5 rounded-md border border-line text-muted hover:text-fg hover:border-fg font-mono text-xs transition-colors flex items-center gap-1.5"
            >
              <Icon name="reset" size={12} />
              Reset to fitted defaults
            </button>
          </Tip>
        </div>
      </div>

      {/* Quick Presets */}
      <div className="flex gap-1.5 flex-wrap items-center">
        {PRESETS.map((p) => (
          <button key={p.label} onClick={() => setSettings((s) => ({ ...s, ...p.s }))} aria-pressed={matches(p.s)}
            className={`h-8 px-2.5 rounded-md border font-mono text-xs transition-colors ${matches(p.s) ? "border-fg bg-track text-fg" : "border-line text-muted hover:text-fg"}`}>
            {p.label}
          </button>
        ))}
        <Tip k="calPresets" />
      </div>

      {/* Main Temperature Dials */}
      <div className="grid sm:grid-cols-2 gap-x-5 gap-y-3">
        {DIALS.map((d) => {
          const value = settings[d.key], base = DEFAULT_SETTINGS[d.key];
          const fitted = d.key === "temp_noul" ? getFittedData("noul") : null;
          return (
            <div key={d.key} className="flex flex-col gap-1">
              <div className="flex items-center gap-1 text-[13px]">
                <label htmlFor={d.key}>{d.label}</label>
                <Tip k={d.help} />
                {fitted && fitted.fitted && (
                  <Tip k="fittedECE">
                    <span className="px-1.5 py-0.5 rounded bg-green-500/10 text-green-600 dark:text-green-400 text-[10px] font-mono">
                      ECE {fitted.ece.toFixed(3)}
                    </span>
                  </Tip>
                )}
                {fitted && !fitted.fitted && d.key !== "bias_noul" && (
                  <Tip k="unfittedWarning">
                    <span className="px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 text-[10px] font-mono">
                      unfitted
                    </span>
                  </Tip>
                )}
                <span className="flex-1" />
                <span className="font-mono tabular-nums">{value.toFixed(2)}</span>
                <Tip k="reset">
                  <button type="button" aria-label={`Reset ${d.label}`} disabled={value === base}
                    onClick={() => setSettings((s) => ({ ...s, [d.key]: base }))}
                    className="w-6 h-6 rounded grid place-items-center text-muted hover:text-fg hover:bg-track disabled:opacity-30 disabled:hover:bg-transparent">
                    <Icon name="reset" size={12} />
                  </button>
                </Tip>
              </div>
              <input id={d.key} type="range" min={d.min} max={d.max} step={0.01} value={value}
                onChange={(e) => setSettings((s) => ({ ...s, [d.key]: parseFloat(e.target.value) }))} className="w-full h-6" />
              <div className="flex justify-between text-[10px] font-mono text-muted"><span>{d.min}</span><span>{base}</span><span>{d.max}</span></div>
            </div>
          );
        })}
      </div>

      {/* Advanced: Per-Mode Temperature Overrides */}
      <div className="flex flex-col gap-2 pt-2 border-t border-line">
        <div
          onClick={() => setShowAdvanced(!showAdvanced)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowAdvanced(!showAdvanced); }}
          className="flex items-center gap-2 text-xs font-medium hover:text-fg text-muted transition-colors cursor-pointer"
        >
          <Icon name={showAdvanced ? "chevron-down" : "chevron-right"} size={12} />
          Per-mode temperature overrides
          <Tip k="perModeTemp" />
        </div>
        {showAdvanced && (
          <div className="grid sm:grid-cols-2 gap-x-5 gap-y-2.5 pl-5">
            {/* Choice mode overrides */}
            {(["listwise", "pointwise", "averaged"] as const).map((mode) => {
              const key = `temp_choice_${mode}` as keyof Settings;
              const value = settings[key] as number | undefined;
              const fitted = getFittedData("choice", mode);
              return (
                <div key={key} className="flex items-center gap-2 text-[12px]">
                  <label htmlFor={key} className="flex-1">Choice · {mode}</label>
                  {fitted && fitted.fitted && (
                    <Tip k="fittedECE">
                      <span className="px-1.5 py-0.5 rounded bg-green-500/10 text-green-600 dark:text-green-400 text-[10px] font-mono">
                        ECE {fitted.ece.toFixed(3)}
                      </span>
                    </Tip>
                  )}
                  {fitted && !fitted.fitted && (
                    <Tip k="unfittedWarning">
                      <span className="px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 text-[10px] font-mono">
                        unfitted
                      </span>
                    </Tip>
                  )}
                  <input
                    id={key}
                    type="number"
                    min={0.25}
                    max={4}
                    step={0.01}
                    value={value ?? ""}
                    placeholder={settings.temp_choice.toFixed(2)}
                    onChange={(e) => {
                      const val = e.target.value === "" ? undefined : parseFloat(e.target.value);
                      setSettings((s) => ({ ...s, [key]: val }));
                    }}
                    className="h-7 w-20 px-2 rounded-md border border-line bg-card font-mono text-xs"
                  />
                </div>
              );
            })}
            {/* Score mode overrides */}
            {(["pointwise", "listwise"] as const).map((mode) => {
              const key = `temp_score_${mode}` as keyof Settings;
              const value = settings[key] as number | undefined;
              const fitted = getFittedData("score", mode);
              return (
                <div key={key} className="flex items-center gap-2 text-[12px]">
                  <label htmlFor={key} className="flex-1">Score · {mode}</label>
                  {fitted && fitted.fitted && (
                    <Tip k="fittedECE">
                      <span className="px-1.5 py-0.5 rounded bg-green-500/10 text-green-600 dark:text-green-400 text-[10px] font-mono">
                        ECE {fitted.ece.toFixed(3)}
                      </span>
                    </Tip>
                  )}
                  {fitted && !fitted.fitted && (
                    <Tip k="unfittedWarning">
                      <span className="px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 text-[10px] font-mono">
                        unfitted
                      </span>
                    </Tip>
                  )}
                  <input
                    id={key}
                    type="number"
                    min={0.25}
                    max={4}
                    step={0.01}
                    value={value ?? ""}
                    placeholder={settings.temp_score.toFixed(2)}
                    onChange={(e) => {
                      const val = e.target.value === "" ? undefined : parseFloat(e.target.value);
                      setSettings((s) => ({ ...s, [key]: val }));
                    }}
                    className="h-7 w-20 px-2 rounded-md border border-line bg-card font-mono text-xs"
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>

      {stale && (
        <div className="flex items-center gap-2 pt-1 border-t border-line text-xs text-muted">
          <span className="px-1.5 py-0.5 rounded bg-track text-warn font-medium">Needs a Run</span>
          press Run after you change these
        </div>
      )}

      {/* Mode selectors */}
      <div className="grid sm:grid-cols-2 gap-3 text-[13px]">
        <label className="flex flex-col gap-1">
          <span className="flex items-center gap-1">Multiple choice asked as <Tip k="defaultChoiceMode" /></span>
          <select value={settings.choice_mode} onChange={(e) => setSettings((s) => ({ ...s, choice_mode: e.target.value as Settings["choice_mode"] }))}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs"><option>listwise</option><option>pointwise</option><option>averaged</option></select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center gap-1">Scale questions asked as <Tip k="defaultScoreMode" /></span>
          <select value={settings.score_mode} onChange={(e) => setSettings((s) => ({ ...s, score_mode: e.target.value as Settings["score_mode"] }))}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs"><option>pointwise</option><option>listwise</option></select>
        </label>
      </div>

      <label className="flex items-center gap-2 pt-3 border-t border-line text-[13px]">
        <span className="flex items-center gap-1">Warn below label mass <Tip k="labelMass" /></span>
        <input type="number" min={0} max={1} step={0.05} value={settings.min_label_mass}
          onChange={(e) => setSettings((s) => ({ ...s, min_label_mass: Math.min(1, Math.max(0, parseFloat(e.target.value) || 0)) }))}
          className="h-8 w-20 px-2 rounded-md border border-line bg-card font-mono text-xs" />
      </label>

      <div className="flex items-start gap-1">
        <p className="text-xs text-muted font-mono break-all flex-1">
          poc/minijev.env: MINIJEV_TEMP_NOUL={settings.temp_noul.toFixed(2)} MINIJEV_BIAS_NOUL={settings.bias_noul.toFixed(3)} MINIJEV_TEMP_CHOICE={settings.temp_choice.toFixed(2)} MINIJEV_TEMP_SCORE={settings.temp_score.toFixed(2)} MINIJEV_CHOICE_MODE={settings.choice_mode} MINIJEV_SCORE_MODE={settings.score_mode}
        </p>
        <Tip k="envLine" />
      </div>
    </section>
  );
}
