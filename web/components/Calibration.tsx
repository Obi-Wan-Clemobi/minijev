"use client";
// Calibration dials. The default is the fitted calibration for the loaded model (calibration/<model>.json):
// fitted on the train split, chosen on the val split, reported on the test split (experiments.py heldout, E14).
// Moving a dial overrides the fitted value for that primitive; Reset returns to the fitted value.
import { calibrator } from "@/lib/scoring";
import { useStore } from "@/lib/store";
import type { HelpKey } from "@/lib/help";
import type { Question, Settings } from "@/lib/types";
import { Icon } from "./Icon";
import { Tip } from "./Tip";

type NumKey = "temp_noul" | "bias_noul" | "temp_choice" | "temp_score";
const DIALS: { key: NumKey; label: string; min: number; max: number; help: HelpKey; q: Question }[] = [
  { key: "temp_noul", label: "Temperature · yes/no (Noul)", min: 0.25, max: 5, help: "tempNoul", q: { type: "noul", instructions: "" } },
  { key: "bias_noul", label: "Bias · yes/no (Noul)", min: -2, max: 2, help: "biasNoul", q: { type: "noul", instructions: "" } },
  { key: "temp_choice", label: "Temperature · multiple choice", min: 0.25, max: 5, help: "tempChoice", q: { type: "choice", instructions: "" } },
  { key: "temp_score", label: "Temperature · scale (Score)", min: 0.25, max: 5, help: "tempScore", q: { type: "score", instructions: "" } },
];
const SOURCE = { manual: ["manual", "border-warn text-warn"], fitted: ["fitted", "border-accent text-accent"], none: ["raw", "border-line text-muted"] } as const;

export function Calibration() {
  const { settings, setSettings, fitted, model } = useStore();
  const set = (patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch }));
  const allFitted = settings.calibration === "fitted" && DIALS.every((d) => settings[d.key] === null);
  const allRaw = settings.calibration === "none" && DIALS.every((d) => settings[d.key] === null);
  const selected = fitted?.choice.selected_mode;
  const prov = fitted?.provenance;

  return (
    <section aria-label="Calibration" className="rounded-[10px] border border-line bg-surface p-4 flex flex-col gap-3.5">
      <div className="flex items-center gap-2 flex-wrap">
        <Icon name="sliders" />
        <h2 className="text-sm font-semibold">Calibration</h2>
        <span className="text-xs text-muted">the dials change the answers instantly · no need to run the model</span>
      </div>

      {fitted && prov ? (
        <div className="text-xs leading-relaxed rounded-md border border-line bg-card p-2.5 flex gap-2">
          <Icon name="check" size={14} className="text-ok shrink-0 mt-0.5" />
          <span>
            Fitted for <span className="font-mono">{model.split("/")[1]}</span> on the <strong>train</strong> split
            ({prov.fitted_on.noul.dataset} n = {prov.fitted_on.noul.n}; {prov.fitted_on.choice.dataset} n = {prov.fitted_on.choice.n}),
            chosen on the <strong>val</strong> split, reported on the <strong>test</strong> split.
            Splits file <span className="font-mono">{prov.splits_sha256.slice(0, 12)}</span> · {fitted.created}.
            <Tip k="calProvenance" />
          </span>
        </div>
      ) : (
        <div className="text-xs leading-relaxed rounded-md border border-warn p-2.5 flex gap-2">
          <Icon name="alert" size={14} className="text-warn shrink-0 mt-0.5" />
          <span>No fitted calibration for this model yet, so the answers are raw. Run <code className="font-mono">uv run python experiments.py heldout</code> in poc/.</span>
        </div>
      )}

      <div className="flex gap-1.5 flex-wrap items-center">
        <button onClick={() => set({ calibration: "fitted", temp_noul: null, bias_noul: null, temp_choice: null, temp_score: null })} aria-pressed={allFitted}
          className={`h-8 px-2.5 rounded-md border text-xs transition-colors ${allFitted ? "border-fg bg-track text-fg" : "border-line text-muted hover:text-fg"}`}>
          Fitted on the train split
        </button>
        <button onClick={() => set({ calibration: "none", temp_noul: null, bias_noul: null, temp_choice: null, temp_score: null })} aria-pressed={allRaw}
          className={`h-8 px-2.5 rounded-md border text-xs transition-colors ${allRaw ? "border-fg bg-track text-fg" : "border-line text-muted hover:text-fg"}`}>
          Raw · T = 1
        </button>
        <Tip k="calPresets" />
      </div>

      <div className="grid sm:grid-cols-2 gap-x-5 gap-y-3">
        {DIALS.map((d) => {
          const c = calibrator({ ...d.q, choice_mode: d.key === "temp_choice" ? selected : undefined }, settings, fitted);
          const shown = d.key === "bias_noul" ? c.b : c.t;
          const [badge, tone] = SOURCE[c.source];
          return (
            <div key={d.key} className="flex flex-col gap-1">
              <div className="flex items-center gap-1 text-[13px]">
                <label htmlFor={d.key}>{d.label}</label>
                <Tip k={d.help} />
                <span className="flex-1" />
                <span className={`text-[10px] px-1.5 rounded-full border ${tone}`}>{badge}</span>
                <span className="font-mono tabular-nums">{shown.toFixed(2)}</span>
                <Tip k="reset">
                  <button type="button" aria-label={`Reset ${d.label} to the fitted value`} disabled={settings[d.key] === null}
                    onClick={() => set({ [d.key]: null } as Partial<Settings>)}
                    className="w-6 h-6 rounded grid place-items-center text-muted hover:text-fg hover:bg-track disabled:opacity-30 disabled:hover:bg-transparent">
                    <Icon name="reset" size={12} />
                  </button>
                </Tip>
              </div>
              <input id={d.key} type="range" min={d.min} max={d.max} step={0.01} value={shown}
                onChange={(e) => set({ [d.key]: parseFloat(e.target.value) } as Partial<Settings>)} className="w-full h-6" />
              <div className="flex justify-between text-[10px] font-mono text-muted"><span>{d.min}</span><span>{d.max}</span></div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2 pt-1 border-t border-line text-xs text-muted">
        <span className="px-1.5 py-0.5 rounded bg-track text-warn font-medium">Needs a Run</span>
        press Run after you change these
      </div>
      <div className="grid sm:grid-cols-2 gap-3 text-[13px]">
        <label className="flex flex-col gap-1">
          <span className="flex items-center gap-1">Multiple choice asked as <Tip k="defaultChoiceMode" /></span>
          <select value={settings.choice_mode} onChange={(e) => set({ choice_mode: e.target.value as Settings["choice_mode"] })}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs">
            <option value="selected">selected on val{selected ? ` (${selected})` : ""}</option>
            <option>listwise</option><option>pointwise</option><option>averaged</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center gap-1">Scale questions asked as <Tip k="defaultScoreMode" /></span>
          <select value={settings.score_mode} onChange={(e) => set({ score_mode: e.target.value as Settings["score_mode"] })}
            className="h-8 px-2 rounded-md border border-line bg-card font-mono text-xs"><option>pointwise</option><option>listwise</option></select>
        </label>
      </div>
      <label className="flex items-center gap-2 pt-3 border-t border-line text-[13px]">
        <span className="flex items-center gap-1">Warn below label mass <Tip k="labelMass" /></span>
        <input type="number" min={0} max={1} step={0.05} value={settings.min_label_mass}
          onChange={(e) => set({ min_label_mass: Math.min(1, Math.max(0, parseFloat(e.target.value) || 0)) })}
          className="h-8 w-20 px-2 rounded-md border border-line bg-card font-mono text-xs" />
      </label>
      <div className="flex items-start gap-1">
        <p className="text-xs text-muted font-mono break-all flex-1">
          poc/minijev.env: MINIJEV_CALIBRATION={settings.calibration} MINIJEV_TEMP_NOUL={settings.temp_noul ?? "fitted"} MINIJEV_BIAS_NOUL={settings.bias_noul ?? "fitted"} MINIJEV_TEMP_CHOICE={settings.temp_choice ?? "fitted"} MINIJEV_TEMP_SCORE={settings.temp_score ?? "fitted"} MINIJEV_CHOICE_MODE={settings.choice_mode} MINIJEV_SCORE_MODE={settings.score_mode}
        </p>
        <Tip k="envLine" />
      </div>
    </section>
  );
}
