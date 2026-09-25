"use client";
// E15 on the Findings page: 81 prompt templates (3 wordings x 4 parts) on BoolQ val, as a 9 x 9 grid.
import { Tip } from "./Tip";

type Variant = { parts: Record<string, number>; accuracy: number; ece: number; flip_rate: number };
type Effect = { level: number; text: string; mean_accuracy: number; mean_ece: number; mean_flip_rate: number };
export type TemplateData = {
  data: { n: number }; parts: Record<string, string[]>; variants: Variant[]; effects: Record<string, Effect[]>;
  spread: { accuracy_min: number; accuracy_max: number; accuracy_sd: number; accuracy_ci95_halfwidth_one_template: number;
    flip_rate_mean: number; flip_rate_max: number; items_that_change_under_some_template: number };
};

const PART_NAMES: Record<string, string> = { system: "System line", state_label: "State label", question_label: "Question label", answer_line: "Answer line" };
const pct = (x: number) => `${Math.round(x * 100)}%`;

export function TemplateGrid({ data, size }: { data: TemplateData | null; size: string }) {
  if (!data) return (
    <section className="rounded-xl border border-line bg-card p-6 text-sm text-muted">
      No template results for {size} yet. Run <code className="font-mono">cd poc && uv run python experiments.py template_sensitivity</code>.
    </section>
  );
  const { parts, spread: sp } = data;
  const at = (s: number, l: number, q: number, a: number) =>
    data.variants.find((v) => v.parts.system === s && v.parts.state_label === l && v.parts.question_label === q && v.parts.answer_line === a)!;
  const rows = [0, 1, 2].flatMap((s) => [0, 1, 2].map((l) => [s, l]));
  const cols = [0, 1, 2].flatMap((q) => [0, 1, 2].map((a) => [q, a]));
  const lo = sp.accuracy_min, hi = sp.accuracy_max;
  return (
    <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2 className="text-[15px] font-semibold flex items-center gap-1">Does the wording matter? 81 templates · {size} <Tip k="templateGrid" /></h2>
        <p className="text-sm text-muted max-w-[900px] leading-relaxed">
          {data.data.n} yes/no questions (BoolQ val), asked with every combination of 3 wordings for 4 prompt parts.
          Accuracy goes from {pct(lo)} to {pct(hi)}; one template&apos;s error bar is ±{pct(sp.accuracy_ci95_halfwidth_one_template)}.
          On average {pct(sp.flip_rate_mean)} of answers flip against minijev&apos;s template (at most {pct(sp.flip_rate_max)}),
          and {pct(sp.items_that_change_under_some_template)} of questions flip under at least one template.
        </p>
      </div>
      <div className="grid xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] gap-6">
        <div className="overflow-x-auto">
          <table className="border-separate border-spacing-0.5 text-[10px] font-mono min-w-[520px]" aria-label="Accuracy per template">
            <thead>
              <tr><th className="text-left text-muted font-normal pr-2">system · state ↓ / question · answer →</th>
                {cols.map(([q, a]) => <th key={`${q}${a}`} className="text-muted font-normal px-0.5" title={`${parts.question_label[q]} / ${parts.answer_line[a]}`}>Q{q + 1}A{a + 1}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map(([s, l]) => (
                <tr key={`${s}${l}`}>
                  <th className="text-left text-muted font-normal pr-2" title={`${parts.system[s]} / ${parts.state_label[l]}`}>S{s + 1} · {parts.state_label[l]}</th>
                  {cols.map(([q, a]) => {
                    const v = at(s, l, q, a), base = !s && !l && !q && !a;
                    const x = hi > lo ? (v.accuracy - lo) / (hi - lo) : 0.5;
                    return (
                      <td key={`${q}${a}`} title={`accuracy ${pct(v.accuracy)} · ECE ${v.ece.toFixed(3)} · flips ${pct(v.flip_rate)}`}
                        className={`w-9 h-7 text-center rounded-sm ${base ? "outline outline-2 outline-fg" : ""}`}
                        style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(15 + x * 70)}%, var(--card))`, color: x > 0.6 ? "var(--inv-fg)" : "var(--fg)" }}>
                        {Math.round(v.accuracy * 100)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-muted mt-2">S1 = minijev&apos;s system line, Q1 = &quot;QUESTION:&quot;, A1 = &quot;Answer with Yes or No.&quot; Hover a cell for ECE and flips. The outlined cell is minijev&apos;s template.</p>
        </div>
        <div className="flex flex-col gap-3">
          <span className="text-[13px] font-medium">Average over the 27 templates that use each wording</span>
          {Object.entries(data.effects).map(([part, levels]) => (
            <div key={part} className="flex flex-col gap-1">
              <span className="text-xs text-muted">{PART_NAMES[part] ?? part}</span>
              {levels.map((e) => (
                <div key={e.level} className="grid grid-cols-[minmax(0,1fr)_56px_64px_60px] gap-2 text-xs items-center">
                  <span className="font-mono truncate" title={e.text}>{e.level === 0 ? "● " : ""}{e.text}</span>
                  <span className="font-mono tabular-nums text-right">{pct(e.mean_accuracy)}</span>
                  <span className="font-mono tabular-nums text-right text-muted">ECE {e.mean_ece.toFixed(2)}</span>
                  <span className="font-mono tabular-nums text-right text-muted">flip {pct(e.mean_flip_rate)}</span>
                </div>
              ))}
            </div>
          ))}
          <p className="text-xs text-muted">● = the wording minijev uses. Accuracy and ECE are before calibration.</p>
        </div>
      </div>
    </section>
  );
}
