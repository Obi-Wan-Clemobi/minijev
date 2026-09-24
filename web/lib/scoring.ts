// A port of minijev_poc.py: softmax, answer(), choice_confidence(), score_confidence().
// The page re-scores the returned raw logits with it, so temperature changes need no model run.
// tests/scoring.test.ts checks it against fixtures that the Python code wrote.
import type { Question, Raw, Settings } from "./types";

export function softmax(z: number[], t = 1): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / t));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

export function choiceConfidence(p: number[]): number {
  const u = 1 / p.length;
  return (Math.max(...p) - u) / (1 - u);
}

export function scoreConfidence(p: number[]): number {
  const k = p.length;
  const mode = p.indexOf(Math.max(...p));
  let spread = 0, uniform = 0;
  for (let i = 0; i < k; i++) {
    spread += p[i] * Math.abs(i - mode);
    uniform += Math.abs(i - (k - 1) / 2) / k;
  }
  return Math.max(0, 1 - spread / uniform);
}

export type NoulAnswer = { type: "noul"; noul: number };
export type ChoiceAnswer = { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number };
export type ScoreAnswer = {
  type: "score"; score: number; legend: Record<string, string>;
  probabilities: Record<string, number>; confidence: number;
};
export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export function levelText(level: unknown): string {
  return typeof level === "string" ? level : JSON.stringify(level);
}

export function answer(q: Question, logits: number[], temperature = 1, bias = 0): Answer {
  if (q.type === "noul") return { type: "noul", noul: sigmoid((logits[0] - logits[1]) / temperature + bias) };
  const p = softmax(logits, temperature);
  if (q.type === "choice") {
    const keys = Object.keys(q.criteria as Record<string, unknown>);
    const best = p.indexOf(Math.max(...p));
    return {
      type: "choice", choice: keys[best],
      probabilities: Object.fromEntries(keys.map((k, i) => [k, p[i]])),
      confidence: choiceConfidence(p),
    };
  }
  const levels = q.criteria as unknown[];
  return {
    type: "score",
    score: p.reduce((a, pi, i) => a + i * pi, 0),
    legend: Object.fromEntries(levels.map((l, i) => [String(i), levelText(l)])),
    probabilities: Object.fromEntries(p.map((pi, i) => [String(i), pi])),
    confidence: scoreConfidence(p),
  };
}

export function rescore(questions: Record<string, Question>, raw: Raw, s: Settings): Record<string, Answer> {
  const temp = { noul: s.temp_noul, choice: s.temp_choice, score: s.temp_score };
  return Object.fromEntries(Object.entries(questions).map(([id, q]) =>
    [id, answer(q, raw[id].logits, temp[q.type], q.type === "noul" ? s.bias_noul : 0)]));
}
