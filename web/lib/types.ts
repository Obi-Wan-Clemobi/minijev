export type QType = "noul" | "choice" | "score";

export type Question = {
  type: QType;
  instructions: string;
  // Noul: {true?, false?}; Choice: {option: description | null}; Score: ordered levels.
  criteria?: Record<string, string | null> | string[] | { true?: string; false?: string };
  choice_mode?: "listwise" | "pointwise" | "averaged";
  score_mode?: "pointwise" | "listwise";
  contrastive?: boolean;
};

export type Req = { state: unknown; questions: Record<string, Question> };

// Mirrors poc/minijev_poc.py Settings. A temperature or bias of null means "use the fitted value"
// (calibration/<model>.json, fitted on the train split), or 1.0 / 0.0 when calibration is "none".
export type Settings = {
  calibration: "fitted" | "none";
  temp_noul: number | null;
  temp_choice: number | null;
  temp_score: number | null;
  bias_noul: number | null;
  choice_mode: "listwise" | "pointwise" | "averaged" | "selected";
  score_mode: "pointwise" | "listwise";
  score_contrastive: boolean;
  min_label_mass: number;
};

export const DEFAULT_SETTINGS: Settings = {
  calibration: "fitted", temp_noul: null, temp_choice: null, temp_score: null, bias_noul: null,
  choice_mode: "selected", score_mode: "pointwise", score_contrastive: false, min_label_mass: 0.5,
};

// calibration/<model>.json, as /v1/calibration returns it.
export type Fitted = {
  model: string; created: string;
  provenance: { splits_file: string; splits_sha256: string; selected_on: { split: string; rule: string };
    fitted_on: { noul: { dataset: string; split: string; n: number }; choice: { dataset: string; split: string; n: number } } };
  noul: { temperature: number; platt: { a: number; b: number }; selected: "temperature" | "platt" };
  choice: { temperature: Record<string, number>; selected_mode: "listwise" | "pointwise" | "averaged" };
  score: null;
};

export type Raw = Record<string, { logits: number[]; mass: number[] }>;

export type AskResponse = {
  model: string;
  answers: Record<string, unknown>;
  usage: { input_tokens: number; output_tokens: number; latency_ms: number };
  debug: { raw: Raw; warnings: string[]; calibration?: Record<string, { temperature: number; bias: number; source: string }> };
  questions: Record<string, Question>;
};

export type Preset = { id: string; name: string; state: unknown; questions: Record<string, Question>; jev?: Record<string, unknown> };
