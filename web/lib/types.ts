export type QType = "noul" | "choice" | "score";

export type Question = {
  type: QType;
  instructions: string;
  // Noul: {true?, false?}; Choice: {option: description | null}; Score: ordered levels.
  criteria?: Record<string, string | null> | string[] | { true?: string; false?: string };
  choice_mode?: "listwise" | "pointwise";
  score_mode?: "pointwise" | "listwise";
};

export type Req = { state: unknown; questions: Record<string, Question> };

export type Settings = {
  temp_noul: number;
  temp_choice: number;
  temp_score: number;
  bias_noul: number;
  choice_mode: "listwise" | "pointwise";
  score_mode: "pointwise" | "listwise";
  min_label_mass: number;
};

export const DEFAULT_SETTINGS: Settings = {
  temp_noul: 1, temp_choice: 1, temp_score: 1, bias_noul: 0,
  choice_mode: "listwise", score_mode: "pointwise", min_label_mass: 0.5,
};

export type Raw = Record<string, { logits: number[]; mass: number[] }>;

export type AskResponse = {
  model: string;
  answers: Record<string, unknown>;
  usage: { input_tokens: number; output_tokens: number; latency_ms: number };
  debug: { raw: Raw; warnings: string[] };
  questions: Record<string, Question>;
};

export type Preset = { id: string; name: string; state: unknown; questions: Record<string, Question>; jev?: Record<string, unknown> };
