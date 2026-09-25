import { describe, expect, it } from "vitest";
import { calibrator } from "../lib/scoring";
import { DEFAULT_SETTINGS, type Fitted, type Question } from "../lib/types";

// The same cases as poc/tests/test_maths.py::test_fitted_calibration_is_used_and_overridable.
const fitted = {
  noul: { temperature: 2.0, platt: { a: 0.5, b: 0.3 }, selected: "platt" },
  choice: { temperature: { listwise: 3.0, pointwise: 1.5, averaged: 2.5 }, selected_mode: "pointwise" },
} as unknown as Fitted;
const noul: Question = { type: "noul", instructions: "" };
const choice: Question = { type: "choice", instructions: "" };
const score: Question = { type: "score", instructions: "" };

describe("calibrator mirrors Settings.calibrator()", () => {
  it("uses the fitted values", () => {
    expect(calibrator(noul, DEFAULT_SETTINGS, fitted)).toEqual({ t: 2.0, b: 0.3, source: "fitted" });
    expect(calibrator(choice, DEFAULT_SETTINGS, fitted)).toEqual({ t: 1.5, b: 0, source: "fitted" });
    expect(calibrator({ ...choice, choice_mode: "listwise" }, DEFAULT_SETTINGS, fitted)).toEqual({ t: 3.0, b: 0, source: "fitted" });
    expect(calibrator(score, DEFAULT_SETTINGS, fitted)).toEqual({ t: 1, b: 0, source: "none" });
  });
  it("lets an explicit dial win, and raw mode ignore the file", () => {
    expect(calibrator(choice, { ...DEFAULT_SETTINGS, temp_choice: 4 }, fitted)).toEqual({ t: 4, b: 0, source: "manual" });
    expect(calibrator(noul, { ...DEFAULT_SETTINGS, calibration: "none" }, fitted)).toEqual({ t: 1, b: 0, source: "none" });
    expect(calibrator(noul, DEFAULT_SETTINGS, null)).toEqual({ t: 1, b: 0, source: "none" });
  });
});
