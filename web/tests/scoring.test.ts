import { describe, expect, it } from "vitest";
import fixture from "./scoring-fixture.json";
import { answer } from "../lib/scoring";
import type { Question } from "../lib/types";

// Cases written by poc/tests/scoring_fixture.py with the Python answer().
describe("scoring.ts matches minijev_poc.answer()", () => {
  it.each(fixture.map((c, i) => [i, c] as const))("case %i", (_, c) => {
    const got = answer(c.q as Question, c.logits, c.t, c.b) as Record<string, unknown>;
    const want = c.expected as Record<string, unknown>;
    for (const [k, v] of Object.entries(want)) {
      if (typeof v === "number") expect(got[k] as number).toBeCloseTo(v, 10);
      else if (v && typeof v === "object" && k === "probabilities")
        for (const [kk, vv] of Object.entries(v)) expect((got[k] as Record<string, number>)[kk]).toBeCloseTo(vv as number, 10);
      else expect(got[k]).toEqual(v);
    }
  });
});
