import { describe, expect, it } from "vitest";
import tool from "../../poc/flows/templates/tool-selection.json";
import triage from "../../poc/flows/templates/customer-triage.json";
import {
  addStep, answerLabel, connect, DONE, type Decision, type Flow, handleId, parseLines, removeEdge, removeStep,
  toEdges, toNodes, updateEdge, updateStep,
} from "../lib/flow";

const T = tool as unknown as Flow, C = triage as unknown as Flow;

describe("flow ↔ React Flow", () => {
  it("draws one node per step plus DONE, and one edge per transition", () => {
    const nodes = toNodes(T, [], null, null, null), edges = toEdges(T, []);
    expect(nodes.map((n) => n.id).sort()).toEqual([...Object.keys(T.steps), DONE].sort());
    expect(edges).toHaveLength(Object.values(T.steps).reduce((n, s) => n + s.transitions.length, 0));
    expect(nodes.find((n) => n.id === "tool_type")!.data.start).toBe(true);
  });

  it("starts each edge at the handle of its answer; a fallback uses the any-answer handle", () => {
    const e = toEdges(T, []);
    expect(e.find((x) => x.id === "tool_type:0")!.sourceHandle).toBe(handleId("search"));
    expect(e.find((x) => x.id === "search_tool:0")!.sourceHandle).toBe("*");
    expect(e.find((x) => x.id === "tool_type:0")!.label).toBe("search · sure ≥ 0.5");
  });

  it("marks the run: visited steps, the taken arrows, the step being decided", () => {
    const d = { step: "tool_type", transition: 0, next: "search_tool" } as Decision;
    const nodes = toNodes(T, [d], "search_tool", null, null);
    expect(nodes.find((n) => n.id === "tool_type")!.data.status).toBe("visited");
    expect(nodes.find((n) => n.id === "search_tool")!.data.status).toBe("current");
    expect(toEdges(T, [d]).filter((e) => e.data.taken).map((e) => e.id)).toEqual(["tool_type:0"]);
  });

  it("marks the step where a run stopped", () => {
    const end = { event: "end" as const, status: "no_transition" as const, step: "tool_type", decisions: [], message: "no arrow" };
    expect(toNodes(T, [], null, end, null).find((n) => n.id === "tool_type")!.data.status).toBe("failed");
  });
});

describe("edits", () => {
  it("connect adds a transition from the dragged handle", () => {
    const f = connect(T, "contents_or_names", handleId(false), DONE);
    expect(f.steps.contents_or_names.transitions.at(-1)).toEqual({ from_answer: false, target: DONE });
    expect(T.steps.contents_or_names.transitions).toHaveLength(2); // the input is not changed
  });

  it("removeEdge and updateEdge address a transition by step and index", () => {
    expect(removeEdge(T, "tool_type:1").steps.tool_type.transitions.map((t) => t.target)).toEqual(["search_tool", "file_tool", DONE]);
    expect(updateEdge(T, "tool_type:0", { condition: { confidence_gte: 0.8 } }).steps.tool_type.transitions[0].condition).toEqual({ confidence_gte: 0.8 });
  });

  it("removing an option also removes its arrows", () => {
    const f = updateStep(T, "tool_type", { criteria: { search: "a", file: "b" } });
    expect(f.steps.tool_type.transitions.some((t) => t.from_answer === "network")).toBe(false);
  });

  it("addStep picks a free id and removeStep drops arrows into the step", () => {
    const [f, id] = addStep(C, "score", { x: 0, y: 0 });
    expect(id).toBe("step_5");
    expect(f.steps[id].criteria).toEqual(["low", "medium", "high"]);
    const g = removeStep(C, "human");
    expect(Object.values(g.steps).flatMap((s) => s.transitions).some((t) => t.target === "human")).toBe(false);
  });

  it("labels answers in plain words", () => {
    expect(answerLabel(C.steps.upset, 2)).toBe("2 · frustrated");
    expect(answerLabel(C.steps.urgent, true)).toBe("yes");
    expect(answerLabel(undefined, null)).toBe("any answer");
  });
});

it("parses a stream of JSON lines across chunk borders", () => {
  const [a, tail] = parseLines('{"event":"decision","step":"x"}\n{"event":"en');
  expect(a).toHaveLength(1);
  const [b, rest] = parseLines(tail + 'd","status":"completed"}\n');
  expect(b[0].event).toBe("end");
  expect(rest).toBe("");
});
