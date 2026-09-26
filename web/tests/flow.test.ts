import { describe, expect, it } from "vitest";
import tool from "../../poc/flows/templates/tool-selection.json";
import triage from "../../poc/flows/templates/customer-triage.json";
import {
  addStep, answerLabel, connect, DONE, type Decision, type Flow, handleId, parseLines, removeEdge, removeStep,
  toEdges, toNodes, updateEdge, updateStep,
} from "../lib/flow";
import * as flowLib from "../lib/flow";

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

describe("fan-out, options from an earlier answer, hand-off", () => {
  const L: Flow = {
    id: "ladder", name: "Ladder", description: "", version: "1.0", start: "kind",
    steps: {
      kind: { id: "kind", type: "choice", instructions: "Kind?", criteria: { inspect: null, research: null }, position: { x: 0, y: 0 },
        fanout: [{ id: "parallel", type: "noul", instructions: "Together?" }],
        transitions: [{ from_answer: null, target: "ESCALATE", condition: { confidence_lt: 0.4 } }, { from_answer: null, target: "tool" }] },
      tool: { id: "tool", type: "choice", instructions: "Tool?", position: { x: 300, y: 0 },
        options_from: { step: "kind", map: { inspect: { Read: null, Bash: null }, research: { WebSearch: null, WebFetch: null } } },
        transitions: [{ from_answer: null, target: "DONE" }] },
    },
  };

  it("draws ESCALATE only when an arrow goes there, and marks it after a hand-off", () => {
    expect(toNodes(T, [], null, null, null).some((n) => n.id === "ESCALATE")).toBe(false);
    const end = { event: "end" as const, status: "escalated" as const, step: "kind", decisions: [] };
    const nodes = toNodes(L, [], null, end, null);
    expect(nodes.find((n) => n.id === "ESCALATE")!.data.status).toBe("visited");
    expect(nodes.find((n) => n.id === "kind")!.data.status).toBe("idle"); // a hand-off is not a failure
  });

  it("offers the union of mapped options, and starts fan-out arrows at their own handles", () => {
    const { answersOf, fromHandle } = flowLib;
    expect(answersOf(L.steps.tool)).toEqual(["Read", "Bash", "WebSearch", "WebFetch"]);
    const f = connect(L, "kind", handleId(true, "parallel"), "DONE");
    const t = f.steps.kind.transitions.at(-1)!;
    expect(t).toEqual({ from_answer: true, from_question: "parallel", target: "DONE" });
    expect(toEdges(f, []).find((e) => e.id === "kind:2")!.sourceHandle).toBe("q:parallel:true");
    expect(fromHandle(L.steps.kind, "q:parallel:*")).toEqual({ from_question: "parallel", from_answer: null });
  });

  it("marks only the own decision of a step; fan-out answers do not move the run", () => {
    const own = { step: "kind", question_id: "kind", transition: 1, next: "tool" } as Decision;
    const fan = { step: "kind", question_id: "parallel", answer: true } as Decision;
    expect(toEdges(L, [own, fan]).filter((e) => e.data.taken).map((e) => e.id)).toEqual(["kind:1"]);
    expect(toNodes(L, [own, fan], null, null, null).find((n) => n.id === "kind")!.data.decision).toBe(own);
  });
});

describe("fan-out edits keep arrows valid", () => {
  it("drops arrows of a removed fan-out question or answer", () => {
    const f: Flow = { id: "x", name: "x", description: "", version: "1.0", start: "a", steps: {
      a: { id: "a", type: "noul", instructions: "A?", position: { x: 0, y: 0 },
        fanout: [{ id: "q", type: "choice", instructions: "Q?", criteria: { one: null, two: null } }],
        transitions: [{ from_question: "q", from_answer: "two", target: "DONE" }, { from_question: "q", from_answer: null, target: "DONE" },
                      { from_answer: null, target: "DONE" }] } } };
    const g = updateStep(f, "a", { fanout: [{ id: "q", type: "choice", instructions: "Q?", criteria: { one: null, three: null } }] });
    expect(g.steps.a.transitions).toEqual([{ from_question: "q", from_answer: null, target: "DONE" }, { from_answer: null, target: "DONE" }]);
    expect(updateStep(f, "a", { fanout: [] }).steps.a.transitions).toEqual([{ from_answer: null, target: "DONE" }]);
  });
});
