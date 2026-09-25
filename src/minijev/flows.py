"""Flows: chain minijev questions into a state machine (docs/superpowers/specs/2026-09-24-state-machine-flow-builder-design.md).

A flow is a set of steps. Each step asks one question (Noul, Choice or Score) about a request. Its answer, and
optionally its confidence, picks the transition to the next step, until a transition reaches DONE. Every answer is
added to the decisions list, and the next step's state holds the request plus those decisions, so later questions see
what earlier ones decided.

How an answer picks a transition (the page explains the same rules):
- Noul: the answer is true when P(yes) >= 0.5. Confidence = 2 * max(p, 1 - p) - 1: 0 at p = 0.5, 1 at p = 0 or 1
  (the Choice formula with two options).
- Choice: the answer is the most likely option. Confidence = (p_max - 1/k) / (1 - 1/k), as in the Jev response.
- Score: the answer is the expected level rounded to the nearest level (0 to n-1). Confidence is the ordinal one.
Transitions of a step are checked in order; the first one whose answer (or "any") and condition match wins.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Literal

from pydantic import BaseModel, Field

DONE = "DONE"
MAX_STEPS = 20


class Condition(BaseModel):
    confidence_gte: float | None = None
    confidence_lt: float | None = None


class Transition(BaseModel):
    from_answer: str | bool | int | None = Field(None, description="None matches any answer (a fallback)")
    target: str
    condition: Condition | None = None


class Step(BaseModel):
    id: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict | list | None = None
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    transitions: list[Transition] = Field(default_factory=list)


class Flow(BaseModel):
    id: str
    name: str
    description: str = ""
    version: str = "1.0"
    start: str
    steps: dict[str, Step]


def question(step: Step) -> dict:
    q = {"type": step.type, "instructions": step.instructions}
    if step.criteria is not None:
        q["criteria"] = step.criteria
    return q


def outcome(step: Step, a: dict) -> tuple[str | bool | int, float, dict | None]:
    """(answer, confidence, probabilities) of one Jev-shaped answer, by the rules in the module docstring."""
    if step.type == "noul":
        p = a["noul"]
        return p >= 0.5, 2 * max(p, 1 - p) - 1, {"yes": p, "no": 1 - p}
    if step.type == "choice":
        return a["choice"], a["confidence"], a["probabilities"]
    return int(math.floor(a["score"] + 0.5)), a["confidence"], a["probabilities"]


def matches(t: Transition, answer, confidence: float) -> bool:
    if t.from_answer is not None and t.from_answer != answer:
        return False
    c = t.condition
    if c and c.confidence_gte is not None and confidence < c.confidence_gte:
        return False
    if c and c.confidence_lt is not None and confidence >= c.confidence_lt:
        return False
    return True


def answers_of(step: Step) -> list:
    """Every answer a step can give: True/False, the option keys, or the level numbers."""
    if step.type == "noul":
        return [True, False]
    if step.type == "choice":
        return list(step.criteria or {})
    return list(range(len(step.criteria or [])))


def check(flow: Flow) -> dict:
    """Errors stop a run; warnings do not. Returns {"errors": [...], "warnings": [...]}, each {"step", "message"}."""
    errors, warnings = [], []
    if flow.start not in flow.steps:
        errors.append({"step": None, "message": f"the start step {flow.start!r} does not exist"})
    for sid, s in flow.steps.items():
        if s.id != sid:
            errors.append({"step": sid, "message": f"the key {sid!r} and the id {s.id!r} differ"})
        if not s.instructions.strip():
            errors.append({"step": sid, "message": "the question is empty"})
        if s.type == "choice" and (not isinstance(s.criteria, dict) or len(s.criteria) < 2):
            errors.append({"step": sid, "message": "a Choice needs at least 2 options"})
        if s.type == "score" and (not isinstance(s.criteria, list) or not 2 <= len(s.criteria) <= 10):
            errors.append({"step": sid, "message": "a Score needs 2 to 10 levels"})
        for t in s.transitions:
            if t.target != DONE and t.target not in flow.steps:
                errors.append({"step": sid, "message": f"an arrow goes to {t.target!r}, which does not exist"})
            if t.from_answer is not None and t.from_answer not in answers_of(s):
                errors.append({"step": sid, "message": f"an arrow starts at {t.from_answer!r}, which is not an answer"})
        if not s.transitions:
            warnings.append({"step": sid, "message": "no arrow leaves this step: a run that reaches it stops"})
        else:
            open_ = [a for a in answers_of(s) if not any(t.from_answer in (a, None) for t in s.transitions)]
            if open_:
                warnings.append({"step": sid, "message": f"no arrow for the answer(s) {open_}"})
            unsure = [a for a in answers_of(s) if a not in open_ and not any(
                t.from_answer in (a, None) and not (t.condition and (t.condition.confidence_gte is not None or t.condition.confidence_lt is not None))
                for t in s.transitions)]
            if unsure:
                warnings.append({"step": sid, "message": f"the answer(s) {unsure} have only 'if sure' arrows: "
                                 "a less sure answer stops the run here"})
    seen, todo = set(), [flow.start] if flow.start in flow.steps else []
    while todo:
        sid = todo.pop()
        if sid in seen:
            continue
        seen.add(sid)
        todo += [t.target for t in flow.steps[sid].transitions if t.target in flow.steps]
    for sid in flow.steps.keys() - seen:
        warnings.append({"step": sid, "message": "no path from the start reaches this step"})
    return {"errors": errors, "warnings": warnings}


def state_for(query: str, decisions: list[dict]) -> dict:
    """What the model reads at each step: the request, and what the earlier steps decided."""
    return {"request": query, "decisions so far": [
        {"question": d["question"], "answer": d["answer"], "confidence": round(d["confidence"], 2)} for d in decisions]}


def run(flow: Flow, query: str, ask: Callable[[dict], dict]) -> Iterator[dict]:
    """Yield one event per step ({"event": "decision", ...}), then {"event": "end", "status", ...}.

    ask(request) returns a Jev-shaped response. A run ends with status "completed" (reached DONE), "no_transition",
    "max_steps" or "invalid"; the end event names the step where it stopped."""
    problems = check(flow)
    if problems["errors"]:
        yield {"event": "end", "status": "invalid", "step": None, "decisions": [], **problems}
        return
    decisions: list[dict] = []
    current = flow.start
    while current != DONE:
        if len(decisions) >= MAX_STEPS:
            yield {"event": "end", "status": "max_steps", "step": current, "decisions": decisions,
                   "message": f"stopped after {MAX_STEPS} steps; the flow has a loop without an exit"}
            return
        step = flow.steps[current]
        req = {"state": state_for(query, decisions), "questions": {step.id: question(step)}}
        t0 = time.perf_counter()
        a = ask(req)["answers"][step.id]
        answer, confidence, probs = outcome(step, a)
        taken = next((i for i, t in enumerate(step.transitions) if matches(t, answer, confidence)), None)
        d = {"step": step.id, "type": step.type, "question": step.instructions, "answer": answer,
             "confidence": confidence, "probabilities": probs, "raw": a, "state": req["state"],
             "transition": taken, "next": step.transitions[taken].target if taken is not None else None,
             "ms": round((time.perf_counter() - t0) * 1000)}
        decisions.append(d)
        yield {"event": "decision", **d}
        if taken is None:
            yield {"event": "end", "status": "no_transition", "step": step.id, "decisions": decisions,
                   "message": f"no arrow matches the answer {answer!r} at confidence {confidence:.2f}"}
            return
        current = d["next"]
    yield {"event": "end", "status": "completed", "step": None, "decisions": decisions,
           "path": [d["step"] for d in decisions]}
