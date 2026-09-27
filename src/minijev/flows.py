"""Flows: chain minijev questions into a state machine (docs/superpowers/specs/2026-09-24-state-machine-flow-builder-design.md,
openspec/specs/flow-engine).

A flow is a set of steps. Each step asks one question (Noul, Choice or Score) about a request, and a **fan-out step**
also asks further independent questions in the same request, so one forward pass answers them all. The answers, and
optionally the confidence, pick the transition to the next step, until a transition reaches DONE, or ESCALATE (the
**hand-off**: the large model makes the decision). Every answer is added to the decisions list.

The state of every step is the request only, so the engine's state cache prefills the request once per run. Earlier
decisions go into the question block ("Decided so far: …"), so later questions still see what earlier ones decided.

A Choice step can take its options from an earlier step's answer (options_from): for example, the kind of work picks
which tools the next step offers.

How an answer picks a transition (the page explains the same rules):
- Noul: the answer is true when P(yes) >= 0.5. Confidence = 2 * max(p, 1 - p) - 1: 0 at p = 0.5, 1 at p = 0 or 1
  (the Choice formula with two options).
- Choice: the answer is the most likely option. Confidence = (p_max - 1/k) / (1 - 1/k), as in the Jev response. The
  decision also lists every option, ranked by probability.
- Score: the answer is the expected level rounded to the nearest level (0 to n-1). Confidence is the ordinal one.
Transitions of a step are checked in order; the first one whose answer (or "any") and condition match wins. A
transition tests the step's own question, or with from_question one of its fan-out questions.
"""

from __future__ import annotations

import math
from collections import defaultdict
import time
from typing import Callable, Iterator, Literal

from pydantic import BaseModel, Field

DONE, ESCALATE = "DONE", "ESCALATE"
TERMINALS = (DONE, ESCALATE)
MAX_STEPS = 20
MAX_OPTIONS = 26   # one label letter per option


class Condition(BaseModel):
    confidence_gte: float | None = None
    confidence_lt: float | None = None


class Transition(BaseModel):
    from_answer: str | bool | int | None = Field(None, description="None matches any answer (a fallback)")
    from_question: str | None = Field(None, description="a fan-out question id; None tests the step's own question")
    target: str
    condition: Condition | None = None


class SubQuestion(BaseModel):
    """A fan-out question: asked in the same request as its step's own question."""
    id: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict | list | None = None


def answer_key(a) -> str:
    """The key of an answer in an options map: JSON spelling, so the page (JavaScript String()) and Python agree."""
    return ("true" if a else "false") if isinstance(a, bool) else str(a)


class OptionsFrom(BaseModel):
    """A Choice step's options, from an earlier step's answer: map[answer_key(answer)] is the options of that answer."""
    step: str
    map: dict[str, dict]


class Step(BaseModel):
    id: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict | list | None = None
    options_from: OptionsFrom | None = None
    fanout: list[SubQuestion] = Field(default_factory=list)
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    transitions: list[Transition] = Field(default_factory=list)


class Example(BaseModel):
    """A sample request, and the answer that a step should give to it. A step that the run does not reach is not
    checked. Examples show on the page and let a test catch a flow that gives every request the same answers."""
    query: str
    expect: dict[str, str | bool | int] = Field(default_factory=dict)


class Flow(BaseModel):
    id: str
    name: str
    description: str = ""
    version: str = "1.0"
    start: str
    steps: dict[str, Step]
    examples: list[Example] = Field(default_factory=list)


def decided_so_far(decisions: list[dict]) -> str:
    """The earlier decisions, as the first line of a question block."""
    if not decisions:
        return ""
    parts = [f"{d['question']} → {d['answer']} ({d['confidence']:.2f})" for d in decisions]
    return "Decided so far: " + "; ".join(parts) + "\n"


def question(step: Step | SubQuestion, criteria: dict | list | None = None, decisions: list[dict] | None = None) -> dict:
    q = {"type": step.type, "instructions": decided_so_far(decisions or []) + step.instructions}
    criteria = criteria if criteria is not None else step.criteria
    if criteria is not None:
        q["criteria"] = criteria
    return q


def outcome(step: Step | SubQuestion, a: dict) -> tuple[str | bool | int, float, dict | None]:
    """(answer, confidence, probabilities) of one Jev-shaped answer, by the rules in the module docstring."""
    if step.type == "noul":
        p = a["noul"]
        return p >= 0.5, 2 * max(p, 1 - p) - 1, {"yes": p, "no": 1 - p}
    if step.type == "choice":
        return a["choice"], a["confidence"], a["probabilities"]
    return int(math.floor(a["score"] + 0.5)), a["confidence"], a["probabilities"]


def ranked(probabilities: dict | None) -> list[list] | None:
    """Every option with its probability, highest first."""
    return sorted(([k, v] for k, v in probabilities.items()), key=lambda kv: -kv[1]) if probabilities else None


def matches(t: Transition, answer, confidence: float) -> bool:
    if t.from_answer is not None and t.from_answer != answer:
        return False
    c = t.condition
    if c and c.confidence_gte is not None and confidence < c.confidence_gte:
        return False
    if c and c.confidence_lt is not None and confidence >= c.confidence_lt:
        return False
    return True


def options_of(step: Step | SubQuestion) -> dict | list | None:
    """The options a step can offer: its criteria, or for options_from the union of the mapped options."""
    if isinstance(step, Step) and step.options_from is not None:
        out: dict = {}
        for opts in step.options_from.map.values():
            out.update(opts)
        return out
    return step.criteria


def answers_of(step: Step | SubQuestion) -> list:
    """Every answer a step can give: True/False, the option keys, or the level numbers."""
    if step.type == "noul":
        return [True, False]
    if step.type == "choice":
        return list(options_of(step) or {})
    return list(range(len(step.criteria or [])))


def tested(step: Step, t: Transition) -> Step | SubQuestion | None:
    """The question a transition tests: the step's own, or one of its fan-out questions (None if unknown)."""
    if t.from_question is None:
        return step
    return next((q for q in step.fanout if q.id == t.from_question), None)


def reachable_without(flow: Flow, target: str, avoid: str) -> bool:
    """True if some path from the start reaches target without passing through avoid."""
    seen, todo = set(), [flow.start] if flow.start in flow.steps and flow.start != avoid else []
    while todo:
        sid = todo.pop()
        if sid == target:
            return True
        if sid in seen:
            continue
        seen.add(sid)
        todo += [t.target for t in flow.steps[sid].transitions if t.target in flow.steps and t.target != avoid]
    return False


def check(flow: Flow) -> dict:
    """Errors stop a run; warnings do not. Returns {"errors": [...], "warnings": [...]}, each {"step", "message"}."""
    errors, warnings = [], []
    if flow.start not in flow.steps:
        errors.append({"step": None, "message": f"the start step {flow.start!r} does not exist"})
    for sid, s in flow.steps.items():
        if s.id != sid:
            errors.append({"step": sid, "message": f"the key {sid!r} and the id {s.id!r} differ"})
        if sid in TERMINALS:
            errors.append({"step": sid, "message": f"{sid!r} is a terminal name and cannot be a step"})
        if not s.instructions.strip():
            errors.append({"step": sid, "message": "the question is empty"})
        if s.options_from is not None:
            src = flow.steps.get(s.options_from.step)
            if s.type != "choice":
                errors.append({"step": sid, "message": "only a Choice can take its options from an earlier answer"})
            if src is None:
                errors.append({"step": sid, "message": f"options come from {s.options_from.step!r}, which does not exist"})
            elif src.id == sid:
                errors.append({"step": sid, "message": "options come from this step itself, which has not answered yet"})
            else:
                missing = [a for a in answers_of(src) if answer_key(a) not in s.options_from.map]
                if missing:
                    errors.append({"step": sid, "message": f"no options for the answer(s) {missing} of {src.id!r}"})
                if reachable_without(flow, sid, src.id):
                    errors.append({"step": sid, "message": f"a path reaches this step without {src.id!r}, "
                                   "whose answer gives its options"})
            if any(len(o) < 2 for o in s.options_from.map.values()):
                errors.append({"step": sid, "message": "every answer needs at least 2 options"})
        elif s.type == "choice" and (not isinstance(s.criteria, dict) or len(s.criteria) < 2):
            errors.append({"step": sid, "message": "a Choice needs at least 2 options"})
        if s.type == "score" and (not isinstance(s.criteria, list) or not 2 <= len(s.criteria) <= 10):
            errors.append({"step": sid, "message": "a Score needs 2 to 10 levels"})
        fanout_ids = [q.id for st in flow.steps.values() for q in st.fanout]
        for q in s.fanout:
            if q.id == sid or q.id in flow.steps or fanout_ids.count(q.id) > 1:
                errors.append({"step": sid, "message": f"the fan-out question id {q.id!r} is not unique"})
            if q.type == "choice" and (not isinstance(q.criteria, dict) or len(q.criteria) < 2):
                errors.append({"step": sid, "message": f"the fan-out Choice {q.id!r} needs at least 2 options"})
        if isinstance(options_of(s), dict) and len(options_of(s)) > MAX_OPTIONS:
            warnings.append({"step": sid, "message": f"more than {MAX_OPTIONS} options: only {MAX_OPTIONS} label letters"})
        for t in s.transitions:
            if t.target not in TERMINALS and t.target not in flow.steps:
                errors.append({"step": sid, "message": f"an arrow goes to {t.target!r}, which does not exist"})
            q = tested(s, t)
            if q is None:
                errors.append({"step": sid, "message": f"an arrow tests {t.from_question!r}, which is not a fan-out question here"})
            elif t.from_answer is not None and t.from_answer not in answers_of(q):
                errors.append({"step": sid, "message": f"an arrow starts at {t.from_answer!r}, which is not an answer"})
        own = [t for t in s.transitions if t.from_question is None]
        if not s.transitions:
            warnings.append({"step": sid, "message": "no arrow leaves this step: a run that reaches it stops"})
        elif own:   # coverage of the step's own answers, by its own-question arrows
            open_ = [a for a in answers_of(s) if not any(t.from_answer in (a, None) for t in own)]
            if open_:
                warnings.append({"step": sid, "message": f"no arrow for the answer(s) {open_}"})
            unsure = [a for a in answers_of(s) if a not in open_ and not any(
                t.from_answer in (a, None) and not (t.condition and (t.condition.confidence_gte is not None or t.condition.confidence_lt is not None))
                for t in own)]
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
    for i, ex in enumerate(flow.examples):
        for sid, want in ex.expect.items():
            if sid not in flow.steps:
                errors.append({"step": None, "message": f"example {i + 1} expects an answer from unknown step {sid!r}"})
            elif answer_key(want) not in map(answer_key, answers_of(flow.steps[sid])):
                errors.append({"step": sid, "message": f"example {i + 1} expects {want!r}, which is not an answer of {sid!r}"})
    return {"errors": errors, "warnings": warnings}


def try_examples(flow: Flow, ask: Callable[[dict], dict]) -> dict:
    """Run the flow on each example. Count the expected answers that the run reached (checked) and matched (hits). Also
    count the distinct answers of each step over all examples. A step with one distinct answer does not depend on the
    request."""
    out, answers = [], defaultdict(set)
    for ex in flow.examples:
        got = {e["step"]: e["answer"] for e in run(flow, ex.query, ask)
               if e["event"] == "decision" and e.get("question_id") in (None, e["step"])}
        for sid, a in got.items():
            answers[sid].add(answer_key(a))
        checked = [s for s in ex.expect if s in got]
        out.append({"query": ex.query, "got": got, "checked": len(checked),
                    "hits": sum(answer_key(got[s]) == answer_key(ex.expect[s]) for s in checked)})
    return {"examples": out, "checked": sum(e["checked"] for e in out), "hits": sum(e["hits"] for e in out),
            "distinct": {s: len(v) for s, v in answers.items()}}


def run(flow: Flow, query: str, ask: Callable[[dict], dict]) -> Iterator[dict]:
    """Yield one event per question ({"event": "decision", ...}), then {"event": "end", "status", ...}.

    ask(request) returns a Jev-shaped response. A run ends with status "completed" (reached DONE), "escalated"
    (reached ESCALATE: hand the decision to the large model), "no_transition", "max_steps" or "invalid"; the end
    event names the step where it stopped, and reports the input tokens and the number of requests (forward passes)."""
    cost = {"input_tokens": 0, "requests": 0}
    problems = check(flow)
    if problems["errors"]:
        yield {"event": "end", "status": "invalid", "step": None, "decisions": [], "usage": cost, **problems}
        return
    decisions: list[dict] = []
    current, steps = flow.start, 0
    while current not in TERMINALS:
        if steps >= MAX_STEPS:
            yield {"event": "end", "status": "max_steps", "step": current, "decisions": decisions, "usage": cost,
                   "message": f"stopped after {MAX_STEPS} steps; the flow has a loop without an exit"}
            return
        steps += 1
        step = flow.steps[current]
        criteria = None
        if step.options_from is not None:
            source = next((d for d in reversed(decisions) if d["question_id"] == step.options_from.step), None)
            if source is None:   # check() rejects such a flow; this guards a flow that bypassed check()
                yield {"event": "end", "status": "invalid", "step": step.id, "decisions": decisions, "usage": cost,
                       "message": f"{step.options_from.step!r} did not run, so {step.id!r} has no options"}
                return
            criteria = step.options_from.map[answer_key(source["answer"])]
        asked = [(step.id, step, criteria)] + [(q.id, q, None) for q in step.fanout]
        req = {"state": query, "questions": {qid: question(q, c, decisions) for qid, q, c in asked}}
        t0 = time.perf_counter()
        resp = ask(req)
        cost["requests"] += 1
        cost["input_tokens"] += (resp.get("usage") or {}).get("input_tokens", 0)
        ms = round((time.perf_counter() - t0) * 1000)
        results = {}
        for qid, q, _ in asked:
            answer, confidence, probs = outcome(q, resp["answers"][qid])
            results[qid] = (answer, confidence)
            d = {"step": step.id, "question_id": qid, "type": q.type, "question": q.instructions, "answer": answer,
                 "confidence": confidence, "probabilities": probs, "ranked": ranked(probs) if q.type == "choice" else None,
                 "raw": resp["answers"][qid], "state": req["state"], "ms": ms}
            decisions.append(d)
        taken = next((i for i, t in enumerate(step.transitions)
                      if matches(t, *results[t.from_question or step.id])), None)
        own = decisions[-len(asked)]   # the step's own question is asked first
        own["transition"] = taken
        own["next"] = step.transitions[taken].target if taken is not None else None
        for d in decisions[-len(asked):]:
            yield {"event": "decision", **d}
        if taken is None:
            tested_ids = [step.id] + [t.from_question for t in step.transitions if t.from_question and t.from_question != step.id]
            said = ", ".join(f"{qid} = {results[qid][0]!r} (sure {results[qid][1]:.2f})" for qid in dict.fromkeys(tested_ids))
            yield {"event": "end", "status": "no_transition", "step": step.id, "decisions": decisions, "usage": cost,
                   "message": f"no arrow matches: {said}"}
            return
        current = own["next"]
        if current == ESCALATE:
            t = step.transitions[taken]
            yield {"event": "end", "status": "escalated", "step": step.id, "decisions": decisions, "usage": cost,
                   "confidence": results[t.from_question or step.id][1],
                   "condition": t.condition.model_dump() if t.condition else None,
                   "path": [d["step"] for d in decisions if d["question_id"] == d["step"]]}
            return
    yield {"event": "end", "status": "completed", "step": None, "decisions": decisions, "usage": cost,
           "path": [d["step"] for d in decisions if d["question_id"] == d["step"]]}
