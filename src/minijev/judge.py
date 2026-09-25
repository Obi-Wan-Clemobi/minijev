"""Questions to branches, one forward pass, and the Jev contract: ask()."""

from __future__ import annotations

import math
import time

from .engine import Branch, Engine
from .primitives import answer, class_logits, rounded, softmax
from .prompt import LETTERS, choice_block, noul_block, proposal_block, render_inline, score_listwise_block
from .settings import Settings


def validate(req: dict) -> None:
    for qid, q in req["questions"].items():
        t = q.get("type")
        if t == "choice":
            n = len(q["criteria"])
            assert 2 <= n <= len(LETTERS), f"{qid}: choice needs 2..{len(LETTERS)} options in this POC (Jev: 255)"
        elif t == "score":
            assert 2 <= len(q["criteria"]) <= 10, f"{qid}: score needs 2..10 levels"
        elif t != "noul":
            raise ValueError(f"{qid}: unknown type {t!r}")


def branches_for(engine: Engine, qid: str, q: dict) -> list[Branch]:
    if q["type"] == "noul":
        return [Branch(engine.suffix_ids(noul_block(q)), [engine.yes, engine.no], qid)]
    k = len(q["criteria"])
    if q["type"] == "choice":
        if q.get("choice_mode", "listwise") == "averaged":  # every option at every letter position once
            keys = list(q["criteria"])
            out = []
            for r in range(k):
                order = [(r + j) % k for j in range(k)]
                shown = {**q, "criteria": {keys[i]: q["criteria"][keys[i]] for i in order}}
                out.append(Branch(engine.suffix_ids(choice_block(shown)), engine.letters[:k], qid, order=order))
            return out
        if q.get("choice_mode", "listwise") == "pointwise":  # Jev's stage 1 for large option sets
            return [
                Branch(engine.suffix_ids(proposal_block(q, f"{key}: {render_inline(d)}" if d is not None else key)),
                       [engine.yes, engine.no], qid, pointwise=True)
                for key, d in q["criteria"].items()
            ]
        return [Branch(engine.suffix_ids(choice_block(q)), engine.letters[:k], qid)]
    if q.get("score_mode", "pointwise") == "listwise":  # ablation; Jev judges levels separately
        return [Branch(engine.suffix_ids(score_listwise_block(q)), engine.letters[:k], qid)]
    return [
        Branch(engine.suffix_ids(proposal_block(q, render_inline(level))), [engine.yes, engine.no], qid, pointwise=True)
        for level in q["criteria"]
    ]


def raw_scores(engine: Engine, req: dict, mode: str = "packed") -> tuple[dict, dict]:
    """Per question: the uncalibrated logits the answer is built from, plus label mass and usage."""
    validate(req)
    prefix = engine.prefix_ids(req["state"])
    branches = [b for qid, q in req["questions"].items() for b in branches_for(engine, qid, q)]
    t0 = time.perf_counter()
    lp = engine.readouts(prefix, branches, mode)
    elapsed = time.perf_counter() - t0
    raw: dict = {qid: {"logits": [], "mass": []} for qid in req["questions"]}
    averaged: dict = {}  # question -> summed option probabilities over the rotations
    for b, row in zip(branches, lp):
        z, mass = class_logits(row, b.classes)
        if b.pointwise:
            raw[b.question]["logits"].append(z[0] - z[1])  # this level's (or option's) yes/no log-odds
        elif b.order is not None:  # one rotation: letter position j showed option b.order[j]
            p = softmax(z)
            acc = averaged.setdefault(b.question, [0.0] * len(z))
            for j, i in enumerate(b.order):
                acc[i] += p[j] / len(z)
        else:
            raw[b.question]["logits"] = z
        raw[b.question]["mass"].append(mass)
    for qid, p in averaged.items():  # log of the mean probability: softmax(logits) gives the average back
        raw[qid]["logits"] = [math.log(max(v, 1e-12)) for v in p]
    usage = {
        "input_tokens": len(prefix) + sum(len(b.ids) for b in branches),
        "output_tokens": 0,
        "latency_ms": round(elapsed * 1000),  # minijev extension; not in Jev's usage object
    }
    return raw, usage


def with_modes(q: dict, s: Settings) -> dict:
    """Fill in the readout mode from the settings when the question does not set it."""
    if q["type"] == "choice":
        return {"choice_mode": s.resolved_choice_mode(), **q}
    if q["type"] == "score":
        return {"score_mode": s.score_mode, **q}
    return q


def ask(engine: Engine, req: dict, mode: str = "packed", settings: Settings | None = None, debug: bool = False) -> dict:
    """The Jev contract: {state, questions} in, {model, answers, usage} out.

    settings defaults to Settings.load(): minijev.env plus MINIJEV_* environment variables.
    """
    s = settings or Settings.load()
    req = {**req, "questions": {qid: with_modes(q, s) for qid, q in req["questions"].items()}}
    raw, usage = raw_scores(engine, req, mode)
    answers, warnings, applied = {}, [], {}
    for qid, q in req["questions"].items():
        readout_mode = q.get("choice_mode") if q["type"] == "choice" else None
        t, b, source = s.calibrator(q["type"], readout_mode)
        applied[qid] = {"temperature": t, "bias": b, "source": source}
        a = answer(q, raw[qid]["logits"], t, b)
        answers[qid] = a if debug else rounded(a)
        low = min(raw[qid]["mass"])
        if low < s.min_label_mass:  # the model wanted to say something other than a label: prompt problem
            warnings.append(f"{qid}: only {low:.2f} of next-token mass is on the labels")
    resp = {"model": f"minijev-poc ({engine.name})", "answers": answers, "usage": usage}
    if debug:
        prov = s.fitted.get("provenance") if s.fitted else None
        resp["debug"] = {"raw": raw, "warnings": warnings, "calibration": applied, "calibration_provenance": prov}
    return resp
