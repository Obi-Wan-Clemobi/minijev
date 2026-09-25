"""Questions to branches, one forward pass, and the Jev contract: ask()."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from .engine import Branch, Engine, groups
from .primitives import answer, class_logits, rounded, softmax
from .prompt import LETTERS, choice_block, noul_block, proposal_block, render, render_inline, score_listwise_block
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
    paired = [q["opposite_of"] for q in req["questions"].values() if q.get("opposite_of")]
    for qid, q in req["questions"].items():
        other = q.get("opposite_of")
        if other:
            assert q["type"] == "noul" and req["questions"].get(other, {}).get("type") == "noul", \
                f"{qid}: opposite_of must name another noul in the same request"
            assert other != qid and paired.count(other) == 1 and qid not in paired \
                and not req["questions"][other].get("opposite_of"), f"{qid}: a question can be in one opposite pair only"


def consistent(p_x: float, p_not_x: float) -> tuple[float, float]:
    """Make an opposite pair sum to 1 (W6, PLAN Task 4.5). The two answers are two estimates of the same log-odds,
    logit P(X) and -logit P(not X); their mean is the combined estimate. When the pair already sums to 1, nothing
    changes."""
    eps = 1e-12
    lx = math.log(max(p_x, eps) / max(1 - p_x, eps))
    lnx = math.log(max(p_not_x, eps) / max(1 - p_not_x, eps))
    z = (lx - lnx) / 2
    p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return p, 1 - p


def share_head(engine: Engine, branches: list[Branch], blocks: list[str], head_text: str) -> list[Branch]:
    """The two-level tree: move the text that every block starts with (the question) into a shared head, but only
    when the tokens stay exactly the same as the joined text, so that the model reads the same tokens either way."""
    head = engine.tok.encode(head_text, add_special_tokens=False)
    children = [engine.suffix_ids(block[len(head_text):]) for block in blocks]
    if not all(block.startswith(head_text) for block in blocks) or any(
            head + c != b.ids for c, b in zip(children, branches)):
        return branches  # a token merges across the split: keep the flat layout
    return [replace(b, ids=c, head=tuple(head)) for b, c in zip(branches, children)]


def branches_for(engine: Engine, qid: str, q: dict, share_question: bool = False) -> list[Branch]:
    """The branches of one question. share_question: pointwise items and averaged rotations share the question text
    in one head (state -> question -> item), when the tokens allow it (share_head)."""
    flat = _branches(engine, qid, q)
    if not share_question or len(flat) < 2:
        return flat
    question = f"QUESTION: {render(q['instructions'])}\n"
    if q["type"] == "choice" and q.get("choice_mode") == "averaged":
        keys = list(q["criteria"])
        blocks = [choice_block({**q, "criteria": {keys[i]: q["criteria"][keys[i]] for i in b.order}}) for b in flat]
        return share_head(engine, flat, blocks, question + "OPTIONS:\n")
    if flat[0].pointwise:
        items = ([f"{key}: {render_inline(d)}" if d is not None else key for key, d in q["criteria"].items()]
                 if q["type"] == "choice" else score_items(q))
        return share_head(engine, flat, [proposal_block(q, it) for it in items], question)
    return flat


def _branches(engine: Engine, qid: str, q: dict) -> list[Branch]:
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
    return [Branch(engine.suffix_ids(proposal_block(q, item)), [engine.yes, engine.no], qid, pointwise=True)
            for item in score_items(q)]


def score_items(q: dict) -> list[str]:
    """The proposed answer of each pointwise Score level. contrastive: each level also names its neighbours as what
    it is not, "frustrated (not mildly annoyed; not angry)", because a small model says yes to any plausible level
    when it sees one level alone (W5, PLAN Task 4.4)."""
    levels = [render_inline(level) for level in q["criteria"]]
    if not q.get("contrastive"):
        return levels
    return [f"{level} (" + "; ".join(f"not {levels[j]}" for j in (i - 1, i + 1) if 0 <= j < len(levels)) + ")"
            for i, level in enumerate(levels)]


def raw_scores(engine: Engine, req: dict, mode: str = "packed", share_question: bool = False) -> tuple[dict, dict]:
    """Per question: the uncalibrated logits the answer is built from, plus label mass and usage."""
    validate(req)
    prefix = engine.prefix_ids(req["state"])
    branches = [b for qid, q in req["questions"].items() for b in branches_for(engine, qid, q, share_question)]
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
        "input_tokens": len(prefix) + sum(len(b.ids) for b in branches) + sum(len(h) for h, _ in groups(branches)),
        "output_tokens": 0,
        "latency_ms": round(elapsed * 1000),  # minijev extension; not in Jev's usage object
    }
    return raw, usage


def with_modes(q: dict, s: Settings) -> dict:
    """Fill in the readout mode from the settings when the question does not set it."""
    if q["type"] == "choice":
        return {"choice_mode": s.resolved_choice_mode(), **q}
    if q["type"] == "score":
        return {"score_mode": s.score_mode, **({"contrastive": True} if s.score_contrastive else {}), **q}
    return q


def ask(engine: Engine, req: dict, mode: str = "packed", settings: Settings | None = None, debug: bool = False) -> dict:
    """The Jev contract: {state, questions} in, {model, answers, usage} out.

    settings defaults to Settings.load(): minijev.env plus MINIJEV_* environment variables.
    """
    s = settings or Settings.load()
    req = {**req, "questions": {qid: with_modes(q, s) for qid, q in req["questions"].items()}}
    raw, usage = raw_scores(engine, req, mode, s.share_question)
    answers, warnings, applied = {}, [], {}
    for qid, q in req["questions"].items():
        readout_mode = q.get("choice_mode") if q["type"] == "choice" else None
        t, b, source = s.calibrator(q["type"], readout_mode)
        applied[qid] = {"temperature": t, "bias": b, "source": source}
        answers[qid] = answer(q, raw[qid]["logits"], t, b)
        low = min(raw[qid]["mass"])
        if low < s.min_label_mass:  # the model wanted to say something other than a label: prompt problem
            warnings.append(f"{qid}: only {low:.2f} of next-token mass is on the labels")
    pairs = {}
    for qid, q in req["questions"].items():  # opposite pairs: "opposite_of" names the question it negates
        if q.get("opposite_of"):
            other = q["opposite_of"]
            before = answers[other]["noul"] + answers[qid]["noul"]
            p, p_not = consistent(answers[other]["noul"], answers[qid]["noul"])
            answers[other] = {**answers[other], "noul": p}
            answers[qid] = {**answers[qid], "noul": p_not}
            pairs[qid] = {"opposite_of": other, "sum_before": before}
    if not debug:
        answers = {qid: rounded(a) for qid, a in answers.items()}
    resp = {"model": f"minijev-poc ({engine.name})", "answers": answers, "usage": usage}
    if debug:
        prov = s.fitted.get("provenance") if s.fitted else None
        resp["debug"] = {"raw": raw, "warnings": warnings, "calibration": applied, "calibration_provenance": prov,
                         "opposite_pairs": pairs}
    return resp
