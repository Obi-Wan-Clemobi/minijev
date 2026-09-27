"""The choice of the plan_cost predictor on development data (openspec/changes/add-plan-cost, tasks 2.1 and 2.2).

Development data is one frozen split, "dev". A **CV group** is one of 5 groups of whole sessions. To make them, sort
the session ids, shuffle them with random.Random(SEED), and deal them in turn into groups 0 to 4. Each turn gets a
held-out prediction from a predictor fitted on the other 4 groups. A predictor maps a row to P(long).

Each baseline maps a state to a category and predicts P(long) of that category on its training turns, with add-one
smoothing. Tercile cuts come from the training turns. The model is the zero-shot Noul readout, with a bias and
temperature fitted on the training turns.

The choice rules (design.md): the baseline with the lowest held-out log loss, after the tie guard; the model replaces
it only if the paired session-bootstrap interval of the log-loss difference is below zero; then the screen. If no
baseline passes the tie guard, the result is a no go. The chosen predictor is refitted on all development turns, and it
is stored with its threshold and a sha256.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Callable

from .dataset import ACCESS, Paths, load_split
from .turn_eval import KEYWORDS, SHORT_CHARS, nearest_rank, session_bootstrap, session_of

GROUPS, SEED = 5, 2026
FLAG_PERCENTILE = 0.8          # flag the top 20% of training scores
MAX_TRAIN_FLAGS = 0.30         # the tie guard
QUESTION_CHARS = 200           # plan_question looks at the end of the plan text
PLAN = "End of the previous answer:\n"
_KEYWORD = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b")
_STEP = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.M)


def parts(text: str) -> tuple[str | None, str]:
    """(plan text or None, user message) of a plan_cost state."""
    body = text.rsplit("\nPrevious turn: ", 1)[0]
    head, _, user = body.rpartition("User message:\n")
    return (head[len(PLAN):].rstrip("\n") if head.startswith(PLAN) else None), user


def is_keyword_reply(text: str) -> bool:
    msg = parts(text)[1].strip().lower()
    return len(msg) <= SHORT_CHARS and bool(_KEYWORD.search(msg))


def plan_steps(plan: str) -> int:
    return len(_STEP.findall(plan))


def terciles(values: list[float]) -> tuple[float, float]:
    return (nearest_rank(values, 1 / 3), nearest_rank(values, 2 / 3)) if values else (0.0, 0.0)


def bucket(v: float, cuts: tuple[float, float]) -> str:
    return "low" if v <= cuts[0] else "mid" if v <= cuts[1] else "high"


def categorizers(train: list[dict]) -> dict[str, tuple[Callable[[str], str], dict]]:
    """The six baselines (design.md), each a map from state to category, with the parameters it learned on train."""
    plans = [p for p in (parts(r["text"])[0] for r in train) if p is not None]
    length_cuts, step_cuts = terciles([len(p) for p in plans]), terciles([plan_steps(p) for p in plans])

    def on_plan(f):
        return lambda t: "none" if parts(t)[0] is None else f(parts(t)[0])
    keywords = lambda t: "keyword reply" if is_keyword_reply(t) else "other"
    steps = on_plan(lambda p: bucket(plan_steps(p), step_cuts))
    return {
        "prior": (lambda t: "all", {}),
        "keywords": (keywords, {}),
        "plan_length": (on_plan(lambda p: bucket(len(p), length_cuts)), {"cuts": length_cuts}),
        "plan_steps": (steps, {"cuts": step_cuts}),
        "plan_question": (on_plan(lambda p: "asks" if "?" in p[-QUESTION_CHARS:] else "no question"), {}),
        "keywords_steps": (lambda t: f"{keywords(t)} / {steps(t)}", {"cuts": step_cuts}),
    }


def fit(cat: Callable[[str], str], train: list[dict]) -> tuple[Callable[[dict], float], dict]:
    """P(long) per category, add-one smoothing; a category not seen on train gets the prior."""
    n, by = Counter(), defaultdict(Counter)
    for r in train:
        by[cat(r["text"])][r["y"]] += 1
        n[r["y"]] += 1
    prior = (n[1] + 1) / (len(train) + 2)
    table = {c: (k[1] + 1) / (sum(k.values()) + 2) for c, k in by.items()}
    return (lambda r: table.get(cat(r["text"]), prior)), {"table": table, "prior": prior}


def groups(rows: list[dict]) -> dict[str, int]:
    ids = sorted({session_of(r) for r in rows})
    random.Random(SEED).shuffle(ids)
    return {s: i % GROUPS for i, s in enumerate(ids)}


def log_loss(scored: list[dict]) -> float:
    return sum(-math.log(max(r["p"] if r["y"] else 1 - r["p"], 1e-12)) for r in scored) / len(scored)


def paired_delta(a: list[dict], b: list[dict]) -> dict:
    """Mean log loss of b minus a on the same rows, with a two-sided 95% session-bootstrap interval."""
    rows = [{"session": x["session"], "d": log_loss([y]) - log_loss([x])} for x, y in zip(a, b)]
    mean = lambda v: sum(r["d"] for r in v) / len(v)
    lo, _, hi = session_bootstrap(rows, mean)
    return {"mean": mean(rows), "ci95": [lo, hi]}


def precision(rows: list[dict]) -> float:
    """Precision of the flagged rows. A bootstrap resample without a flag counts 0.0: this lowers the bound (toward no go)."""
    flagged = [r for r in rows if r["flag"]]
    return sum(r["y"] for r in flagged) / len(flagged) if flagged else 0.0


def held_out(rows: list[dict], group: dict[str, int], make: Callable[[list[dict]], Callable[[dict], float]]) -> dict:
    """Held-out scores and flags for every row, and the training flag rate of each CV group fit."""
    scored, train_flag_rates = [], []
    for g in range(GROUPS):
        train = [r for r in rows if group[session_of(r)] != g]
        predict = make(train)
        train_scores = [predict(r) for r in train]
        cut = nearest_rank(train_scores, FLAG_PERCENTILE)
        train_flag_rates.append(sum(s >= cut for s in train_scores) / len(train_scores))
        for r in rows:
            if group[session_of(r)] == g:
                p = predict(r)
                scored.append({"id": r["id"], "session": session_of(r), "p": p, "y": r["y"], "flag": p >= cut})
    scored.sort(key=lambda r: r["id"])
    return {"scored": scored, "train_flag_rates": train_flag_rates}


def zero_shot_maker(logits: dict[str, list[float]]):
    """A factory: fit a bias and temperature on the training rows' readout logits, then predict P(yes)."""
    from ..calibrate import fit_bias_temperature, softmax_list

    def make(train: list[dict]) -> Callable[[dict], float]:
        temp, bias = fit_bias_temperature([logits[r["id"]] for r in train], [r["y"] for r in train])
        return lambda r: softmax_list([v / temp + b for v, b in zip(logits[r["id"]], bias)])[1]
    return make


def choose(version: str, paths: Paths | None = None, engine=None, readouts: bool = True) -> dict:
    """Tasks 2.1 and 2.2 on the dev split of a before-cutoff version."""
    paths = paths or Paths()
    rows = load_split("plan_cost", "dev", version, paths)
    group = groups(rows)
    base_rate = sum(r["y"] for r in rows) / len(rows)
    candidates = {}
    for name in categorizers(rows):
        def make(train, name=name):
            cat, _ = categorizers(train)[name]
            return fit(cat, train)[0]
        candidates[name] = held_out(rows, group, make)
    logits = {}
    if readouts:
        from .evaluate import load_engine, readout
        from .questions import QUESTIONS
        engine = engine or load_engine()
        q = QUESTIONS["plan_cost"]
        for r in rows:
            logits[r["id"]] = readout(engine, q, r["text"])
        candidates["zero_shot"] = held_out(rows, group, zero_shot_maker(logits))
    res = {"version": version, "rows": len(rows), "sessions": len(group), "long": sum(r["y"] for r in rows),
           "base_rate": base_rate, "groups": GROUPS, "seed": SEED, "predictors": {}}
    for name, c in candidates.items():
        res["predictors"][name] = {"log_loss": log_loss(c["scored"]), "train_flag_rates": c["train_flag_rates"],
                                   "tie_guard_ok": max(c["train_flag_rates"]) <= MAX_TRAIN_FLAGS,
                                   "held_out_flags": sum(r["flag"] for r in c["scored"]),
                                   "held_out_precision": precision(c["scored"])}
    # the refit on all development turns: the last fit that the tie guard checks
    refits = {}
    for name, (cat, params) in categorizers(rows).items():
        predict, table = fit(cat, rows)
        refits[name] = (predict, {"categorizer": name, **params, **table})
    if readouts:
        from ..calibrate import fit_bias_temperature, softmax_list
        temp, bias = fit_bias_temperature([logits[r["id"]] for r in rows], [r["y"] for r in rows])
        refits["zero_shot"] = (lambda r: softmax_list([v / temp + b for v, b in zip(logits[r["id"]], bias)])[1],
                               {"model": engine.name, "temperature": temp, "bias": bias})
    for name, (predict, _) in refits.items():
        scores = [predict(r) for r in rows]
        cut = nearest_rank(scores, FLAG_PERCENTILE)
        rate = sum(s >= cut for s in scores) / len(scores)
        m = res["predictors"][name]
        m |= {"refit_threshold": cut, "refit_flag_rate": rate}
        m["tie_guard_ok"] = m["tie_guard_ok"] and rate <= MAX_TRAIN_FLAGS
    baselines = sorted((n for n in res["predictors"] if n != "zero_shot"), key=lambda n: res["predictors"][n]["log_loss"])
    passing = [n for n in baselines if res["predictors"][n]["tie_guard_ok"]]
    chosen = passing[0] if passing else None
    if chosen and readouts:
        d = paired_delta(candidates[chosen]["scored"], candidates["zero_shot"]["scored"])
        res["predictors"]["zero_shot"]["minus_best_baseline"] = d
        if d["ci95"][1] < 0 and res["predictors"]["zero_shot"]["tie_guard_ok"]:
            chosen = "zero_shot"
    res["chosen"] = chosen
    if chosen is None:
        res["screen"] = {"passed": False, "reason": "no predictor passes the tie guard"}
    else:
        scored = candidates[chosen]["scored"]
        bound = session_bootstrap(scored, precision)[1]   # the one-sided 5% bound
        res["screen"] = {"precision": precision(scored), "precision_lower_5pct": bound, "base_rate": base_rate,
                         "flagged": sum(r["flag"] for r in scored), "long_flagged": sum(r["y"] for r in scored if r["flag"]),
                         "passed": bound > base_rate}
        if res["screen"]["passed"]:
            stored = {"version": version, "predictor": chosen, "threshold": res["predictors"][chosen]["refit_threshold"],
                      **refits[chosen][1]}
            blob = json.dumps(stored, sort_keys=True).encode()
            res["deployed"] = stored | {"sha256": hashlib.sha256(blob).hexdigest()}
    res["splits_accessed"] = list(ACCESS)
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"plan-choice-{version}{'' if readouts else '-counts'}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"plan_cost {version}: {len(rows)} turns, {res['long']} long, base rate {base_rate:.3f}")
    for name, m in sorted(res["predictors"].items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"  {name:15} held-out log loss {m['log_loss']:.4f}  flags {m['held_out_flags']:3}"
              f"  precision {m['held_out_precision']:.3f}  tie guard {'ok' if m['tie_guard_ok'] else 'FAILS'}"
              f" (max train flag rate {max(m['train_flag_rates'] + [m['refit_flag_rate']]):.2f})")
    if "minus_best_baseline" in res["predictors"].get("zero_shot", {}):
        d = res["predictors"]["zero_shot"]["minus_best_baseline"]
        print(f"  zero_shot minus best baseline: {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print(f"chosen: {chosen}; screen: {res['screen']}")
    print(f"wrote {path}")
    return res
