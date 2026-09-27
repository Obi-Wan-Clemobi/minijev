"""The turn-cost baselines on val (openspec/changes/add-turn-cost-routing, task 2.1). Train and val only.

Each baseline maps the state of a turn to a category. It predicts the level distribution of that category on train,
with add-one smoothing. A category not seen on train gets the prior:

- prior: one category for every turn.
- length: the user message length, in train terciles.
- keywords: a short reply (20 characters or fewer) that holds one of KEYWORDS as a whole word or phrase, or not.
- previous_turn: the level of the previous turn, or "none" at the session start.
- context: the context size, in train terciles.

Measures: log loss; ordinal accuracy (the share of rows whose most probable level is within one level of the true
level). Every interval comes from a bootstrap over sessions, because turns of one session depend on each other.

Routing: the score of a turn is P(31+). A turn is flagged when its score is at or above the threshold. The threshold is
the score of the train turn at the given train percentile, by nearest rank. Ties are flagged together. A split with
fewer than MIN_LONG long turns gives counts only, no recall or precision.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Callable

from .dataset import ACCESS, Paths, load_split, manifest
from .turns import LEVELS, previous_level

KEYWORDS = ("go", "yes", "ok", "continue", "sure", "do it", "implement", "apply", "fix all", "push", "deploy")
SHORT_CHARS = 20
N_BOOT, SEED = 2000, 2026
MIN_LONG = 10                      # below this many long turns, routing measures are counts only
PERCENTILES = (0.8, 0.9)           # flag the top 20% and the top 10% of train scores
LONG = len(LEVELS) - 1             # the level "31+"

_NEXT = re.compile(r"\n(?:Previous user message:\n|Earlier turns: )")
_KEYWORD = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b")


def user_message(text: str) -> str:
    body = text.split("User message:\n", 1)[1]
    m = _NEXT.search(body)
    return body[:m.start()] if m else body


def context_size(text: str) -> int:
    return int(re.search(r"\nContext size: (\d+) thousand tokens$", text).group(1))


def is_keyword_reply(text: str) -> bool:
    msg = user_message(text).strip().lower()
    return len(msg) <= SHORT_CHARS and bool(_KEYWORD.search(msg))


def nearest_rank(values: list[float], q: float) -> float:
    """The value at quantile q by nearest rank: always one of the values, never an interpolation."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    return s[max(math.ceil(q * len(s)), 1) - 1]


def terciles(values: list[float]) -> Callable[[float], str]:
    lo, hi = nearest_rank(values, 1 / 3), nearest_rank(values, 2 / 3)
    return lambda v: "low" if v <= lo else "mid" if v <= hi else "high"


def categorizers(train: list[dict]) -> dict[str, Callable[[str], str]]:
    """The baselines, each a map from state text to category. Tercile cuts come from train."""
    length = terciles([len(user_message(r["text"])) for r in train])
    context = terciles([context_size(r["text"]) for r in train])
    return {
        "prior": lambda t: "all",
        "length": lambda t: length(len(user_message(t))),
        "keywords": lambda t: "keyword reply" if is_keyword_reply(t) else "other",
        "previous_turn": previous_level,
        "context": lambda t: context(context_size(t)),
    }


def fit(cat: Callable[[str], str], train: list[dict]) -> Callable[[str], list[float]]:
    k = len(LEVELS)
    prior, by = Counter(r["y"] for r in train), defaultdict(Counter)
    for r in train:
        by[cat(r["text"])][r["y"]] += 1
    p_prior = [(prior[j] + 1) / (len(train) + k) for j in range(k)]
    table = {c: [(n[j] + 1) / (sum(n.values()) + k) for j in range(k)] for c, n in by.items()}
    return lambda t: table.get(cat(t), p_prior)


def session_of(row: dict) -> str:
    return row.get("session") or row["id"].rsplit(":t", 1)[0]


def session_bootstrap(rows: list[dict], stat: Callable[[list[dict]], float], n_boot: int = N_BOOT,
                      seed: int = SEED) -> list[float]:
    """Percentile values of stat over resamples of whole sessions (with replacement): 2.5%, 5% and 97.5%."""
    groups = defaultdict(list)
    for r in rows:
        groups[session_of(r)].append(r)
    keys, rng = sorted(groups), random.Random(seed)
    stats = sorted(stat([r for _ in keys for r in groups[keys[rng.randrange(len(keys))]]]) for _ in range(n_boot))
    return [stats[int(0.025 * n_boot)], stats[int(0.05 * n_boot)], stats[int(0.975 * n_boot) - 1]]


def log_loss(rows: list[dict]) -> float:
    return sum(-math.log(max(r["p"][r["y"]], 1e-12)) for r in rows) / len(rows)


def ordinal_accuracy(rows: list[dict]) -> float:
    return sum(abs(max(range(len(r["p"])), key=r["p"].__getitem__) - r["y"]) <= 1 for r in rows) / len(rows)


def routing(scored: list[dict], train_scores: list[float]) -> dict:
    """Flags on scored rows ({"p", "y"}) at each train percentile. Counts always; recall and precision only with
    MIN_LONG or more long turns."""
    long_n = sum(r["y"] == LONG for r in scored)
    out = {"long": long_n, "rows": len(scored), "spread": len(set(train_scores)) > 1}
    for q in PERCENTILES:
        cut = nearest_rank(train_scores, q)
        flagged = [r for r in scored if r["p"][LONG] >= cut]
        hits = sum(r["y"] == LONG for r in flagged)
        m = {"threshold": cut, "flagged": len(flagged), "flag_rate": len(flagged) / len(scored), "long_flagged": hits}
        if long_n >= MIN_LONG:
            m |= {"recall": hits / long_n, "precision": hits / len(flagged) if flagged else None}
        out[f"top_{round((1 - q) * 100)}"] = m
    return out


def evaluate(train: list[dict], val: list[dict]) -> dict:
    out = {}
    for name, cat in categorizers(train).items():
        predict = fit(cat, train)
        scored = [{"session": session_of(r), "p": predict(r["text"]), "y": r["y"]} for r in val]
        out[name] = {"log_loss": log_loss(scored), "log_loss_ci": session_bootstrap(scored, log_loss),
                     "ordinal_accuracy": ordinal_accuracy(scored),
                     "ordinal_accuracy_ci": session_bootstrap(scored, ordinal_accuracy),
                     "routing": routing(scored, [predict(r["text"])[LONG] for r in train])}
    return out


def baselines(version: str, fold: str | None = None, paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    if "turn_cost" not in manifest(version, paths)["questions"]:
        raise ValueError(f"{version} has no turn_cost")
    train, val = load_split("turn_cost", "train", version, paths, fold), load_split("turn_cost", "val", version, paths, fold)
    if not train or not val:
        raise ValueError(f"turn_cost {version} {fold or ''}: no train or val rows")
    res = {"version": version, "fold": fold, "split": "val", "n": len(val),
           "labels": dict(Counter(LEVELS[r["y"]] for r in val)), "baselines": evaluate(train, val),
           "splits_accessed": list(ACCESS)}
    print(f"turn_cost {version}{' fold ' + fold if fold else ''}  val n={len(val)}  {res['labels']}")
    for name, m in res["baselines"].items():
        r20 = m["routing"]["top_20"]
        print(f"  {name:14} log loss {m['log_loss']:.3f} [{m['log_loss_ci'][0]:.3f},{m['log_loss_ci'][2]:.3f}]"
              f"  ordinal acc {m['ordinal_accuracy']:.3f}"
              f"  top 20%: {r20['flagged']} flagged, {r20['long_flagged']} of {m['routing']['long']} long")
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"turn-baselines-{version}{'-' + fold if fold else ''}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")
    return res


def paired_session_delta(a: list[dict], b: list[dict]) -> dict:
    """Mean log loss of b minus a on the same rows, with a two-sided 95% interval from the session bootstrap."""
    rows = [{"session": x["session"], "d": -math.log(max(y["p"][y["y"]], 1e-12)) + math.log(max(x["p"][x["y"]], 1e-12))}
            for x, y in zip(a, b)]
    mean = lambda v: sum(r["d"] for r in v) / len(v)
    lo, _, hi = session_bootstrap(rows, mean)
    return {"mean": mean(rows), "ci95": [lo, hi]}


def zero_shot(version: str, fold: str | None = None, paths: Paths | None = None, engine=None) -> dict:
    """Task 2.2 and 2.3: the zero-shot readout, with a bias and temperature fitted on train (as the baselines are
    counted on train, so that val compares them fairly), and the choice of the deployed predictor by the fixed rule."""
    from ..calibrate import fit_bias_temperature, softmax_list
    from .evaluate import load_engine, readout
    from .questions import QUESTIONS
    paths, q = paths or Paths(), QUESTIONS["turn_cost"]
    train, val = load_split("turn_cost", "train", version, paths, fold), load_split("turn_cost", "val", version, paths, fold)
    engine = engine or load_engine()
    z_train, z_val = [readout(engine, q, r["text"]) for r in train], [readout(engine, q, r["text"]) for r in val]
    temp, bias = fit_bias_temperature(z_train, [r["y"] for r in train])
    score = lambda probs: [{"session": session_of(r), "p": p, "y": r["y"]} for r, p in zip(val, probs)]
    models = {"zero_shot": (score([softmax_list(z) for z in z_val]), [softmax_list(z)[LONG] for z in z_train]),
              "zero_shot_bias_temp": (score([softmax_list([v / temp + b for v, b in zip(z, bias)]) for z in z_val]),
                                      [softmax_list([v / temp + b for v, b in zip(z, bias)])[LONG] for z in z_train])}
    base = {}
    for name, cat in categorizers(train).items():
        predict = fit(cat, train)
        base[name] = (score([predict(r["text"]) for r in val]), [predict(r["text"])[LONG] for r in train])
    best = min(base, key=lambda n: log_loss(base[n][0]))
    res = {"version": version, "fold": fold, "split": "val", "model": engine.name, "temperature": temp, "bias": bias,
           "best_baseline": best, "best_baseline_log_loss": log_loss(base[best][0]), "models": {}}
    for name, (scored, train_scores) in models.items():
        res["models"][name] = {"log_loss": log_loss(scored), "log_loss_ci": session_bootstrap(scored, log_loss),
                               "ordinal_accuracy": ordinal_accuracy(scored),
                               "ordinal_accuracy_ci": session_bootstrap(scored, ordinal_accuracy),
                               "routing": routing(scored, train_scores),
                               "minus_best_baseline": paired_session_delta(base[best][0], scored)}
    beats = [n for n, m in res["models"].items() if m["minus_best_baseline"]["ci95"][1] < 0]
    chosen = min(beats, key=lambda n: res["models"][n]["log_loss"]) if beats else best
    spread = (res["models"][chosen]["routing"] if beats else routing(*base[best]))["spread"]
    res["deployed"] = {"predictor": chosen, "model_beats_baseline": bool(beats), "spread": spread}
    res["splits_accessed"] = list(ACCESS)
    print(f"turn_cost {version}{' fold ' + fold if fold else ''}  best baseline {best} {res['best_baseline_log_loss']:.3f}")
    for name, m in res["models"].items():
        d = m["minus_best_baseline"]
        print(f"  {name:20} log loss {m['log_loss']:.3f}  ordinal acc {m['ordinal_accuracy']:.3f}"
              f"  minus {best}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    print(f"  deployed predictor: {chosen}{'' if spread else ' (no spread: no go)'}")
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"turn-zero-shot-{version}{'-' + fold if fold else ''}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")
    return res
