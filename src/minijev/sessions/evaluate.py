"""Baselines for the session questions, on the val split of a frozen version. Torch loads only here.

- prior: the train class frequencies (add-one smoothing).
- previous_call: P(label | question.condition(state)), counted on train. For the built-in questions the condition is the
  previous call in the turn. Calls in one turn are strongly correlated, so this is the real bar to beat.
- zero_shot: the minijev readout of the base model with the question's prompt.
- zero_shot_bias_temp: the same readout with a bias and temperature fitted on FIT_ROWS train rows. It changes no
  weight, so every fine-tune gain must beat it (the val-fitted bias control of E22).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..calibrate import bootstrap_ci, fit_bias_temperature, multiclass_metrics, softmax_list
from .dataset import ACCESS, Paths, load_split, manifest
from .questions import QUESTIONS, Question

FIT_ROWS, SEED = 600, 2026
MIN_TRAIN_ROWS, TARGET_ACCURACY = 20, 0.9   # a class with fewer train rows is "unsupported"; hand-off target on val
MIN_KEPT = 30   # a hand-off threshold must keep at least this many val rows, or its accuracy is too noisy to use


def counted(q: Question, train: list[dict]) -> tuple[list[float], dict]:
    k = len(q.labels)
    prior = Counter(r["y"] for r in train)
    by = defaultdict(Counter)
    if q.condition:
        for r in train:
            by[q.condition(r["text"])][r["y"]] += 1
    p_prior = [(prior[j] + 1) / (len(train) + k) for j in range(k)]
    return p_prior, {c: [(n[j] + 1) / (sum(n.values()) + k) for j in range(k)] for c, n in by.items()}


def prompt_ids(engine, q: Question, text: str) -> tuple[list[int], list[list[int]]]:
    """(token ids of state + question block, label token variants per class). Training (poc/train_lora.py) and the
    readout use this one function, so they see the same prompt."""
    from ..prompt import choice_block, noul_block
    if q.type == "noul":
        block, classes = noul_block({"type": "noul", **q.prompt}), [engine.no, engine.yes]
    else:
        block, classes = choice_block({"type": "choice", **q.prompt}), engine.letters[:len(q.prompt["criteria"])]
    return engine.prefix_ids(text) + engine.suffix_ids(block), classes


def readout(engine, q: Question, text: str) -> list[float]:
    """Class logits (log-probabilities, logsumexp over label variants) at the last prompt token."""
    import torch
    ids, classes = prompt_ids(engine, q, text)
    with torch.no_grad():
        lp = torch.log_softmax(engine.model(torch.tensor([ids]), logits_to_keep=1).logits[0, -1].float(), -1)
    return [torch.logsumexp(lp[c], 0).item() for c in classes]


def report(probs: list[list[float]], labels: list[int]) -> dict:
    nll = [-math.log(max(p[y], 1e-12)) for p, y in zip(probs, labels)]
    pairs = list(zip(probs, labels))
    return {"log_loss": sum(nll) / len(nll), "log_loss_ci95": bootstrap_ci(nll, lambda v: sum(v) / len(v)),
            **multiclass_metrics(probs, labels),
            "accuracy_ci95": bootstrap_ci(pairs, lambda v: multiclass_metrics(*zip(*v))["accuracy"])}


def confidence(p: list[float]) -> float:
    """The confidence of a decision: (p_max - 1/k) / (1 - 1/k), 0 for a uniform guess, 1 for certainty (as in flows)."""
    k = len(p)
    return (max(p) - 1 / k) / (1 - 1 / k)


def risk_coverage(probs: list[list[float]], labels: list[int], steps: int = 20) -> list[dict]:
    """For thresholds 0, 1/steps, …, 1: the share of decisions kept (confidence >= threshold), and the accuracy on
    them. A decision below the threshold is handed off to the large model."""
    rows = [(confidence(p), max(range(len(p)), key=p.__getitem__) == y) for p, y in zip(probs, labels)]
    out = []
    for i in range(steps + 1):
        t = i / steps
        kept = [ok for c, ok in rows if c >= t]
        out.append({"threshold": t, "kept": len(kept), "coverage": len(kept) / len(rows),
                    "accuracy": sum(kept) / len(kept) if kept else None})
    return out


def choose_threshold(curve: list[dict], target: float = TARGET_ACCURACY, min_kept: int = MIN_KEPT) -> dict | None:
    """The lowest threshold whose accuracy on kept decisions reaches target (so the most decisions are kept), among the
    thresholds that keep at least min_kept rows."""
    return next((c for c in curve if c["kept"] >= min_kept and c["accuracy"] >= target), None)


def load_engine(model: str | None = None):
    from ..engine import Engine
    return Engine(model) if model else Engine()


def baselines(version: str, names: list[str], model: str | None = None, paths: Paths | None = None,
              readouts: bool = True, fold: str | None = None, engine=None) -> dict:
    """The baselines on val of one version (and one fold, for a leave-one-project-out version). Train and val only."""
    paths = paths or Paths()
    frozen = manifest(version, paths)["questions"]
    out = {"version": version, "fold": fold, "split": "val", "fit_rows": FIT_ROWS, "questions": {}}
    engine = (engine or load_engine(model)) if readouts else None
    if engine is not None:
        out["model"] = engine.name
    for name in names:
        if name not in frozen:
            print(f"{name}: not in {version}, skipped")
            continue
        q = QUESTIONS[name]
        train, val = load_split(name, "train", version, paths, fold), load_split(name, "val", version, paths, fold)
        if not train or not val:
            print(f"{name}: no train or val rows, skipped")
            continue
        labels = [r["y"] for r in val]
        p_prior, p_by = counted(q, train)
        train_counts = Counter(q.labels[r["y"]] for r in train)
        res = {"n": len(val), "labels": dict(Counter(q.labels[y] for y in labels)), "train_labels": dict(train_counts),
               "unsupported": [k for k in q.labels if train_counts[k] < MIN_TRAIN_ROWS],
               "prior": report([p_prior] * len(val), labels)}
        if q.condition:
            probs = [p_by.get(q.condition(r["text"]), p_prior) for r in val]
            curve = risk_coverage(probs, labels)
            res["previous_call"] = report(probs, labels) | {"risk_coverage": curve,
                                                            "hand_off": choose_threshold(curve)}
        if engine is not None:
            t0 = time.time()
            z_val = [readout(engine, q, r["text"]) for r in val]
            fit = random.Random(SEED).sample(train, min(FIT_ROWS, len(train)))
            z_fit = [readout(engine, q, r["text"]) for r in fit]
            temp, bias = fit_bias_temperature(z_fit, [r["y"] for r in fit])
            res["zero_shot"] = report([softmax_list(z) for z in z_val], labels)
            probs = [softmax_list([v / temp + b for v, b in zip(z, bias)]) for z in z_val]
            curve = risk_coverage(probs, labels)
            res["zero_shot_bias_temp"] = {**report(probs, labels), "temperature": temp, "bias": bias,
                                          "risk_coverage": curve, "hand_off": choose_threshold(curve)}
            res["readout_seconds"] = time.time() - t0
        out["questions"][name] = res
        print(f"\n{name}{' fold ' + fold if fold else ''}  n={len(val)}  {res['labels']}"
              + (f"  unsupported: {res['unsupported']}" if res["unsupported"] else ""))
        for key in ("prior", "previous_call", "zero_shot", "zero_shot_bias_temp"):
            if key in res:
                m = res[key]
                print(f"  {key:20} log loss {m['log_loss']:.3f} [{m['log_loss_ci95'][0]:.3f},{m['log_loss_ci95'][1]:.3f}]"
                      f"  acc {m['accuracy']:.3f}  ece {m['ece']:.3f}"
                      + (f"  hand-off at {m['hand_off']['threshold']:.2f}: keeps {m['hand_off']['coverage']:.0%}"
                         if m.get("hand_off") else ""))
    out["splits_accessed"] = list(ACCESS)
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"baselines-{version}{'-' + fold if fold else ''}-{out.get('model', 'counts').split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    return out


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Mean of b - a over the same rows, with a 95% bootstrap interval over rows."""
    d = [y - x for x, y in zip(a, b)]
    return {"mean": sum(d) / len(d), "ci95": bootstrap_ci(d, lambda v: sum(v) / len(v))}


def ladder_groups(kind_probs: list[list[float]], texts: list[str], tool_models: dict) -> list[list[float]]:
    """Tool-group probabilities from the ladder: P(group) = sum over kinds of P(kind) * P(tool | kind), over the tools
    of that group. P(tool | kind) is the previous-call count model of tool_<kind>; a kind with one tool gives that
    tool probability 1. The groups are those of next_tool, so the ladder and the flat question share one label space."""
    from .logs import tool_group
    from .patterns import SINGLE_TOOL, TOOLS_BY_KIND, WORK_KINDS
    from .questions import NEXT_TOOL
    group_of = lambda t: tool_group({"browser": "mcp__claude-in-chrome__x", "mcp": "mcp__x__y"}.get(t, t))
    out = []
    for pk, text in zip(kind_probs, texts):
        g = [0.0] * len(NEXT_TOOL)
        for k, p in zip(WORK_KINDS, pk):
            if k in SINGLE_TOOL:
                g[NEXT_TOOL.index(group_of(SINGLE_TOOL[k]))] += p
                continue
            q, (p_prior, p_by) = tool_models[k]
            for t, pt in zip(TOOLS_BY_KIND[k], p_by.get(q.condition(text), p_prior)):
                g[NEXT_TOOL.index(group_of(t))] += p * pt
        out.append(g)
    return out


def check_adapter(adapter: str, version: str, fold: str | None, name: str) -> None:
    """Stop unless the adapter's train_log.json says it was trained on this version, fold and question."""
    log_path = Path(adapter).parent / "train_log.json"
    if not log_path.exists():
        raise SystemExit(f"{log_path} is missing: cannot confirm what {adapter} was trained on")
    log = json.loads(log_path.read_text())
    want = {"sessions_version": version, "sessions_fold": fold, "task": f"sessions-{name}"}
    got = {k: log.get(k) for k in want}
    if got != want:
        raise SystemExit(f"{adapter} was trained on {got}, not {want}")


def test_reads(paths: Paths) -> Path:
    """The ledger of every (version, fold, question) whose test split a final report has read."""
    return paths.out / "results" / "test-reads.jsonl"


def compare(version: str, name: str, adapter: str, paths: Paths | None = None, attn: str = "eager",
            fold: str | None = None, require_trained_on: bool = True) -> dict:
    """The one final report of a (version, fold, question): it alone reads test, once. It stops before any read if
    the adapter was not trained on this version, fold and question, or if the test split of any question it reads is
    in the test-read ledger. Rules fixed
    before the first result (docs/SESSIONS_METHOD.md §7.1; openspec/changes/add-decision-patterns/design.md):
    - base model and adapter use the same attention implementation as the baselines (eager);
    - each model gets its own bias and temperature, fitted on val (E22);
    - the adapter must beat the previous-call baseline and the calibrated base model on test log loss: paired
      bootstrap deltas over the same rows; a gain counts only if the 95% interval of the delta is below zero;
    - the hand-off threshold is chosen on val and applied to test;
    - for work_kind, the ladder (kind, then tool) is compared with flat next_tool on tool groups;
    - results per project (threat T4)."""
    from ..engine import Engine
    paths = paths or Paths()
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"compare-{version}{'-' + fold if fold else ''}-{name}-{Path(adapter).parent.name}-{Path(adapter).name}.json"
    if require_trained_on:
        check_adapter(adapter, version, fold, name)
    reads = [{"version": version, "fold": fold, "question": q} for q in [name] + (["next_tool"] if name == "work_kind" else [])]
    ledger = test_reads(paths)
    done = [json.loads(l) for l in ledger.open()] if ledger.exists() else []
    seen = [r for r in reads if any({k: d.get(k) for k in r} == r for d in done)]
    if path.exists() or seen:
        raise SystemExit(f"test was already read for {seen or path}: a final report reads test once")
    with ledger.open("a") as f:   # written before test is read, so a run that crashes still counts as the one read
        for r in reads:
            f.write(json.dumps(r | {"report": path.name}) + "\n")
    q = QUESTIONS[name]
    train, val, test = (load_split(name, s, version, paths, fold) for s in ("train", "val", "test"))
    y_val, y_test = [r["y"] for r in val], [r["y"] for r in test]
    p_prior, p_by = counted(q, train)
    cond = lambda r: p_by.get(q.condition(r["text"]), p_prior) if q.condition else p_prior
    probs = {"prior": [p_prior] * len(test), "previous_call": [cond(r) for r in test]}
    val_probs = {"previous_call": [cond(r) for r in val]}
    out = {"version": version, "fold": fold, "question": name, "attn": attn, "n_test": len(test), "models": {}}
    for label, engine in (("base", Engine(attn=attn)), ("adapter", Engine(adapter=adapter, attn=attn))):
        z_val = [readout(engine, q, r["text"]) for r in val]
        z_test = [readout(engine, q, r["text"]) for r in test]
        temp, bias = fit_bias_temperature(z_val, y_val)
        cal = lambda z: softmax_list([v / temp + b for v, b in zip(z, bias)])
        probs[f"{label}_bias_temp"] = [cal(z) for z in z_test]
        probs[f"{label}_raw"] = [softmax_list(z) for z in z_test]
        val_probs[f"{label}_bias_temp"] = [cal(z) for z in z_val]
        out["models"][label] = {"name": engine.name, "adapter": engine.adapter, "adapter_sha256": engine.adapter_sha256,
                                "temperature": temp, "bias": bias}
    nll = {k: [-math.log(max(p[y], 1e-12)) for p, y in zip(v, y_test)] for k, v in probs.items()}
    hit = {k: [float(max(range(len(p)), key=p.__getitem__) == y) for p, y in zip(v, y_test)] for k, v in probs.items()}
    out["test"] = {k: report(v, y_test) for k, v in probs.items()}
    out["deltas"] = {f"adapter_bias_temp_minus_{ref}": {"log_loss": paired_delta(nll[ref], nll["adapter_bias_temp"]),
                                                        "accuracy": paired_delta(hit[ref], hit["adapter_bias_temp"])}
                     for ref in ("previous_call", "base_bias_temp")}
    out["hand_off"] = {}
    for k, vp in val_probs.items():
        chosen = choose_threshold(risk_coverage(vp, y_val))
        if chosen is None:
            out["hand_off"][k] = None
            continue
        kept = [h for p, h in zip(probs[k], hit[k]) if confidence(p) >= chosen["threshold"]]
        out["hand_off"][k] = {"threshold": chosen["threshold"], "val": chosen,
                              "test": {"coverage": len(kept) / len(test), "accuracy": sum(kept) / len(kept) if kept else None}}
    if name == "work_kind":
        out["ladder"] = ladder_section(version, fold, paths, test, probs)
    by_project = defaultdict(list)
    for k, r in enumerate(test):
        by_project[r["project"]].append(k)
    out["per_project"] = {proj: {"n": len(ks), **{m: sum(nll[m][k] for k in ks) / len(ks) for m in nll}}
                          for proj, ks in sorted(by_project.items())}
    out["splits_accessed"] = list(ACCESS)
    path.write_text(json.dumps(out, indent=1))
    for ref, d in out["deltas"].items():
        ll = d["log_loss"]
        print(f"{ref}: log loss {ll['mean']:+.3f} [{ll['ci95'][0]:+.3f}, {ll['ci95'][1]:+.3f}]")
    print(f"wrote {path}")
    return out


def ladder_section(version: str, fold: str | None, paths: Paths, test: list[dict], kind_probs: dict) -> dict:
    """Ladder against flat next_tool, on the tool groups of next_tool, over the same test rows (the row ids of
    work_kind and next_tool are the same calls). Every tool_<kind> model is the previous-call count model on train."""
    from .patterns import TOOLS_BY_KIND
    flat_q = QUESTIONS["next_tool"]
    flat_train = load_split("next_tool", "train", version, paths, fold)
    flat_test = {r["id"]: r for r in load_split("next_tool", "test", version, paths, fold)}
    rows = [r for r in test if r["id"] in flat_test]
    y = [flat_test[r["id"]]["y"] for r in rows]
    fp, fb = counted(flat_q, flat_train)
    tool_models = {k: (QUESTIONS[f"tool_{k}"], counted(QUESTIONS[f"tool_{k}"],
                                                        load_split(f"tool_{k}", "train", version, paths, fold)))
                   for k in TOOLS_BY_KIND}
    index = {r["id"]: i for i, r in enumerate(test)}
    methods = {"flat_prior": [fp] * len(rows), "flat_previous_call": [fb.get(flat_q.condition(r["text"]), fp) for r in rows]}
    for source in ("previous_call", "adapter_bias_temp"):
        methods[f"ladder_{source}"] = ladder_groups([kind_probs[source][index[r["id"]]] for r in rows],
                                                    [r["text"] for r in rows], tool_models)
    nll = {k: [-math.log(max(p[t], 1e-12)) for p, t in zip(v, y)] for k, v in methods.items()}
    return {"n": len(rows), "test": {k: report(v, y) for k, v in methods.items()},
            "deltas": {f"{m}_minus_flat_previous_call": paired_delta(nll["flat_previous_call"], nll[m])
                       for m in ("ladder_previous_call", "ladder_adapter_bias_temp")}}
