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


def baselines(version: str, names: list[str], model: str | None = None, paths: Paths | None = None,
              readouts: bool = True) -> dict:
    paths = paths or Paths()
    frozen = manifest(version, paths)["questions"]
    out = {"version": version, "split": "val", "fit_rows": FIT_ROWS, "questions": {}}
    engine = None
    if readouts:
        from ..engine import Engine
        engine = Engine(model) if model else Engine()
        out["model"] = engine.name
    for name in names:
        if name not in frozen:
            print(f"{name}: not in {version}, skipped")
            continue
        q = QUESTIONS[name]
        train, val = load_split(name, "train", version, paths), load_split(name, "val", version, paths)
        if not train or not val:
            print(f"{name}: no train or val rows, skipped")
            continue
        labels = [r["y"] for r in val]
        p_prior, p_by = counted(q, train)
        res = {"n": len(val), "labels": dict(Counter(q.labels[y] for y in labels)),
               "prior": report([p_prior] * len(val), labels)}
        if q.condition:
            res["previous_call"] = report([p_by.get(q.condition(r["text"]), p_prior) for r in val], labels)
        if engine is not None:
            t0 = time.time()
            z_val = [readout(engine, q, r["text"]) for r in val]
            fit = random.Random(SEED).sample(train, min(FIT_ROWS, len(train)))
            z_fit = [readout(engine, q, r["text"]) for r in fit]
            temp, bias = fit_bias_temperature(z_fit, [r["y"] for r in fit])
            res["zero_shot"] = report([softmax_list(z) for z in z_val], labels)
            res["zero_shot_bias_temp"] = {**report([softmax_list([v / temp + b for v, b in zip(z, bias)]) for z in z_val],
                                                   labels), "temperature": temp, "bias": bias}
            res["readout_seconds"] = time.time() - t0
        out["questions"][name] = res
        print(f"\n{name}  n={len(val)}  {res['labels']}")
        for key in ("prior", "previous_call", "zero_shot", "zero_shot_bias_temp"):
            if key in res:
                m = res[key]
                print(f"  {key:20} log loss {m['log_loss']:.3f} [{m['log_loss_ci95'][0]:.3f},{m['log_loss_ci95'][1]:.3f}]"
                      f"  acc {m['accuracy']:.3f}  ece {m['ece']:.3f}")
    out["splits_accessed"] = list(ACCESS)
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"baselines-{version}-{out.get('model', 'counts').split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    return out


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Mean of b - a over the same rows, with a 95% bootstrap interval over rows."""
    d = [y - x for x, y in zip(a, b)]
    return {"mean": sum(d) / len(d), "ci95": bootstrap_ci(d, lambda v: sum(v) / len(v))}


def compare(version: str, name: str, adapter: str, paths: Paths | None = None, attn: str = "eager") -> dict:
    """The fixed comparison for an adapter (docs/SESSIONS_METHOD.md §7.1), written before the first result:
    - base model and adapter use the same attention implementation as the baselines (eager);
    - each model gets its own bias and temperature, fitted on val (E22), and test is read once, here;
    - the adapter must beat the previous-call baseline and the calibrated base model on test log loss: paired
      bootstrap deltas over the same rows; a gain counts only if the 95% interval of the delta is below zero;
    - results per project, because two projects hold most rows (threat T4)."""
    from ..engine import Engine
    paths = paths or Paths()
    q = QUESTIONS[name]
    train, val, test = (load_split(name, s, version, paths) for s in ("train", "val", "test"))
    y_val, y_test = [r["y"] for r in val], [r["y"] for r in test]
    p_prior, p_by = counted(q, train)
    probs = {"previous_call": [p_by.get(q.condition(r["text"]), p_prior) for r in test] if q.condition else
             [p_prior] * len(test)}
    out = {"version": version, "question": name, "attn": attn, "n_test": len(test), "models": {}}
    for label, engine in (("base", Engine(attn=attn)), ("adapter", Engine(adapter=adapter, attn=attn))):
        z_val = [readout(engine, q, r["text"]) for r in val]
        z_test = [readout(engine, q, r["text"]) for r in test]
        temp, bias = fit_bias_temperature(z_val, y_val)
        probs[f"{label}_bias_temp"] = [softmax_list([v / temp + b for v, b in zip(z, bias)]) for z in z_test]
        probs[f"{label}_raw"] = [softmax_list(z) for z in z_test]
        out["models"][label] = {"name": engine.name, "adapter": engine.adapter, "adapter_sha256": engine.adapter_sha256,
                                "temperature": temp, "bias": bias}
    nll = {k: [-math.log(max(p[y], 1e-12)) for p, y in zip(v, y_test)] for k, v in probs.items()}
    hit = {k: [float(max(range(len(p)), key=p.__getitem__) == y) for p, y in zip(v, y_test)] for k, v in probs.items()}
    out["test"] = {k: report(v, y_test) for k, v in probs.items()}
    out["deltas"] = {f"adapter_bias_temp_minus_{ref}": {"log_loss": paired_delta(nll[ref], nll["adapter_bias_temp"]),
                                                        "accuracy": paired_delta(hit[ref], hit["adapter_bias_temp"])}
                     for ref in ("previous_call", "base_bias_temp")}
    by_project = defaultdict(list)
    for k, r in enumerate(test):
        by_project[r["project"]].append(k)
    out["per_project"] = {proj: {"n": len(ks), **{m: sum(nll[m][k] for k in ks) / len(ks) for m in nll}}
                          for proj, ks in sorted(by_project.items())}
    out["splits_accessed"] = list(ACCESS)
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    path = folder / f"compare-{version}-{name}-{Path(adapter).parent.name}-{Path(adapter).name}.json"
    path.write_text(json.dumps(out, indent=1))
    for ref, d in out["deltas"].items():
        ll = d["log_loss"]
        print(f"{ref}: log loss {ll['mean']:+.3f} [{ll['ci95'][0]:+.3f}, {ll['ci95'][1]:+.3f}]")
    print(f"wrote {path}")
    return out
