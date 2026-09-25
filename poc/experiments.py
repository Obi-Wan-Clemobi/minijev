"""minijev POC experiments. Run from poc/:

    uv run python experiments.py <name> [--model Qwen/Qwen2.5-1.5B-Instruct]

    demo         DESIGN.md §4 request → Jev-shaped response; naive / kv / packed agree
    tree         TypeSafe's 13 GDPR cookbook questions: modes agree, and batched == single
    latency      E3: time vs number of questions, per mode, at two state lengths
    jevdocs      Jev's documented examples: our answers next to Jev's published ones
    permutation  E4: rotate Choice option order; listwise letters move, pointwise can't
    calibration  BoolQ Nouls: raw vs contextual vs temperature-scaled accuracy / ECE / Brier / NLL
    llm_vs_minijev  E10: the same model generating its answer (label / JSON / 1 token + logprobs) vs a
                 minijev readout: AG News accuracy + calibration + speed, and a direct prefill/decode
                 cost measurement
    fanout       E10, many questions on one state: six methods over 1 / 4 / 13 questions
    quality      E11, labelled BoolQ + AG News: readout (raw, calibrated) vs the same model writing its
                 answer or a probability: accuracy, ECE, parse failures, time per question
    same_format  E12, one request answered in minijev's JSON: read out vs the same model writing that JSON
    order_bias   E13, AG News in 4 option orders: listwise vs averaged vs debiased vs pointwise
    lora         E20, base vs a LoRA adapter (--adapter, from train_lora.py) on the test split, with order flips
    all          everything above

Results go to poc/results/<name>[-<model>].json. Downloads are cached in poc/data/.
Everything runs on CPU in fp32; see docs/RESEARCH.md §7 for measured results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from minijev.calibrate import (cross_fit, cross_fit_multiclass, fit_affine, fit_temperature_multiclass, logit, nll,
                               nll_multi, sigmoid, softmax_list)
from minijev.engine import MODES, Branch, Engine
from minijev.fixtures import (AG_OPTIONS, AG_QUESTION, SST5_LEVELS, SST5_QUESTION, GDPR_JEV, GDPR_QUESTIONS, JEV_DOC_CASES,
                              SHOES, SUPPORT_TICKET, fetch, gdpr_state, gdpr_text)
from minijev.generation import (generate, generate_batched, generate_logprobs,
                                match_option, number, options_text, parse_json,
                                parse_probs_lenient, same_format_run, verbal_probability)
from minijev.judge import ask, raw_scores
from minijev.primitives import answer, class_logits
from minijev.prompt import CONTENT_FREE_STATE, SYSTEM, choice_block, noul_block
from minijev.settings import MODEL, Settings

HERE = Path(__file__).parent
DATA, RESULTS = HERE / "data", HERE / "results"
DATASETS = HERE / "datasets"
MANIFEST_PATH = DATASETS / "manifest.json"

# ---------------------------------------------------------------------------
# Dataset versioning (W17)

def _row_checksum(row: dict) -> str:
    """Compute SHA256 checksum of a dataset row (serialized as stable JSON)."""
    stable = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode()).hexdigest()


def _compute_manifest_entry(dataset_name: str, rows: list[dict]) -> dict:
    """Compute manifest entry for a dataset with per-row checksums."""
    checksums = [_row_checksum(row) for row in rows]
    return {
        "revision": "main",
        "rows": len(rows),
        "sha256": checksums,
        "fetched": time.strftime("%Y-%m-%d")
    }


def _verify_dataset_checksums(dataset_name: str, rows: list[dict]) -> None:
    """Verify that dataset rows match the manifest checksums. Fails loudly on mismatch."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    if dataset_name not in manifest:
        print(f"Warning: {dataset_name} not in manifest. Run with --update-manifest to pin it.")
        return

    entry = manifest[dataset_name]
    expected_checksums = entry["sha256"]

    if len(rows) != entry["rows"]:
        raise ValueError(
            f"Dataset {dataset_name} row count mismatch:\n"
            f"  Expected: {entry['rows']} rows (from manifest {entry['fetched']})\n"
            f"  Got: {len(rows)} rows\n"
            f"This indicates the upstream dataset has changed.\n"
            f"Run with --update-manifest to update the pinned version."
        )

    # Check each row's checksum
    mismatches = []
    for i, row in enumerate(rows):
        actual = _row_checksum(row)
        if actual != expected_checksums[i]:
            mismatches.append((i, expected_checksums[i], actual))

    if mismatches:
        # Show first few mismatches
        details = "\n".join(
            f"  Row {i}: expected {exp[:12]}..., got {act[:12]}..."
            for i, exp, act in mismatches[:5]
        )
        if len(mismatches) > 5:
            details += f"\n  ... and {len(mismatches) - 5} more rows"

        raise ValueError(
            f"Dataset {dataset_name} checksum mismatch:\n"
            f"  {len(mismatches)}/{len(rows)} rows differ from manifest ({entry['fetched']})\n"
            f"{details}\n"
            f"This indicates silent dataset drift. Options:\n"
            f"  1. Run with --update-manifest to accept the new version\n"
            f"  2. Investigate why the dataset changed\n"
            f"  3. Delete {DATA} to re-download from scratch"
        )


def update_manifest() -> None:
    """Generate or update manifest.json with checksums for all datasets."""
    DATASETS.mkdir(exist_ok=True)

    print("Downloading datasets and computing checksums...")

    # BoolQ: download all rows (no sampling, no verification during download)
    print("\n  BoolQ validation split...")
    boolq_path = DATA / "boolq_validation.json"
    if not boolq_path.exists():
        rows, offset = [], 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=google/boolq&config=default"
                   f"&split=validation&offset={offset}&length=100")
            page = json.loads(fetch(url, DATA / f"boolq_page_{offset}.json").read_text())
            rows += [r["row"] for r in page["rows"]]
            offset += 100
            print(f"    Downloaded {len(rows)} rows...", end="\r", flush=True)
            if offset >= page["num_rows_total"]:
                break
        boolq_path.write_text(json.dumps(rows))
    boolq_rows = json.loads(boolq_path.read_text())
    print(f"    Downloaded {len(boolq_rows)} rows")

    # AG News: download from four offsets
    print("\n  AG News test split...")
    agnews_rows = []
    for offset in (0, 1900, 3800, 5700):
        url = ("https://datasets-server.huggingface.co/rows?dataset=fancyzhx/ag_news&config=default"
               f"&split=test&offset={offset}&length=100")
        agnews_rows += [r["row"] for r in json.loads(fetch(url, DATA / f"agnews_{offset}.json").read_text())["rows"]]
    print(f"    Downloaded {len(agnews_rows)} rows")

    # Compute manifest
    print("\n  Computing checksums...")
    manifest = {
        "boolq": _compute_manifest_entry("boolq", boolq_rows),
        "ag_news": _compute_manifest_entry("ag_news", agnews_rows),
    }

    # Write manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ Manifest written to {MANIFEST_PATH}")
    print(f"  - boolq: {manifest['boolq']['rows']} rows")
    print(f"  - ag_news: {manifest['ag_news']['rows']} rows")
    print(f"  - Fetched: {manifest['boolq']['fetched']}")

# ---------------------------------------------------------------------------
# Shared fixtures


def tracked(q: dict, a: dict) -> float:
    """The one number the cookbook tracks per answer."""
    if a["type"] == "noul":
        return a["noul"]
    if a["type"] == "choice":
        return max(a["probabilities"].values())
    return a["score"] / (len(q["criteria"]) - 1)


def full_answers(engine: Engine, req: dict, mode: str) -> tuple[dict, dict, float]:
    t0 = time.perf_counter()
    raw, usage = raw_scores(engine, req, mode)
    dt = time.perf_counter() - t0
    return {qid: answer(q, raw[qid]["logits"]) for qid, q in req["questions"].items()}, raw, dt


def max_logit_gap(a: dict, b: dict) -> float:
    return max(abs(x - y) for qid in a for x, y in zip(a[qid]["logits"], b[qid]["logits"]))


# ---------------------------------------------------------------------------
# Experiments


def demo(engine: Engine) -> dict:
    req = {
        "model": "minijev-latest",
        "state": "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
        "questions": {
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
                "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages", "sales": "Pricing, new plans"}},
            "tone": {"type": "score", "instructions": "How upset is the customer?",
                     "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
        },
    }
    resp = ask(engine, req, "packed", settings=Settings())  # uncalibrated defaults, not minijev.env
    print(json.dumps(resp, indent=2))
    raws = {m: raw_scores(engine, req, m)[0] for m in MODES}
    gaps = {m: max_logit_gap(raws["naive"], raws[m]) for m in ("kv", "packed")}
    print(f"max |logit difference| vs naive: kv {gaps['kv']:.1e}, packed {gaps['packed']:.1e}")
    return {"request": req, "response": resp, "max_logit_gap_vs_naive": gaps}


def tree(engine: Engine, n_tokens: int = 1000) -> dict:
    state = gdpr_state(engine, n_tokens)
    req = {"state": state, "questions": GDPR_QUESTIONS}
    results = {m: full_answers(engine, req, m) for m in MODES}
    gaps = {m: max_logit_gap(results["naive"][1], results[m][1]) for m in ("kv", "packed")}
    batched = results["packed"][0]
    singles = {qid: full_answers(engine, {"state": state, "questions": {qid: q}}, "packed")[0][qid]
               for qid, q in GDPR_QUESTIONS.items()}
    rows = []
    print(f"state: first {n_tokens} tokens of the GDPR article; {len(GDPR_QUESTIONS)} questions")
    print(f"{'question':20s} {'batched':>8s} {'single':>8s} {'|diff|':>8s} {'Jev*':>6s}")
    for qid, q in GDPR_QUESTIONS.items():
        b, s = tracked(q, batched[qid]), tracked(q, singles[qid])
        rows.append({"question": qid, "batched": b, "single": s, "jev_full_article": GDPR_JEV[qid]})
        print(f"{qid:20s} {b:8.3f} {s:8.3f} {abs(b - s):8.1e} {GDPR_JEV[qid]:6.3f}")
    isolation_gap = max(abs(r["batched"] - r["single"]) for r in rows)
    print(f"max |logit difference| vs naive: kv {gaps['kv']:.1e}, packed {gaps['packed']:.1e}")
    print(f"max |batched − single| over tracked numbers: {isolation_gap:.1e}")
    print("time: " + ", ".join(f"{m} {results[m][2]:.2f}s" for m in MODES))
    print("*Jev column: TypeSafe's cookbook on the FULL article (jev-1.12); ours sees only the first "
          f"{n_tokens} tokens, so it is context, not a like-for-like score.")
    return {"n_tokens": n_tokens, "max_logit_gap_vs_naive": gaps, "max_batched_single_gap": isolation_gap,
            "seconds": {m: results[m][2] for m in MODES}, "rows": rows}


def latency(engine: Engine, lengths=(250, 1000), counts=(1, 4, 13), repeats: int = 5) -> dict:
    """E3: latency vs question count, with improved timing measurement (warm-up + rotation + median + IQR)."""
    keys = list(GDPR_QUESTIONS)
    out = []
    # Warm-up: 3 runs to load model and compile kernels
    for _ in range(3):
        raw_scores(engine, {"state": "warm up", "questions": {"q": GDPR_QUESTIONS["breach_72h"]}}, "packed")

    for n_tokens in lengths:
        state = gdpr_state(engine, n_tokens)
        for n in counts:
            req = {"state": state, "questions": {k: GDPR_QUESTIONS[k] for k in keys[:n]}}
            # Every mode runs `repeats` times. Each round shifts the order of the modes (ABC, BCA, CAB, ...), so that
            # a slow drift of the CPU speed spreads over all modes instead of landing on the one timed last.
            mode_times = {mode: [] for mode in MODES}
            usage_per_mode = {}
            for r in range(repeats):
                for mode in MODES[r % len(MODES):] + MODES[:r % len(MODES)]:
                    t0 = time.perf_counter()
                    _, usage = raw_scores(engine, req, mode)
                    mode_times[mode].append(time.perf_counter() - t0)
                    usage_per_mode[mode] = usage

            for mode in MODES:
                times = mode_times[mode]
                median = statistics.median(times)
                q1 = statistics.quantiles(times, n=4)[0] if len(times) >= 2 else median
                q3 = statistics.quantiles(times, n=4)[2] if len(times) >= 2 else median
                row = {"state_tokens": n_tokens, "questions": n, "mode": mode,
                       "median_s": median, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                       "all_times": times,
                       "input_tokens": usage_per_mode[mode]["input_tokens"]}
                out.append(row)
                print(f"state {n_tokens:5d} tok  questions {n:2d}  {mode:6s}  median {median:6.3f}s "
                      f"[Q1={q1:.3f}, Q3={q3:.3f}]  (input tokens {usage_per_mode[mode]['input_tokens']})")
    return {"repeats": repeats, "rows": out,
            "note": f"median and IQR over {repeats} runs with rotation order; warm-up 3x; CPU fp32; 6 threads",
            "rotation_pattern": "each mode runs every round; the mode order shifts each round (ABC, BCA, CAB, ...)"}


def jevdocs(engine: Engine) -> dict:
    """Our answers on Jev's documented inputs, raw and with contextual calibration (cc).

    cc (Zhao et al. 2021): also run the question on a content-free state and subtract those
    logits, which removes the model's prior for each label, level or yes/no.
    """
    rows = []
    for label, state, q, jev in JEV_DOC_CASES:
        variants = {"": q}
        if q["type"] == "score":
            variants = {"": q, " [listwise]": {**q, "score_mode": "listwise"}}
        if q["type"] == "choice":
            variants = {"": q, " [pointwise]": {**q, "choice_mode": "pointwise"}}
        for suffix, qq in variants.items():
            z = raw_scores(engine, {"state": state, "questions": {"q": qq}})[0]["q"]["logits"]
            z_cf = raw_scores(engine, {"state": CONTENT_FREE_STATE, "questions": {"q": qq}})[0]["q"]["logits"]
            for cc in (False, True):
                a = answer(qq, [x - y for x, y in zip(z, z_cf)] if cc else z)
                ours = a["noul"] if a["type"] == "noul" else a["choice"] if a["type"] == "choice" else a["score"]
                rows.append({"case": label + suffix, "cc": cc, "state": state, "jev": jev, "ours": ours,
                             "probabilities": a.get("probabilities")})
            shown = [f"{r['ours']:.2f}" if isinstance(r["ours"], float) else r["ours"] for r in rows[-2:]]
            print(f"{label + suffix:48s} jev {jev!s:>10}  ours {shown[0]:>10}  +cc {shown[1]:>10}   {state[:40]}")

    # Agreement with Jev's own published outputs. Not accuracy: Jev is not ground truth.
    def pick(kind: str, cc: bool, variant: str = "") -> list[dict]:
        return [r for r in rows if r["cc"] == cc and r["case"].startswith(kind) and "numbers-only" not in r["case"]
                and (variant in r["case"] if variant else "[" not in r["case"])]

    summary = {}
    for cc in (False, True):
        nouls = pick("noul", cc)
        s = {
            "noul_mean_abs_diff": statistics.mean(abs(r["ours"] - r["jev"]) for r in nouls),
            "noul_same_side_of_0.5": sum((r["ours"] > 0.5) == (r["jev"] > 0.5) for r in nouls) / len(nouls),
            "noul_pearson": pearson([r["ours"] for r in nouls], [r["jev"] for r in nouls]),
        }
        for name, rs in (("score_pointwise", pick("score", cc)), ("score_listwise", pick("score", cc, "listwise"))):
            s[f"{name}_mean_abs_diff"] = statistics.mean(abs(r["ours"] - r["jev"]) for r in rs)
            s[f"{name}_same_rounded_level"] = sum(round(r["ours"]) == round(r["jev"]) for r in rs) / len(rs)
        for name, rs in (("choice_listwise", pick("choice", cc)), ("choice_pointwise", pick("choice", cc, "pointwise"))):
            s[f"{name}_same_argmax"] = sum(r["ours"] == r["jev"] for r in rs) / len(rs)
        summary["cc" if cc else "raw"] = s
    summary["n"] = {"noul": len(pick("noul", False)), "score": len(pick("score", False)),
                    "choice": len(pick("choice", False))}
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


def permutation(engine: Engine) -> dict:
    """E4: rotate the option order of every documented Choice and see how far the answer moves.

    Listwise letter labels carry position/label bias (Zheng et al. 2024). Pointwise judging is
    order-invariant by construction; this also demonstrates the IIA test proposed for real Jev.
    """
    rows = []
    for label, state, q, jev in JEV_DOC_CASES:
        if q["type"] != "choice":
            continue
        keys = list(q["criteria"])
        for mode in ("listwise", "pointwise"):
            dists = []
            for r in range(len(keys)):
                order = keys[r:] + keys[:r]
                qq = {**q, "criteria": {k: q["criteria"][k] for k in order}, "choice_mode": mode}
                dists.append(ask(engine, {"state": state, "questions": {"q": qq}}, settings=Settings(), debug=True)["answers"]["q"]["probabilities"])
            winners = [max(d, key=d.get) for d in dists]
            rows.append({
                "case": label, "mode": mode, "k": len(keys),
                "max_abs_dp": max(abs(d[k] - dists[0][k]) for d in dists for k in keys),
                "argmax_changes": sum(w != winners[0] for w in winners[1:]) / (len(winners) - 1),
                "winners": winners, "jev": jev,
            })
            print(f"{label:36s} {mode:9s} k={len(keys)}  max|Δp| {rows[-1]['max_abs_dp']:.3f}  "
                  f"argmax changes {rows[-1]['argmax_changes']:.2f}  winners {winners}")
    summary = {}
    for mode in ("listwise", "pointwise"):
        rs = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"mean_max_abs_dp": statistics.mean(r["max_abs_dp"] for r in rs),
                         "mean_argmax_change_rate": statistics.mean(r["argmax_changes"] for r in rs)}
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# E10: the usual way (ask the model to *write* its answer) vs minijev (read the answer)
#
# Same model, same weights, same CPU: the only difference is generation vs readout.


def ag_news(n: int, seed: int = 0, verify: bool = True) -> list[dict]:
    """A seeded stratified sample of AG News test items, taken from four places in the 7,600-row split.

    Uses stratified sampling to preserve the label distribution across 4 topics (W15).
    """
    rows = []
    for offset in (0, 1900, 3800, 5700):
        url = ("https://datasets-server.huggingface.co/rows?dataset=fancyzhx/ag_news&config=default"
               f"&split=test&offset={offset}&length=100")
        rows += [r["row"] for r in json.loads(fetch(url, DATA / f"agnews_{offset}.json").read_text())["rows"]]
    if verify and MANIFEST_PATH.exists():
        _verify_dataset_checksums("ag_news", rows)
    return stratified_sample(rows, n, "label", seed)


@torch.inference_mode()
def decode_cost(engine: Engine, context: int = 600, steps: int = 20, repeats: int = 5) -> dict:
    """Prefill and decode cost per token, measured directly at one fixed context length.

    Prefill: one pass over `context` tokens. Decode: `steps` single-token passes on top of that
    context. Uses improved timing measurement: warm-up + median + IQR over repeats.
    """
    ids = engine.tok.encode(gdpr_text(), add_special_tokens=False)[:context]

    # Warm-up: 3 runs
    for _ in range(3):
        cache = DynamicCache()
        out = engine.model(torch.tensor([ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(steps):
            nxt = engine.model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)

    # Measure: 5 runs
    prefill, decode = [], []
    for _ in range(repeats):
        cache = DynamicCache()
        t0 = time.perf_counter()
        out = engine.model(torch.tensor([ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        prefill.append((time.perf_counter() - t0) / context)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            nxt = engine.model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        decode.append((time.perf_counter() - t0) / steps)

    ms_prefill = [1000 * v for v in prefill]
    ms_decode = [1000 * v for v in decode]

    prefill_median = statistics.median(ms_prefill)
    prefill_q1 = statistics.quantiles(ms_prefill, n=4)[0] if len(ms_prefill) >= 2 else prefill_median
    prefill_q3 = statistics.quantiles(ms_prefill, n=4)[2] if len(ms_prefill) >= 2 else prefill_median

    decode_median = statistics.median(ms_decode)
    decode_q1 = statistics.quantiles(ms_decode, n=4)[0] if len(ms_decode) >= 2 else decode_median
    decode_q3 = statistics.quantiles(ms_decode, n=4)[2] if len(ms_decode) >= 2 else decode_median

    return {"context_tokens": context, "decode_steps": steps, "repeats": repeats,
            "ms_per_prefill_token": prefill_median, "prefill_q1": prefill_q1, "prefill_q3": prefill_q3, "prefill_iqr": prefill_q3 - prefill_q1,
            "ms_per_decode_token": decode_median, "decode_q1": decode_q1, "decode_q3": decode_q3, "decode_iqr": decode_q3 - decode_q1,
            "decode_over_prefill": decode_median / prefill_median,
            "all_ms_per_prefill_token": ms_prefill,
            "all_ms_per_decode_token": ms_decode}


def bootstrap_ci(values: list, stat, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile interval of stat(sample) over resamples of the items."""
    rng = random.Random(seed)
    stats = sorted(stat([values[rng.randrange(len(values))] for _ in values]) for _ in range(n_boot))
    return [stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]]


def multiclass_metrics(probs: list[list[float]], labels: list[int]) -> dict:
    """Accuracy, top-label ECE (10 equal-width bins on [0, 1]) and multiclass Brier."""
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for p, y in zip(probs, labels):
        conf, pred = max(p), max(range(len(p)), key=p.__getitem__)
        b = bins[min(int(conf * 10), 9)]
        b[0], b[1], b[2] = b[0] + 1, b[1] + conf, b[2] + (pred == y)
    n = len(labels)
    return {
        "accuracy": sum(max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)) / n,
        "ece": sum(abs(c[1] - c[2]) for c in bins if c[0]) / n,
        "brier": sum(sum((pk - (k == y)) ** 2 for k, pk in enumerate(p)) for p, y in zip(probs, labels)) / n,
    }


def with_ci(probs: list[list[float]], labels: list[int]) -> dict:
    """multiclass_metrics plus 95% bootstrap intervals for accuracy and ECE."""
    pairs = list(zip(probs, labels))
    m = multiclass_metrics(probs, labels)
    for k in ("accuracy", "ece"):
        m[f"{k}_ci95"] = bootstrap_ci(pairs, lambda s, k=k: multiclass_metrics([p for p, _ in s], [y for _, y in s])[k])
    return m


def mean_ci(xs: list[float]) -> dict:
    return {"mean_s": statistics.mean(xs), "mean_s_ci95": bootstrap_ci(xs, statistics.mean)}


def llm_vs_minijev(engine: Engine, n: int = 120, n_json: int = 60) -> dict:
    """E10. JSON-with-probabilities is slow to generate, so it runs on the first n_json items only."""
    keys = list(AG_OPTIONS)
    q = {"type": "choice", "instructions": AG_QUESTION, "criteria": AG_OPTIONS}
    ask_label = "Answer with the name of the best option only."
    ask_probs = ("Reply with a JSON object that maps every option name to the probability that it is the right "
                 "answer. The probabilities must add up to 1. Reply with the JSON object only.")
    generate(engine, "warm up", 2)
    rows = []
    t_start = time.perf_counter()
    for i, ex in enumerate(ag_news(n)):
        head = f"STATE:\n{ex['text']}\n\nQUESTION: {AG_QUESTION}\nOPTIONS:\n{options_text(AG_OPTIONS)}\n"
        t0 = time.perf_counter()
        raw, usage = raw_scores(engine, {"state": ex["text"], "questions": {"q": q}}, "packed")
        t_readout = time.perf_counter() - t0
        p_readout = answer(q, raw["q"]["logits"])["probabilities"]
        # The readout's own prompt through an LLM API that returns logprobs: generate 1 token.
        t0 = time.perf_counter()
        lp = generate_logprobs(engine, engine.prefix_ids(ex["text"]) + engine.suffix_ids(choice_block(q)))
        p_logprobs = answer(q, class_logits(lp, engine.letters[:len(keys)])[0])["probabilities"]
        t_logprobs = time.perf_counter() - t0
        g_label = generate(engine, head + ask_label, 10)
        row = {
            "label": ex["label"],
            "readout": {"probs": [p_readout[k] for k in keys], "seconds": t_readout,
                        "input_tokens": usage["input_tokens"]},
            "generate_1_token_logprobs": {"probs": [p_logprobs[k] for k in keys], "seconds": t_logprobs},
            "generate_label": {"answer": match_option(g_label["text"], keys), "text": g_label["text"],
                               "seconds": g_label["seconds"], "new_tokens": g_label["new_tokens"]},
        }
        if i < n_json:
            g_probs = generate(engine, head + ask_probs, 16 * len(keys) + 16)
            parsed = parse_json(g_probs["text"]) or {}
            verbal = [max(0.0, number(parsed.get(k))) for k in keys]
            row["generate_json_probs"] = {"probs": verbal if sum(verbal) > 0 else None,
                                          "probs_lenient": parse_probs_lenient(g_probs["text"], keys),
                                          "text": g_probs["text"], "seconds": g_probs["seconds"],
                                          "new_tokens": g_probs["new_tokens"], "raw_sum": sum(verbal)}
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}  ({time.perf_counter() - t_start:.0f}s)", flush=True)

    labels = [r["label"] for r in rows]
    uniform = [1 / len(keys)] * len(keys)
    argmax = lambda p: max(range(len(p)), key=p.__getitem__)
    label_answers = [r["generate_label"]["answer"] for r in rows]
    json_rows = [r["generate_json_probs"] for r in rows[:n_json]]

    def json_entry(field: str) -> dict:
        probs = [j[field] for j in json_rows]
        correct = [p is not None and argmax(p) == y for p, y in zip(probs, labels[:n_json])]  # unparsed = wrong
        # ECE and Brier need a distribution for every item, so unparsed replies get a uniform one. That flatters
        # ECE (uniform rows sit in the lowest bin), so do not read ECE here as evidence of calibration.
        m = multiclass_metrics([[v / sum(p) for v in p] if p else uniform for p in probs], labels[:n_json])
        return {**m, "accuracy": sum(correct) / n_json, "accuracy_ci95": bootstrap_ci(correct, statistics.mean),
                "ece_note": "unparsed replies filled with a uniform distribution",
                "n": n_json, "parse_failures": sum(p is None for p in probs),
                **mean_ci([j["seconds"] for j in json_rows]),
                "output_tokens": statistics.mean(j["new_tokens"] for j in json_rows)}

    label_correct = [a == keys[y] for a, y in zip(label_answers, labels)]
    single = {
        "readout (minijev)": {**with_ci([r["readout"]["probs"] for r in rows], labels), "n": n, "parse_failures": 0,
                              **mean_ci([r["readout"]["seconds"] for r in rows]), "output_tokens": 0},
        "generate 1 token + logprobs": {**with_ci([r["generate_1_token_logprobs"]["probs"] for r in rows], labels),
                                        "n": n, "parse_failures": 0,
                                        **mean_ci([r["generate_1_token_logprobs"]["seconds"] for r in rows]),
                                        "output_tokens": 1},
        "generate label": {"accuracy": statistics.mean(label_correct),
                           "accuracy_ci95": bootstrap_ci(label_correct, statistics.mean),
                           "ece": None, "brier": None, "n": n,
                           "parse_failures": sum(a is None for a in label_answers),
                           **mean_ci([r["generate_label"]["seconds"] for r in rows]),
                           "output_tokens": statistics.mean(r["generate_label"]["new_tokens"] for r in rows)},
        f"readout (minijev), first {n_json}": {**with_ci([r["readout"]["probs"] for r in rows[:n_json]], labels[:n_json]),
                                               "n": n_json, "parse_failures": 0,
                                               **mean_ci([r["readout"]["seconds"] for r in rows[:n_json]]),
                                               "output_tokens": 0},
        "generate JSON probabilities (strict parser)": json_entry("probs"),
        "generate JSON probabilities (lenient parser)": json_entry("probs_lenient"),
    }
    readout_pick = [keys[argmax(r["readout"]["probs"])] for r in rows]
    agreement = sum(a == b for a, b in zip(label_answers, readout_pick)) / n
    logprob_gap = max(abs(a - b) for r in rows
                      for a, b in zip(r["readout"]["probs"], r["generate_1_token_logprobs"]["probs"]))
    costs = decode_cost(engine)

    print("\nAG News (same model, same CPU). Accuracy by chance = 0.25. [..] = 95% bootstrap interval.")
    print(f"{'method':46s} {'n':>4s} {'acc':>6s} {'acc CI':>13s} {'ECE':>6s} {'fails':>6s} {'mean s':>7s} {'out tok':>8s}")
    for name, m in single.items():
        fmt = lambda v: f"{v:6.3f}" if isinstance(v, float) else f"{'—':>6s}"
        ci = m["accuracy_ci95"]
        print(f"{name:46s} {m['n']:4d} {fmt(m['accuracy'])} [{ci[0]:.3f},{ci[1]:.3f}] {fmt(m['ece'])} "
              f"{m['parse_failures']:6d} {m['mean_s']:7.2f} {m['output_tokens']:8.1f}")
    print(f"generated label == minijev argmax on {agreement:.1%} of items")
    print(f"max |p(readout) − p(1 token + logprobs)| = {logprob_gap:.1e}")
    print(f"cost on this CPU at {costs['context_tokens']} tokens of context: prefill "
          f"{costs['ms_per_prefill_token']:.1f} ms/token, decode {costs['ms_per_decode_token']:.0f} ms/token "
          f"({costs['decode_over_prefill']:.0f}×)")

    return {"n": n, "single_decision": single, "label_vs_readout_agreement": agreement,
            "readout_vs_logprobs_max_abs_dp": logprob_gap, "generation_costs": costs, "rows": rows}


def as_choice(q: dict) -> dict:
    """The cookbook's Nouls and Scores as plain Choices, so every method answers the same kind of question."""
    if q["type"] == "noul":
        return {"type": "choice", "instructions": q["instructions"], "criteria": {"yes": None, "no": None}}
    if q["type"] == "score":
        return {"type": "choice", "instructions": q["instructions"],
                "criteria": {lv.split(":")[0]: lv.split(":", 1)[1].strip() for lv in q["criteria"]}}
    return q


def fan_out(engine: Engine, state_tokens: int, counts=(1, 4, 13), repeats: int = 3) -> list[dict]:
    """Many questions about one state: six ways to answer them, each timed `repeats` times (median reported)."""
    state = gdpr_state(engine, state_tokens)
    doc = json.dumps(state, indent=2, ensure_ascii=False)
    questions = {k: as_choice(q) for k, q in GDPR_QUESTIONS.items()}
    prefix = engine.prefix_ids(state)
    generate(engine, "warm up", 2)  # the first generate() call pays one-time setup costs
    raw_scores(engine, {"state": "warm up", "questions": {"q": questions["breach_72h"]}}, "packed")

    def by_name(q: dict) -> str:
        return (f"QUESTION: {q['instructions']}\nOPTIONS:\n{options_text(q['criteria'])}\n"
                "Answer with the name of the best option only.")

    def letter_pick(q: dict, lp: torch.Tensor) -> str:
        keys = list(q["criteria"])
        return answer(q, class_logits(lp, engine.letters[:len(keys)])[0])["choice"]

    out = []
    print(f"\nFan-out: GDPR article cut to {state_tokens} tokens, Choice questions, median of {repeats} runs")
    for n in counts:
        qs = dict(list(questions.items())[:n])

        def per_question():  # 1) one completion per question: the state is prefilled every time
            ans, tokens = {}, 0
            for qid, q in qs.items():
                g = generate(engine, f"STATE:\n{doc}\n\n{by_name(q)}", 12)
                ans[qid], tokens = match_option(g["text"], list(q["criteria"])), tokens + g["new_tokens"]
            return ans, tokens

        @torch.inference_mode()
        def per_question_cached():  # 2) the same, with the state's KV cache reused (prompt caching)
            cache, ans, tokens = DynamicCache(), {}, 0
            engine.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            for qid, q in qs.items():
                x = torch.tensor([prefix + engine.suffix_ids(by_name(q))])
                new = engine.model.generate(x, attention_mask=torch.ones_like(x), past_key_values=cache,
                                            max_new_tokens=12, do_sample=False, temperature=None, top_p=None,
                                            top_k=None, repetition_penalty=1.0,
                                            pad_token_id=engine.tok.eos_token_id)[0, x.shape[1]:]
                ans[qid] = match_option(engine.tok.decode(new, skip_special_tokens=True), list(q["criteria"]))
                tokens += len(new)
                cache.crop(len(prefix))  # back to the shared state for the next question
            return ans, tokens

        def batched_cached():  # 3) state cached, and all questions decoded together as one batch
            new = generate_batched(engine, prefix, [engine.suffix_ids(by_name(q)) for q in qs.values()], 12)
            return ({qid: match_option(engine.tok.decode(t, skip_special_tokens=True), list(q["criteria"]))
                     for (qid, q), t in zip(qs.items(), new)}, sum(map(len, new)))

        def logprobs_cached():  # 4) state cached, 1 generated token per question, with logprobs
            cache, ans = DynamicCache(), {}
            with torch.inference_mode():
                engine.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            for qid, q in qs.items():
                ans[qid] = letter_pick(q, generate_logprobs(engine, prefix + engine.suffix_ids(choice_block(q)), cache))
                cache.crop(len(prefix))
            return ans, len(qs)

        def one_json():  # 5) one completion that answers every question as JSON
            listing = "\n\n".join(f"q{i}: {q['instructions']}\nOPTIONS:\n{options_text(q['criteria'])}"
                                  for i, q in enumerate(qs.values(), start=1))
            g = generate(engine, f"STATE:\n{doc}\n\nAnswer each question below by choosing one of its options.\n\n"
                                 f"{listing}\n\nReply with a JSON object that maps each question id (q1, q2, ...) to "
                                 f"the name of the chosen option. Reply with the JSON object only.", 24 * n + 40)
            parsed = parse_json(g["text"]) or {}
            return ({qid: match_option(str(parsed.get(f"q{i}", "")), list(q["criteria"]))
                     for i, (qid, q) in enumerate(qs.items(), start=1)}, g["new_tokens"])

        def readout():  # 6) minijev: one packed pass, one readout per question, nothing generated
            raw, _ = raw_scores(engine, {"state": state, "questions": qs}, "packed")
            return {qid: answer(q, raw[qid]["logits"])["choice"] for qid, q in qs.items()}, 0

        methods = {"generate_per_question": per_question, "generate_per_question_cached": per_question_cached,
                   "generate_batched_cached": batched_cached, "generate_1_token_logprobs_cached": logprobs_cached,
                   "generate_one_json": one_json, "readout_packed": readout}
        row: dict = {"questions": n, "input_tokens_packed": len(prefix) + sum(
            len(engine.suffix_ids(choice_block(q))) for q in qs.values())}
        # Round-robin over methods, in a new random order each round, so that slow drift of the CPU speed
        # (turbo, heat, other load) spreads over all methods instead of landing on the ones timed last.
        results, times = {}, {name: [] for name in methods}
        order = list(methods)
        for r in range(repeats):
            random.Random(1000 * n + r).shuffle(order)
            for name in order:
                t0 = time.perf_counter()
                results[name] = methods[name]()
                times[name].append(time.perf_counter() - t0)
        for name in methods:
            t = times[name]
            median = statistics.median(t)
            q1 = statistics.quantiles(t, n=4)[0] if len(t) >= 2 else median
            q3 = statistics.quantiles(t, n=4)[2] if len(t) >= 2 else median
            row[name] = {"median_s": median, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
                         "seconds_all": t, "output_tokens": results[name][1]}
        picks = results["readout_packed"][0]
        for name in methods:
            ans = results[name][0]
            row[name]["parse_failures"] = sum(v is None for v in ans.values())
            row[name]["agrees_with_readout"] = sum(ans[k] == picks[k] for k in qs) / n
        row["generate_batched_cached"]["agrees_with_per_question_cached"] = sum(
            results["generate_batched_cached"][0][k] == results["generate_per_question_cached"][0][k] for k in qs) / n
        out.append(row)
        print(f"  {n:2d} questions: " + "  |  ".join(
            f"{name} {row[name]['median_s']:.2f}s [Q1={row[name]['q1_s']:.2f}, Q3={row[name]['q3_s']:.2f}] "
            f"({row[name]['output_tokens']} tok)" for name in methods))
    return out


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


# ---------------------------------------------------------------------------
# Calibration on BoolQ (passage + yes/no question with a ground-truth answer)


def stratified_sample(rows: list[dict], n: int, label_key: str, seed: int) -> list[dict]:
    """Stratified sampling that preserves label distribution (W15).

    Groups rows by label, samples proportionally from each group, and ensures the
    final sample maintains the original label distribution.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    # Group by label
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[label_key]].append(row)

    # Calculate proportional sample sizes
    total = len(rows)
    samples = []
    for label, group in sorted(by_label.items()):
        group_size = len(group)
        # Sample proportionally, ensuring at least 1 from each label if possible
        n_from_group = max(1, round(n * group_size / total))
        n_from_group = min(n_from_group, group_size)  # Can't sample more than available
        sampled = rng.sample(group, n_from_group)
        samples.extend(sampled)

    # If we have too many (due to rounding), randomly drop some
    if len(samples) > n:
        samples = rng.sample(samples, n)

    # Shuffle to avoid label ordering
    rng.shuffle(samples)
    return samples


def boolq(n: int, seed: int = 0, verify: bool = True) -> list[dict]:
    """A seeded stratified sample of BoolQ validation (3,270 rows), via the Hugging Face datasets-server API.

    Uses stratified sampling to preserve the label distribution of true/false answers (W15).
    """
    path = DATA / "boolq_validation.json"
    if not path.exists():
        rows, offset = [], 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=google/boolq&config=default"
                   f"&split=validation&offset={offset}&length=100")
            page = json.loads(fetch(url, DATA / f"boolq_page_{offset}.json").read_text())
            rows += [r["row"] for r in page["rows"]]
            offset += 100
            if offset >= page["num_rows_total"]:
                break
        path.write_text(json.dumps(rows))
    rows = json.loads(path.read_text())
    if verify and MANIFEST_PATH.exists():
        _verify_dataset_checksums("boolq", rows)
    return stratified_sample(rows, n, "answer", seed)


def binary_metrics(z: list[float], y: list[int]) -> dict:
    """z = log-odds of 'yes'. ECE uses 10 equal-width bins on the predicted class's confidence."""
    p = [sigmoid(v) for v in z]
    eps = 1e-12
    bins = [[0, 0.0, 0.0] for _ in range(10)]  # count, sum confidence, sum correct
    for pi, yi in zip(p, y):
        conf, correct = max(pi, 1 - pi), int((pi > 0.5) == bool(yi))
        b = min(int((conf - 0.5) / 0.05), 9) if conf >= 0.5 else 0  # top-label confidence lives in [0.5, 1]
        bins[b][0] += 1
        bins[b][1] += conf
        bins[b][2] += correct
    n = len(y)
    return {
        "accuracy": sum(int((pi > 0.5) == bool(yi)) for pi, yi in zip(p, y)) / n,
        "ece": sum(abs(c[1] - c[2]) for c in bins if c[0]) / n,
        "brier": sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n,
        "nll": -sum(math.log(max(pi if yi else 1 - pi, eps)) for pi, yi in zip(p, y)) / n,
        "mean_p_yes": sum(p) / n,
        "reliability": [{"conf_bin": f"{0.5 + 0.05 * i:.2f}-{0.55 + 0.05 * i:.2f}", "n": c[0],
                         "mean_conf": c[1] / c[0] if c[0] else None, "accuracy": c[2] / c[0] if c[0] else None}
                        for i, c in enumerate(bins)],
    }


def prompt_fingerprint(engine: Engine) -> str:
    """A hash of every fixed prompt piece: system line, chat template split, and the Noul block."""
    probe = noul_block({"instructions": "\x00"})
    return hashlib.sha1("\x01".join([SYSTEM, engine._head, engine._tail, probe]).encode()).hexdigest()[:12]


def calibration(engine: Engine, n: int = 400) -> dict:
    # Model outputs are cached before any fitting, so changing the analysis never costs a rerun.
    # The cache is reused only if the model, n and every prompt piece are unchanged; else the model runs again.
    key = {"model": engine.name, "n": n, "prompt_sha1": prompt_fingerprint(engine)}
    cache = RESULTS / f"calibration_logodds{tag(engine.name)}.json"
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get("key") == key:
        print(f"reusing model outputs from {cache.name} (same model, n and prompt); delete it to run the model again")
        z_raw, z_cf, y = saved["raw"], saved["content_free"], saved["labels"]
    else:
        z_raw, z_cf, y = [], [], []
        t0 = time.perf_counter()
        for i, ex in enumerate(boolq(n)):
            question = ex["question"][0].upper() + ex["question"][1:] + "?"
            q = {"type": "noul", "instructions": question}
            raw, _ = raw_scores(engine, {"state": ex["passage"], "questions": {"q": q}}, "packed")
            cf, _ = raw_scores(engine, {"state": CONTENT_FREE_STATE, "questions": {"q": q}}, "packed")
            z_raw.append(raw["q"]["logits"][0] - raw["q"]["logits"][1])
            z_cf.append(cf["q"]["logits"][0] - cf["q"]["logits"][1])
            y.append(int(ex["answer"]))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        cache.write_text(json.dumps({"key": key, "raw": z_raw, "content_free": z_cf, "labels": y}))
    z_cc = [a - b for a, b in zip(z_raw, z_cf)]  # contextual calibration: divide out the content-free prior
    variants = {
        "raw": z_raw,
        "contextual": z_cc,
        "temperature (2-fold)": cross_fit(z_raw, y, slope_only=True),
        "contextual + temperature (2-fold)": cross_fit(z_cc, y, slope_only=True),
        "Platt a*z+b (2-fold)": cross_fit(z_raw, y, slope_only=False),
    }
    table = {name: binary_metrics(z, y) for name, z in variants.items()}
    for name, z in variants.items():  # evaluation noise only: the 2-fold fits are held fixed
        pairs = list(zip(z, y))
        for k in ("accuracy", "ece"):
            table[name][f"{k}_ci95"] = bootstrap_ci(pairs, lambda s, k=k: binary_metrics(
                [a for a, _ in s], [b for _, b in s])[k], n_boot=1000)
    print(f"BoolQ dev, n={n}, base rate yes={sum(y) / n:.3f}. [..] = 95% bootstrap interval")
    print(f"{'variant':36s} {'acc':>6s} {'ECE':>6s} {'ECE CI':>13s} {'Brier':>6s} {'NLL':>6s} {'mean p':>7s}")
    for name, m in table.items():
        ci = m["ece_ci95"]
        print(f"{name:36s} {m['accuracy']:6.3f} {m['ece']:6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {m['brier']:6.3f} "
              f"{m['nll']:6.3f} {m['mean_p_yes']:7.3f}")
    T_full = 1 / fit_affine(z_raw, y, slope_only=True)[0]
    a, b = fit_affine(z_raw, y, slope_only=False)
    print(f"fitted on all {n}: temperature T = {T_full:.2f};  Platt a = {a:.3f}, b = {b:.3f}")
    print(f"to use them, set in poc/minijev.env either  MINIJEV_TEMP_NOUL={T_full:.2f}  (temperature)\n"
          f"  or  MINIJEV_TEMP_NOUL={1 / a:.2f} and MINIJEV_BIAS_NOUL={b:.3f}  (Platt)")
    return {"n": n, "base_rate_yes": sum(y) / n, "temperature_all": T_full, "platt_all": [a, b],
            "metrics": table}


# ---------------------------------------------------------------------------
# E11: quality on labelled data. The same model answers the same questions in four ways.


def binary_entry(z: list[float], y: list[int], seconds: list[float], out_tokens: float, fails: int = 0,
                 has_probs: bool = True) -> dict:
    """binary_metrics plus 95% bootstrap intervals for accuracy and ECE, and the time per question."""
    pairs = list(zip(z, y))
    m = binary_metrics(z, y)
    m.pop("reliability")
    for k in ("accuracy", "ece"):
        m[f"{k}_ci95"] = bootstrap_ci(pairs, lambda s, k=k: binary_metrics([a for a, _ in s], [b for _, b in s])[k], n_boot=1000)
    if not has_probs:
        m["ece"] = m["ece_ci95"] = m["brier"] = m["nll"] = None
    return {**m, "n": len(y), "parse_failures": fails, **mean_ci(seconds), "output_tokens": out_tokens, "has_probs": has_probs}


def multiclass_entry(probs: list[list[float]], y: list[int], seconds: list[float], out_tokens: float,
                     fails: int = 0, has_probs: bool = True) -> dict:
    m = with_ci(probs, y)
    if not has_probs:
        m["ece"] = m["ece_ci95"] = m["brier"] = None
    return {**m, "n": len(y), "parse_failures": fails, **mean_ci(seconds), "output_tokens": out_tokens, "has_probs": has_probs}


def quality(engine: Engine, n_boolq: int = 500, n_ag: int = 300) -> dict:
    """E11. BoolQ (yes/no) and AG News (4 topics), each question answered four ways by the same model:
    readout; readout with 2-fold temperature scaling; the model writes its answer; the model writes a probability.
    Unparsed replies count as wrong, and get an uninformative probability (0.5, or uniform) for ECE and Brier.

    Sample sizes expanded in W15: BoolQ 200→500, AG News 120→300 to narrow confidence intervals.
    """
    generate(engine, "warm up", 2)
    out = {}

    # ---- BoolQ: a yes/no question about a passage (Noul)
    y, z_read, t_read, a_gen, t_gen, k_gen, p_verb, t_verb, k_verb = [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    for i, ex in enumerate(boolq(n_boolq)):
        question = ex["question"][0].upper() + ex["question"][1:] + "?"
        y.append(int(ex["answer"]))
        t0 = time.perf_counter()
        raw, _ = raw_scores(engine, {"state": ex["passage"], "questions": {"q": {"type": "noul", "instructions": question}}})
        t_read.append(time.perf_counter() - t0)
        z_read.append(raw["q"]["logits"][0] - raw["q"]["logits"][1])
        head = f"STATE:\n{ex['passage']}\n\nQUESTION: {question}\n"
        g = generate(engine, head + "Answer with Yes or No.", 4)
        word = g["text"].strip().strip(".!*\"'").lower()
        a_gen.append(1 if word.startswith("yes") else 0 if word.startswith("no") else None)
        t_gen.append(g["seconds"]); k_gen.append(g["new_tokens"])
        g = generate(engine, head + "How likely is it that the answer is yes? Reply with only a probability between 0 and 1.", 8)
        p_verb.append(verbal_probability(g["text"]))
        t_verb.append(g["seconds"]); k_verb.append(g["new_tokens"])
        if (i + 1) % 50 == 0:
            print(f"  BoolQ {i + 1}/{n_boolq}  ({time.perf_counter() - t_start:.0f}s)", flush=True)
    wrong = lambda yi: logit(0.01) if yi else logit(0.99)  # an unparsed answer counts as wrong
    z_gen = [logit(0.99) if a == 1 else logit(0.01) if a == 0 else wrong(yi) for a, yi in zip(a_gen, y)]
    z_verb = [logit(p) if p is not None else 0.0 for p in p_verb]
    verb_correct = [p is not None and (p > 0.5) == bool(yi) for p, yi in zip(p_verb, y)]
    methods = {
        "readout": binary_entry(z_read, y, t_read, 0),
        "readout, calibrated": binary_entry(cross_fit(z_read, y, slope_only=True), y, t_read, 0),
        "writes the answer": binary_entry(z_gen, y, t_gen, statistics.mean(k_gen), sum(a is None for a in a_gen), has_probs=False),
        "writes a probability": binary_entry(z_verb, y, t_verb, statistics.mean(k_verb), sum(p is None for p in p_verb)),
    }
    methods["writes a probability"]["accuracy"] = sum(verb_correct) / len(y)  # unparsed = wrong, not a coin flip
    methods["writes a probability"]["accuracy_ci95"] = bootstrap_ci(verb_correct, statistics.mean)
    out["boolq"] = {"n": n_boolq, "base_rate": sum(y) / len(y), "chance": max(sum(y), len(y) - sum(y)) / len(y),
                    "methods": methods,
                    "rows": [{"y": yi, "p_readout": sigmoid(zr), "generated": a, "verbal": pv}
                             for yi, zr, a, pv in zip(y, z_read, a_gen, p_verb)]}

    # ---- AG News: pick the topic of a news article (Choice)
    keys = list(AG_OPTIONS)
    q = {"type": "choice", "instructions": AG_QUESTION, "criteria": AG_OPTIONS}
    ask_probs = ("Reply with a JSON object that maps every option name to the probability that it is the right "
                 "answer. The probabilities must add up to 1. Reply with the JSON object only.")
    y, logits, t_read, a_gen, t_gen, k_gen, p_verb, t_verb, k_verb = [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    for i, ex in enumerate(ag_news(n_ag)):
        y.append(ex["label"])
        t0 = time.perf_counter()
        raw, _ = raw_scores(engine, {"state": ex["text"], "questions": {"q": q}})
        t_read.append(time.perf_counter() - t0)
        logits.append(raw["q"]["logits"])
        head = f"STATE:\n{ex['text']}\n\nQUESTION: {AG_QUESTION}\nOPTIONS:\n{options_text(AG_OPTIONS)}\n"
        g = generate(engine, head + "Answer with the name of the best option only.", 10)
        a_gen.append(match_option(g["text"], keys))
        t_gen.append(g["seconds"]); k_gen.append(g["new_tokens"])
        g = generate(engine, head + ask_probs, 16 * len(keys) + 16)
        p_verb.append(parse_probs_lenient(g["text"], keys))
        t_verb.append(g["seconds"]); k_verb.append(g["new_tokens"])
        if (i + 1) % 30 == 0:
            print(f"  AG News {i + 1}/{n_ag}  ({time.perf_counter() - t_start:.0f}s)", flush=True)
    uniform = [1 / len(keys)] * len(keys)
    onehot = lambda a, yi: [1.0 if k == a else 0.0 for k in keys] if a else [1.0 if j == (yi + 1) % len(keys) else 0.0 for j in range(len(keys))]
    methods = {
        "readout": multiclass_entry([softmax_list(z) for z in logits], y, t_read, 0),
        "readout, calibrated": multiclass_entry(cross_fit_multiclass(logits, y), y, t_read, 0),
        "writes the answer": multiclass_entry([onehot(a, yi) for a, yi in zip(a_gen, y)], y, t_gen, statistics.mean(k_gen),
                                              sum(a is None for a in a_gen), has_probs=False),
        "writes a probability": multiclass_entry([p if p else uniform for p in p_verb], y, t_verb, statistics.mean(k_verb),
                                                 sum(p is None for p in p_verb)),
    }
    verb_correct = [p is not None and max(range(len(keys)), key=p.__getitem__) == yi for p, yi in zip(p_verb, y)]
    methods["writes a probability"]["accuracy"] = sum(verb_correct) / len(y)
    methods["writes a probability"]["accuracy_ci95"] = bootstrap_ci(verb_correct, statistics.mean)
    out["ag_news"] = {"n": n_ag, "chance": 0.25, "methods": methods,
                      "rows": [{"y": yi, "p_readout": softmax_list(z), "generated": a, "verbal": pv}
                               for yi, z, a, pv in zip(y, logits, a_gen, p_verb)]}

    for task, r in out.items():
        print(f"\n{task} (n={r['n']}). [..] = 95% bootstrap interval")
        print(f"{'method':24s} {'acc':>6s} {'acc CI':>13s} {'ECE':>6s} {'fails':>6s} {'s/q':>6s} {'out tok':>8s}")
        for name, m in r["methods"].items():
            ece = f"{m['ece']:6.3f}" if m["ece"] is not None else f"{'—':>6s}"
            ci = m["accuracy_ci95"]
            print(f"{name:24s} {m['accuracy']:6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {ece} {m['parse_failures']:6d} "
                  f"{m['mean_s']:6.2f} {m['output_tokens']:8.1f}")
    return {"tasks": out}


def same_format(engine: Engine, repeats: int = 5) -> dict:
    """E12: the same request answered as minijev's JSON, read out vs written by the same model.

    Uses improved timing measurement: warm-up + median + IQR over 5 runs.
    """
    shoes = {"state": SHOES, "questions": {label.split(": ")[1].replace(" ", "_"): q
                                           for label, state, q, _ in JEV_DOC_CASES if state == SHOES}}
    # Warm-up: 3 runs
    for _ in range(3):
        generate(engine, "warm up", 2)

    out = {}
    for name, body in (("support_ticket", SUPPORT_TICKET), ("jev_shoes_5_choices", shoes)):
        runs = [same_format_run(engine, body) for _ in range(repeats)]
        r, w = [x["readout"]["seconds"] for x in runs], [x["written"]["seconds"] for x in runs]
        st = [x["structured"]["seconds"] for x in runs]
        last = runs[-1]

        # Compute median and IQR for each method
        r_median, r_q1, r_q3 = statistics.median(r), statistics.quantiles(r, n=4)[0], statistics.quantiles(r, n=4)[2]
        w_median, w_q1, w_q3 = statistics.median(w), statistics.quantiles(w, n=4)[0], statistics.quantiles(w, n=4)[2]
        st_median, st_q1, st_q3 = statistics.median(st), statistics.quantiles(st, n=4)[0], statistics.quantiles(st, n=4)[2]

        out[name] = {"questions": len(body["questions"]),
                     "readout_median_s": r_median, "readout_q1_s": r_q1, "readout_q3_s": r_q3, "readout_iqr_s": r_q3 - r_q1, "readout_all": r,
                     "written_median_s": w_median, "written_q1_s": w_q1, "written_q3_s": w_q3, "written_iqr_s": w_q3 - w_q1, "written_all": w,
                     "written_tokens": [x["written"]["output_tokens"] for x in runs],
                     "usable": [x["written"]["usable"] for x in runs], "valid_json": [x["written"]["valid_json"] for x in runs],
                     "agree": [sum(c["agree"] for c in x["compare"].values()) for x in runs],
                     "structured_median_s": st_median, "structured_q1_s": st_q1, "structured_q3_s": st_q3, "structured_iqr_s": st_q3 - st_q1, "structured_all": st,
                     "structured_decided_tokens": [x["structured"]["output_tokens"] for x in runs],
                     "structured_forced_tokens": [x["structured"]["forced_tokens"] for x in runs],
                     "structured_usable": [x["structured"]["usable"] for x in runs],
                     "structured_agree": [sum(c["agree"] for c in x["compare_structured"].values()) for x in runs],
                     "example": last}
        print(f"{name}: readout {r_median:.2f}s [Q1={r_q1:.2f}, Q3={r_q3:.2f}] | "
              f"written {w_median:.2f}s [Q1={w_q1:.2f}, Q3={w_q3:.2f}] ({w_median / r_median:.1f}x), "
              f"tokens {out[name]['written_tokens']}, usable {out[name]['usable']}/{len(body['questions'])}, "
              f"agree {out[name]['agree']} | structured {st_median:.2f}s [Q1={st_q1:.2f}, Q3={st_q3:.2f}], "
              f"decided {out[name]['structured_decided_tokens']}, usable {out[name]['structured_usable']}, "
              f"agree {out[name]['structured_agree']}")
    return out


# ---------------------------------------------------------------------------
# E13: the order flaw. A listwise Choice can prefer a letter position over the content. Three fixes, measured.


def order_bias(engine: Engine, n: int = 300, orders_per_item: int = 4, seed: int = 0) -> dict:
    """E13: AG News, each article asked in several option orders (the original plus random shuffles).

    One packed pass per (article, order) holds three questions: listwise (1 branch), averaged over every rotation
    (k branches) and pointwise (k branches). A fourth method, debiased, divides the listwise probabilities by the
    model's measured liking for each letter position (fitted on the other half of the articles; PriDe-style).

    Sample size expanded in W15: 120→300 to narrow confidence intervals.
    """
    keys, k = list(AG_OPTIONS), len(AG_OPTIONS)
    rng = random.Random(seed)
    rows = []
    t_start = time.perf_counter()
    for i, ex in enumerate(ag_news(n)):
        orders = [list(range(k))] + [rng.sample(range(k), k) for _ in range(orders_per_item - 1)]
        for order in orders:
            crit = {keys[j]: AG_OPTIONS[keys[j]] for j in order}
            q = {"type": "choice", "instructions": AG_QUESTION, "criteria": crit}
            qs = {m: {**q, "choice_mode": m} for m in ("listwise", "averaged", "pointwise")}
            raw, _ = raw_scores(engine, {"state": ex["text"], "questions": qs})
            canon = lambda p: [p[order.index(c)] for c in range(k)]  # back to the fixed option order
            pos = softmax_list(raw["listwise"]["logits"])  # by letter position
            rows.append({"item": i, "y": ex["label"], "order": order, "pos": pos,
                         "listwise": canon(pos),
                         "averaged": canon(softmax_list(raw["averaged"]["logits"])),
                         "pointwise": canon(softmax_list(raw["pointwise"]["logits"]))})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  ({time.perf_counter() - t_start:.0f}s)", flush=True)

    # Debiased: the average probability per letter position, over articles whose options were shuffled, is the
    # model's liking for that position (the content averages out). Fit on one half of the articles, apply to the other.
    half = n // 2
    for fit, apply in ((range(half, n), range(half)), (range(half), range(half, n))):
        fit_rows = [r for r in rows if r["item"] in fit]
        prior = [statistics.mean(r["pos"][j] for r in fit_rows) for j in range(k)]
        for r in rows:
            if r["item"] in apply:
                d = [r["pos"][j] / prior[j] for j in range(k)]
                d = [v / sum(d) for v in d]
                r["debiased"] = [d[r["order"].index(c)] for c in range(k)]
    liking = [statistics.mean(r["pos"][j] for r in rows) for j in range(k)]
    picks = [statistics.mean(max(range(k), key=r["pos"].__getitem__) == j for r in rows) for j in range(k)]
    truth = [statistics.mean(r["order"].index(r["y"]) == j for r in rows) for j in range(k)]

    methods = {}
    for m, branches in (("listwise", 1), ("debiased", 1), ("averaged", k), ("pointwise", k)):
        by_item: dict = {}
        for r in rows:
            by_item.setdefault(r["item"], []).append(r)
        winners = {i: [max(range(k), key=r[m].__getitem__) for r in rs] for i, rs in by_item.items()}
        flips = [len(set(w)) > 1 for w in winners.values()]
        spread = [max(max(r[m][c] for r in rs) - min(r[m][c] for r in rs) for c in range(k)) for rs in by_item.values()]
        item_acc = [statistics.mean(w == rs[0]["y"] for w in winners[i]) for i, rs in by_item.items()]
        met = multiclass_metrics([r[m] for r in rows], [r["y"] for r in rows])
        ece_ci = bootstrap_ci(list(by_item.values()), lambda s: multiclass_metrics(
            [r[m] for rs in s for r in rs], [r["y"] for rs in s for r in rs])["ece"], n_boot=500)
        methods[m] = {"branches": branches, "flip_rate": statistics.mean(flips),
                      "flip_rate_ci95": bootstrap_ci(flips, statistics.mean),
                      "mean_max_dp": statistics.mean(spread),
                      "accuracy": statistics.mean(item_acc), "accuracy_ci95": bootstrap_ci(item_acc, statistics.mean),
                      "ece": met["ece"], "ece_ci95": ece_ci}

    print(f"\nAG News, n={n} articles x {orders_per_item} option orders. [..] = 95% bootstrap interval")
    print("position:            " + "  ".join(f"{'ABCD'[j]:>6s}" for j in range(k)))
    print("mean probability:    " + "  ".join(f"{v:6.3f}" for v in liking))
    print("model picks:         " + "  ".join(f"{v:6.3f}" for v in picks))
    print("right answer is at:  " + "  ".join(f"{v:6.3f}" for v in truth))
    print(f"{'method':10s} {'branches':>8s} {'flips':>6s} {'flips CI':>13s} {'max dp':>7s} {'acc':>6s} {'acc CI':>13s} {'ECE':>6s}")
    for m, r in methods.items():
        print(f"{m:10s} {r['branches']:8d} {r['flip_rate']:6.3f} [{r['flip_rate_ci95'][0]:.2f},{r['flip_rate_ci95'][1]:.2f}] "
              f"{r['mean_max_dp']:7.3f} {r['accuracy']:6.3f} [{r['accuracy_ci95'][0]:.2f},{r['accuracy_ci95'][1]:.2f}] {r['ece']:6.3f}")
    return {"n": n, "orders_per_item": orders_per_item,
            "position": {"mean_probability": liking, "model_picks": picks, "right_answer_at": truth},
            "methods": methods}


# ---------------------------------------------------------------------------
# E14: held-out calibration and evaluation on frozen splits (poc/data.py, datasets/splits_v2.json).
# train: fit temperatures. val: choose the Choice mode and the Noul calibrator. test: report only.

CALIBRATION = HERE / "calibration"
CHOICE_MODES = ("listwise", "averaged", "pointwise")


def readout_dir(engine: Engine) -> Path:
    name = tag(engine.name).lstrip('-') or 'Qwen2.5-0.5B-Instruct'
    return RESULTS / "readouts" / (f"{name}-lora-{engine.adapter_sha256[:8]}" if engine.adapter else name)


def readout_cache(engine: Engine, dataset: str, split: str) -> list[dict]:
    """Raw readout logits for one split, cached. A readout never uses the labels, so computing it for val and test
    is not tuning on them. The cache is keyed by model, prompt fingerprint and splits file."""
    import data
    key = {"model": engine.name, "prompt_sha1": prompt_fingerprint(engine), "choice_block_sha1":
           hashlib.sha1(choice_block({"instructions": "?", "criteria": AG_OPTIONS}).encode()).hexdigest()[:12],
           "splits_sha256": data.splits_fingerprint(),
           # AG News options are shown in a per-item random order (seeded by the source index), so that the one
           # order-sensitive mode, listwise, is judged under ordinary orders and not flattered by one fixed order.
           **({"option_order": "per-item shuffle, random.Random(source_index)"} if dataset == "ag_news" else {})}
    if engine.adapter:  # a fine-tuned model is a different model: never reuse the base readouts
        key["adapter_sha256"] = engine.adapter_sha256
    path = readout_dir(engine) / f"{dataset}_{split}.json"
    rows = data.load_split(dataset, split)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["key"] == key:
            return saved["items"]
    items, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        if dataset == "boolq":
            q = {"type": "noul", "instructions": r["question"][0].upper() + r["question"][1:] + "?"}
            raw, _ = raw_scores(engine, {"state": r["passage"], "questions": {"q": q}})
            items.append({"source_index": r["source_index"], "y": r["y"], "z": raw["q"]["logits"][0] - raw["q"]["logits"][1]})
        else:
            keys = list(AG_OPTIONS)
            order = random.Random(r["source_index"]).sample(range(len(keys)), len(keys))
            q = {"type": "choice", "instructions": AG_QUESTION, "criteria": {keys[j]: AG_OPTIONS[keys[j]] for j in order}}
            raw, _ = raw_scores(engine, {"state": r["text"], "questions": {m: {**q, "choice_mode": m} for m in CHOICE_MODES}})
            canon = lambda z: [z[order.index(c)] for c in range(len(keys))]  # back to the fixed label order
            items.append({"source_index": r["source_index"], "y": r["y"], "order": order,
                          **{m: canon(raw[m]["logits"]) for m in CHOICE_MODES},
                          "mass": min(min(raw[m]["mass"]) for m in CHOICE_MODES)})
        if (i + 1) % 100 == 0:
            print(f"  {dataset}/{split} {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "items": items}))
    return items


def heldout(engine: Engine) -> dict:
    """E14. Fit on train, select on val, report on test. Writes calibration/<model>.json with its provenance."""
    import data
    data.ACCESS.clear()
    R = {(d, s): readout_cache(engine, d, s) for d in ("boolq", "ag_news") for s in ("train", "val", "test")}

    # ---- fit on train
    bq = R[("boolq", "train")]
    zt, yt = [it["z"] for it in bq], [it["y"] for it in bq]
    a_t, _ = fit_affine(zt, yt, slope_only=True)
    a_p, b_p = fit_affine(zt, yt, slope_only=False)
    ag = R[("ag_news", "train")]
    choice_t = {m: fit_temperature_multiclass([it[m] for it in ag], [it["y"] for it in ag]) for m in CHOICE_MODES}

    # ---- select on val (lowest negative log-likelihood: a proper scoring rule)
    bv = R[("boolq", "val")]
    zv, yv = [it["z"] for it in bv], [it["y"] for it in bv]
    noul_val = {"temperature": nll(zv, yv, a_t, 0.0), "platt": nll(zv, yv, a_p, b_p)}
    noul_pick = min(noul_val, key=noul_val.get)
    av = R[("ag_news", "val")]
    choice_val = {m: nll_multi([it[m] for it in av], [it["y"] for it in av], choice_t[m]) for m in CHOICE_MODES}
    choice_pick = min(choice_val, key=choice_val.get)

    # ---- report on test
    bt = R[("boolq", "test")]
    zs, ys = [it["z"] for it in bt], [it["y"] for it in bt]
    noul_test = {
        "raw": binary_entry(zs, ys, [0.0] * len(ys), 0),
        "temperature": binary_entry([a_t * v for v in zs], ys, [0.0] * len(ys), 0),
        "platt": binary_entry([a_p * v + b_p for v in zs], ys, [0.0] * len(ys), 0),
    }
    for m in noul_test.values():
        for k in ("mean_s", "mean_s_ci95", "output_tokens", "parse_failures", "has_probs"):
            m.pop(k, None)
    at = R[("ag_news", "test")]
    yat = [it["y"] for it in at]
    choice_test = {}
    for m in CHOICE_MODES:
        raw_p = [softmax_list(it[m]) for it in at]
        cal_p = [softmax_list([v / choice_t[m] for v in it[m]]) for it in at]
        choice_test[m] = {"raw": with_ci(raw_p, yat), "calibrated": with_ci(cal_p, yat),
                          "nll_raw": nll_multi([it[m] for it in at], yat), "nll_calibrated": nll_multi([it[m] for it in at], yat, choice_t[m])}

    tagname = engine.name.split("/")[-1]
    cal = {
        "model": engine.name, "created": time.strftime("%Y-%m-%d"),
        "provenance": {"splits_file": "datasets/splits_v2.json", "splits_sha256": data.splits_fingerprint(),
                       "fitted_on": {"noul": {"dataset": "boolq", "split": "train", "n": len(bq)},
                                     "choice": {"dataset": "ag_news", "split": "train", "n": len(ag)}},
                       "selected_on": {"split": "val", "rule": "lowest negative log-likelihood"},
                       "prompt_sha1": prompt_fingerprint(engine), "experiment": "E14 heldout"},
        "noul": {"temperature": 1 / a_t, "platt": {"a": a_p, "b": b_p}, "selected": noul_pick},
        "choice": {"temperature": choice_t, "selected_mode": choice_pick},
        "score": None,  # no labelled ordinal data yet: Score answers are uncalibrated
    }
    CALIBRATION.mkdir(exist_ok=True)
    (CALIBRATION / f"{tagname}.json").write_text(json.dumps(cal, indent=1))

    print(f"\n{engine.name}: fitted on train, selected on val, reported on test")
    print(f"Noul (BoolQ): T = {1 / a_t:.2f}; Platt a = {a_p:.3f}, b = {b_p:.3f}; val picks {noul_pick}")
    for k, m in noul_test.items():
        print(f"  test {k:12s} acc {m['accuracy']:.3f} [{m['accuracy_ci95'][0]:.2f},{m['accuracy_ci95'][1]:.2f}]  "
              f"ECE {m['ece']:.3f} [{m['ece_ci95'][0]:.2f},{m['ece_ci95'][1]:.2f}]  NLL {m['nll']:.3f}")
    print("Choice (AG News): T = " + ", ".join(f"{m} {t:.2f}" for m, t in choice_t.items()) + f"; val picks {choice_pick}")
    for m, r in choice_test.items():
        c, rw = r["calibrated"], r["raw"]
        print(f"  test {m:10s} acc {c['accuracy']:.3f} [{c['accuracy_ci95'][0]:.2f},{c['accuracy_ci95'][1]:.2f}]  "
              f"ECE raw {rw['ece']:.3f} -> {c['ece']:.3f} [{c['ece_ci95'][0]:.2f},{c['ece_ci95'][1]:.2f}]  "
              f"NLL {r['nll_raw']:.3f} -> {r['nll_calibrated']:.3f}")
    return {"calibration": cal, "val": {"noul_nll": noul_val, "choice_nll": choice_val},
            "test": {"noul": noul_test, "choice": choice_test},
            "base_rates": {"boolq_test_yes": sum(ys) / len(ys), "ag_news_test_per_topic": len(yat) // 4},
            "splits_accessed": list(data.ACCESS),
            "use_of_splits": {"train": "fit temperatures", "val": "choose Noul calibrator and Choice mode",
                              "test": "report only; no value was chosen from it"}}


# ---------------------------------------------------------------------------

# E15: how much the answers depend on the template wording (W7, PLAN Task 4.3). A full grid of 3 wordings for each of
# 4 template parts = 81 templates. Level 0 of every part is the production template. BoolQ val only: this measures,
# it chooses nothing, and the test split stays unread.
TEMPLATE_PARTS = {
    "system": [SYSTEM, "You are a helpful assistant.", "Read the text below and answer the question about it."],
    "state_label": ["STATE:", "TEXT:", "Passage:"],
    "question_label": ["QUESTION:", "Question:", "Q:"],
    "answer_line": ["Answer with Yes or No.", "Reply with Yes or No only.", "Is the answer Yes or No?"],
}


def template_sensitivity(engine: Engine, n: int = 200) -> dict:
    import itertools

    import data
    parts = TEMPLATE_PARTS
    with data.tuning():  # a PermissionError if anything here read the test split
        rows = data.load_split("boolq", "val")[:n]
    sentinel = "\x00SPLIT\x00"
    heads = []
    for system in parts["system"]:
        text = engine.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": sentinel}],
                                              tokenize=False, add_generation_prompt=True)
        head, tail = text.split(sentinel)
        assert tail == engine._tail
        heads.append(head)
    grid = list(itertools.product(*(range(len(v)) for v in parts.values())))  # (system, state, question, answer)
    key = {"model": engine.name, "parts": parts, "n": n, "splits_sha256": data.splits_fingerprint()}
    path = RESULTS / "readouts" / (tag(engine.name).lstrip("-") or "Qwen2.5-0.5B-Instruct") / "template_grid_boolq_val.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("key") == key:
        z, mass = saved["z"], saved["mass"]
    else:
        z = {str(g): [] for g in grid}
        mass = {str(g): [] for g in grid}
        tails = list(itertools.product(range(3), range(3)))
        t0 = time.perf_counter()
        for i, r in enumerate(rows):
            q = r["question"][0].upper() + r["question"][1:] + "?"
            branches = [Branch(engine.suffix_ids(f"{parts['question_label'][qi]} {q}\n{parts['answer_line'][ai]}"),
                               [engine.yes, engine.no], "q") for qi, ai in tails]
            for si, head in enumerate(heads):
                for li, label in enumerate(parts["state_label"]):
                    prefix = engine.tok.encode(f"{head}{label}\n{r['passage']}\n\n", add_special_tokens=False)
                    for (qi, ai), row in zip(tails, engine.readouts(prefix, branches, "packed")):
                        (yes, no), m = class_logits(row, [engine.yes, engine.no])
                        z[str((si, li, qi, ai))].append(yes - no)
                        mass[str((si, li, qi, ai))].append(m)
            if (i + 1) % 25 == 0:
                print(f"  template grid {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "z": z, "mass": mass}))
    # The production template must give the same log-odds as ask() does.
    base = str((0, 0, 0, 0))
    q0 = {"type": "noul", "instructions": rows[0]["question"][0].upper() + rows[0]["question"][1:] + "?"}
    ref = raw_scores(engine, {"state": rows[0]["passage"], "questions": {"q": q0}})[0]["q"]["logits"]
    assert abs((ref[0] - ref[1]) - z[base][0]) < 1e-3, "grid level 0 is not the production template"

    y = [r["y"] for r in rows]
    decide = lambda zs: [v > 0 for v in zs]
    base_dec = decide(z[base])
    variants = []
    for g in grid:
        k = str(g)
        m = binary_metrics(z[k], y)
        variants.append({"parts": dict(zip(parts, g)), "accuracy": m["accuracy"], "ece": m["ece"], "nll": m["nll"],
                         "mean_p_yes": m["mean_p_yes"], "min_mass": min(mass[k]),
                         "median_mass": statistics.median(mass[k]),
                         "flip_rate": sum(a != b for a, b in zip(decide(z[k]), base_dec)) / len(y)})
    accs = [v["accuracy"] for v in variants]
    base_acc = variants[0]["accuracy"]  # grid[0] is (0, 0, 0, 0)
    effects = {part: [{"level": j, "text": text,
                       **{f"mean_{m}": statistics.mean(v[m] for v in variants if v["parts"][part] == j)
                          for m in ("accuracy", "ece", "flip_rate", "mean_p_yes")}}
                      for j, text in enumerate(parts[part])] for part in parts}
    decisions = [decide(z[str(g)]) for g in grid]
    unstable = sum(len({d[i] for d in decisions}) > 1 for i in range(len(y))) / len(y)
    return {
        "design": "full grid, 3 wordings x 4 template parts = 81 templates; level 0 = production template",
        "data": {"dataset": "boolq", "split": "val", "n": len(y), "test_read": False},
        "parts": parts,
        "base": next(v for v in variants if not any(v["parts"].values())),
        "spread": {"accuracy_min": min(accs), "accuracy_max": max(accs), "accuracy_sd": statistics.pstdev(accs),
                   "accuracy_ci95_halfwidth_one_template": 1.96 * math.sqrt(base_acc * (1 - base_acc) / len(y)),
                   "flip_rate_mean": statistics.mean(v["flip_rate"] for v in variants),
                   "flip_rate_max": max(v["flip_rate"] for v in variants),
                   "items_that_change_under_some_template": unstable},
        "effects": effects,
        "variants": variants,
    }


# E16: contrastive Score levels (W5, PLAN Task 4.4) on SST-5, a labelled 5-level set (docs/DATA.md). Each pointwise
# level can name its neighbours: "positive (not neutral; not very positive)". Chosen on val, reported on test.
SCORE_VARIANTS = {"pointwise": {"score_mode": "pointwise"},
                  "contrastive": {"score_mode": "pointwise", "contrastive": True},
                  "listwise": {"score_mode": "listwise"}}


def ordinal_metrics(probs: list[list[float]], y: list[int]) -> dict:
    """Exact accuracy (most probable level), adjacent (off by 1) and far (off by 2 or more) error rates, mean absolute
    error of the expected level, NLL of the true level, and the confusion matrix (rows: true level)."""
    k = len(probs[0])
    pred = [max(range(k), key=p.__getitem__) for p in probs]
    conf = [[0] * k for _ in range(k)]
    for t, pr in zip(y, pred):
        conf[t][pr] += 1
    n = len(y)
    return {"accuracy": sum(a == b for a, b in zip(pred, y)) / n,
            "adjacent_error": sum(abs(a - b) == 1 for a, b in zip(pred, y)) / n,
            "far_error": sum(abs(a - b) >= 2 for a, b in zip(pred, y)) / n,
            "mae_expected": sum(abs(sum(i * pi for i, pi in enumerate(p)) - t) for p, t in zip(probs, y)) / n,
            "nll": -sum(math.log(max(p[t], 1e-12)) for p, t in zip(probs, y)) / n,
            "mean_level_predicted": sum(pred) / n, "confusion": conf}


def score_readouts(engine: Engine, split: str) -> list[dict]:
    """Raw Score logits of every SST-5 item in one split, for each variant. Cached like readout_cache()."""
    import data
    q = {"type": "score", "instructions": SST5_QUESTION, "criteria": SST5_LEVELS}
    key = {"model": engine.name, "prompt_sha1": prompt_fingerprint(engine), "splits_sha256": data.splits_fingerprint("sst5"),
           "question": q, "variants": SCORE_VARIANTS}
    if engine.adapter:  # a fine-tuned model is a different model: never reuse the base readouts
        key["adapter_sha256"] = engine.adapter_sha256
    path = readout_dir(engine) / f"sst5_{split}.json"
    rows = data.load_split("sst5", split)
    if path.exists() and (saved := json.loads(path.read_text()))["key"] == key:
        return saved["items"]
    items, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        raw, _ = raw_scores(engine, {"state": r["text"], "questions": {v: {**q, **extra} for v, extra in SCORE_VARIANTS.items()}},
                            share_question=True)
        items.append({"source_index": r["source_index"], "y": r["y"], **{v: raw[v]["logits"] for v in SCORE_VARIANTS},
                      "mass": {v: min(raw[v]["mass"]) for v in SCORE_VARIANTS}})
        if (i + 1) % 50 == 0:
            print(f"  sst5/{split} {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "items": items}))
    return items


def contrastive_levels(engine: Engine) -> dict:
    import data
    # A temperature per variant, fitted on train: a flatter distribution alone lowers the NLL, so the variants are
    # compared after calibration, as E14 compares Choice modes.
    train = score_readouts(engine, "train")
    temp = {v: fit_temperature_multiclass([it[v] for it in train], [it["y"] for it in train]) for v in SCORE_VARIANTS}
    cal = lambda it, v: softmax_list([z / temp[v] for z in it[v]])
    with data.tuning():  # choose on val; reading test here would raise
        val = score_readouts(engine, "val")
    val_metrics = {v: ordinal_metrics([softmax_list(it[v]) for it in val], [it["y"] for it in val]) for v in SCORE_VARIANTS}
    val_cal = {v: ordinal_metrics([cal(it, v) for it in val], [it["y"] for it in val]) for v in SCORE_VARIANTS}
    chosen = min(("pointwise", "contrastive"), key=lambda v: val_cal[v]["nll"])
    test = score_readouts(engine, "test")
    y = [it["y"] for it in test]
    test_metrics = {}
    for v in SCORE_VARIANTS:
        probs = [softmax_list(it[v]) for it in test]
        m = ordinal_metrics(probs, y)
        pairs = list(zip(probs, y))
        m["accuracy_ci95"] = bootstrap_ci(pairs, lambda s: ordinal_metrics([p for p, _ in s], [t for _, t in s])["accuracy"])
        m["adjacent_error_ci95"] = bootstrap_ci(pairs, lambda s: ordinal_metrics([p for p, _ in s], [t for _, t in s])["adjacent_error"])
        m["min_mass"] = min(it["mass"][v] for it in test)
        m["calibrated"] = {k: x for k, x in ordinal_metrics([cal(it, v) for it in test], y).items() if k != "confusion"}
        test_metrics[v] = m
    a, b = test_metrics["pointwise"]["adjacent_error"], test_metrics["contrastive"]["adjacent_error"]
    pred = {v: [max(range(len(SST5_LEVELS)), key=softmax_list(it[v]).__getitem__) for it in test] for v in SCORE_VARIANTS}
    paired = [(abs(pp - t) == 1, abs(pc - t) == 1) for pp, pc, t in zip(pred["pointwise"], pred["contrastive"], y)]
    diff_ci = bootstrap_ci(paired, lambda s: (sum(c for _, c in s) - sum(p for p, _ in s)) / len(s))
    # The documented Jev Score cases (exploratory: 11 cases, not held out): mean absolute gap to Jev's answer.
    jev = []
    for label, state, q, expected in JEV_DOC_CASES:
        if q["type"] != "score":
            continue
        raw, _ = raw_scores(engine, {"state": state, "questions": {v: {**q, **SCORE_VARIANTS[v]} for v in ("pointwise", "contrastive")}})
        got = {v: answer({**q}, raw[v]["logits"])["score"] for v in ("pointwise", "contrastive")}
        jev.append({"case": label, "state": state, "jev": expected, **got})
    return {
        "data": {"dataset": "sst5", "question": SST5_QUESTION, "levels": SST5_LEVELS,
                 "fitted_on": "train", "chosen_on": "val", "reported_on": "test", "n_train": len(train), "n_val": len(val), "n_test": len(test),
                 "splits_file": "datasets/splits_score_v1.json", "splits_sha256": data.splits_fingerprint("sst5")},
        "temperature": temp,
        "val": {v: {k: m[k] for k in ("accuracy", "adjacent_error", "far_error", "mae_expected", "nll")} for v, m in val_metrics.items()},
        "val_calibrated_nll": {v: val_cal[v]["nll"] for v in SCORE_VARIANTS},
        "chosen": chosen, "chosen_by": "lowest val NLL after a temperature fitted on train",
        "test": test_metrics,
        "adjacent_error_change": (b - a) / a if a else None,  # relative: -0.2 = 20% fewer adjacent errors
        "adjacent_error_diff_ci95": diff_ci,  # paired bootstrap of contrastive minus pointwise, absolute
        "jev_doc_cases": {"cases": jev, "note": "exploratory: 11 documented cases, not held out",
                          **{f"mae_vs_jev_{v}": statistics.mean(abs(c[v] - c["jev"]) for c in jev) for v in ("pointwise", "contrastive")}},
        "splits_accessed": data.ACCESS,
    }


# E17: opposite Nouls (W6, PLAN Task 4.5). Each BoolQ question is also asked in a negated form; the two answers should
# sum to 1. consistent() combines them. Val decides whether combining helps; test reports.
def negated(question: str) -> str:
    return f"Is the answer to the following question No? {question}"


def opposite_readouts(engine: Engine, split: str) -> list[dict]:
    import data
    key = {"model": engine.name, "prompt_sha1": prompt_fingerprint(engine), "splits_sha256": data.splits_fingerprint(),
           "negated": negated("\x00")}
    path = RESULTS / "readouts" / (tag(engine.name).lstrip("-") or "Qwen2.5-0.5B-Instruct") / f"boolq_opposite_{split}.json"
    rows = data.load_split("boolq", split)
    if path.exists() and (saved := json.loads(path.read_text()))["key"] == key:
        return saved["items"]
    items, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        q = r["question"][0].upper() + r["question"][1:] + "?"
        raw, _ = raw_scores(engine, {"state": r["passage"], "questions": {
            "x": {"type": "noul", "instructions": q}, "not_x": {"type": "noul", "instructions": negated(q)}}})
        items.append({"source_index": r["source_index"], "y": r["y"],
                      "z_x": raw["x"]["logits"][0] - raw["x"]["logits"][1],
                      "z_not_x": raw["not_x"]["logits"][0] - raw["not_x"]["logits"][1]})
        if (i + 1) % 100 == 0:
            print(f"  boolq_opposite/{split} {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "items": items}))
    return items


def opposite_metrics(items: list[dict]) -> dict:
    from minijev.judge import consistent
    y = [it["y"] for it in items]
    px = [sigmoid(it["z_x"]) for it in items]
    pn = [sigmoid(it["z_not_x"]) for it in items]
    combined = [consistent(a, b)[0] for a, b in zip(px, pn)]
    z = lambda ps: [logit(min(max(p, 1e-9), 1 - 1e-9)) for p in ps]
    pick = lambda m: {k: m[k] for k in ("accuracy", "ece", "nll", "mean_p_yes")}
    return {"mean_abs_sum_minus_1": statistics.mean(abs(a + b - 1) for a, b in zip(px, pn)),
            "median_sum": statistics.median(a + b for a, b in zip(px, pn)),
            "contradictions": sum((a > 0.5) == (b > 0.5) for a, b in zip(px, pn)) / len(y),
            "x_alone": pick(binary_metrics(z(px), y)),
            "negated_alone": pick(binary_metrics(z([1 - b for b in pn]), y)),
            "combined": pick(binary_metrics(z(combined), y)), "n": len(y)}


def opposite_pairs(engine: Engine) -> dict:
    import data
    with data.tuning():
        val = opposite_metrics(opposite_readouts(engine, "val"))
    use_combined = val["combined"]["nll"] < val["x_alone"]["nll"]
    test_items = opposite_readouts(engine, "test")
    test = opposite_metrics(test_items)
    y = [it["y"] for it in test_items]
    from minijev.judge import consistent
    pairs = [(sigmoid(it["z_x"]), consistent(sigmoid(it["z_x"]), sigmoid(it["z_not_x"]))[0], t) for it, t in zip(test_items, y)]
    test["accuracy_diff_ci95"] = bootstrap_ci(pairs, lambda s: sum((c > 0.5) == bool(t) for _, c, t in s) / len(s)
                                              - sum((a > 0.5) == bool(t) for a, _, t in s) / len(s))
    refund = next(c for c in JEV_DOC_CASES if c[0] == "noul: refund")
    not_refund = next(c for c in JEV_DOC_CASES if c[0] == "noul: not refund")
    r = ask(engine, {"state": refund[1], "questions": {"refund": refund[2], "not_refund": {**not_refund[2], "opposite_of": "refund"}}},
            settings=Settings(), debug=True)
    return {"data": {"dataset": "boolq", "chosen_on": "val", "reported_on": "test", "negated_template": negated("<question>")},
            "val": val, "combine_on_val": use_combined, "test": test,
            "jev_refund_pair": {"jev": {"refund": refund[3], "not_refund": not_refund[3], "sum": refund[3] + not_refund[3]},
                                "minijev_sum_before": r["debug"]["opposite_pairs"]["not_refund"]["sum_before"],
                                "minijev_after": {k: r["answers"][k]["noul"] for k in ("refund", "not_refund")}},
            "splits_accessed": data.ACCESS}


# E18: contrastive criteria for vague questions (W8, PLAN Task 4.6). The documented Jev Noul cases whose question has
# a library entry, asked without and with its criteria. Exploratory: 13 cases, not held out, and the library was
# written after the cases were visible (src/minijev/criteria.py says so).
CASE_TO_LIBRARY = {"noul: human agent": "wants_human", "noul: urgency": "urgent", "noul: strong python": "strong_python",
                   "noul: refund": "refund_request", "noul: refund (jaggedness)": "refund_request"}


def criteria_ablation(engine: Engine) -> dict:
    from minijev.criteria import LIBRARY
    rows = []
    for label, state, q, jev in JEV_DOC_CASES:
        if label not in CASE_TO_LIBRARY:
            continue
        entry = LIBRARY[CASE_TO_LIBRARY[label]]
        req = {"state": state, "questions": {"plain": q, "criteria": {**q, "criteria": entry["criteria"]}}}
        a = ask(engine, req, settings=Settings(), debug=True)["answers"]
        rows.append({"case": label, "state": state, "jev": jev, "plain": a["plain"]["noul"], "criteria": a["criteria"]["noul"]})
    summary = {v: {"mean_abs_gap_to_jev": statistics.mean(abs(r[v] - r["jev"]) for r in rows),
                   "same_side_of_0.5": sum((r[v] > 0.5) == (r["jev"] > 0.5) for r in rows) / len(rows)}
               for v in ("plain", "criteria")}
    return {"note": "exploratory: documented Jev cases, not held out; uncalibrated (Settings())", "n": len(rows),
            "summary": summary, "rows": rows}


# E19: a needle in a long state (W11, PLAN Task 6.5). One fact is put into the GDPR article at the start, the middle or
# the end, at 1k-8k tokens. Two Nouls ask about it: one true, one false. The same true question on the article without
# the fact is the control. kv mode with SDPA attention: no dense mask, so it fits in CPU memory.
NEEDLES = [
    ("The backup generator in building C was last serviced on 14 March.",
     "Does the text say that the backup generator in building C was last serviced on 14 March?",
     "Does the text say that the backup generator in building C was last serviced on 21 June?"),
    ("Maria Okafor was appointed head of the audit committee.",
     "Does the text say that Maria Okafor was appointed head of the audit committee?",
     "Does the text say that Daniel Weiss was appointed head of the audit committee?"),
    ("The access code for the storage room is 4172.",
     "Does the text say that the access code for the storage room is 4172?",
     "Does the text say that the access code for the storage room is 9381?"),
]


def needle(engine: Engine, lengths=(1000, 2000, 4000, 8000), positions=(0.05, 0.5, 0.95)) -> dict:
    ids = engine.tok.encode(gdpr_text(), add_special_tokens=False)
    assert len(ids) >= max(lengths), f"the article has {len(ids)} tokens"
    rows = []
    for n in lengths:
        hay = engine.tok.decode(ids[:n])
        cut = [i + 2 for i in range(len(hay) - 1) if hay[i] == "." and hay[i + 1] in " \n"]  # sentence ends
        control = {f"t{k}": {"type": "noul", "instructions": t} for k, (_, t, _) in enumerate(NEEDLES)}
        t0 = time.perf_counter()
        raw, usage = raw_scores(engine, {"state": hay, "questions": control}, "kv")
        rows.append({"tokens": usage["input_tokens"], "position": None, "needle": None, "seconds": time.perf_counter() - t0,
                     **{f"absent_{k}": sigmoid(raw[f"t{k}"]["logits"][0] - raw[f"t{k}"]["logits"][1]) for k in range(len(NEEDLES))}})
        for pos in positions:
            at = min(cut, key=lambda c: abs(c - pos * len(hay)))
            for k, (fact, true_q, false_q) in enumerate(NEEDLES):
                state = hay[:at] + fact + " " + hay[at:]
                t0 = time.perf_counter()
                raw, usage = raw_scores(engine, {"state": state, "questions": {
                    "true": {"type": "noul", "instructions": true_q}, "false": {"type": "noul", "instructions": false_q}}}, "kv")
                z = {q: raw[q]["logits"][0] - raw[q]["logits"][1] for q in ("true", "false")}
                rows.append({"tokens": usage["input_tokens"], "position": pos, "needle": k, "seconds": time.perf_counter() - t0,
                             "p_true": sigmoid(z["true"]), "p_false": sigmoid(z["false"]),
                             "mass": min(min(raw[q]["mass"]) for q in raw)})
            print(f"  needle {n} tokens at {pos:.2f}", flush=True)
    summary = []
    for n in lengths:
        ctrl = next(r for r in rows if r["position"] is None and abs(r["tokens"] - n) < 400)
        for pos in positions:
            rs = [r for r in rows if r["position"] == pos and abs(r["tokens"] - n) < 400]
            summary.append({"length": n, "position": pos,
                            "mean_p_true": statistics.mean(r["p_true"] for r in rs),
                            "mean_p_false": statistics.mean(r["p_false"] for r in rs),
                            "mean_p_absent": statistics.mean(ctrl[f"absent_{k}"] for k in range(len(NEEDLES))),
                            "correct": (sum(r["p_true"] > 0.5 for r in rs) + sum(r["p_false"] < 0.5 for r in rs)) / (2 * len(rs)),
                            "seconds_per_request": statistics.mean(r["seconds"] for r in rs)})
    return {"note": "uncalibrated (Settings()); kv mode; attention " + engine.model.config._attn_implementation,
            "haystack": "GDPR article, pinned revision", "needles": NEEDLES, "summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# E20: LoRA fine-tune with option-shuffle augmentation (PLAN Task 6.2, hard labels; poc/train_lora.py). The base
# model and the adapter answer the same held-out items. Raw test numbers are the main comparison. Each model also gets
# temperatures fitted on val (the adapter has seen train, so train would flatter it). Order flips: every AG News test
# item in all 4 rotations of its option order, listwise.
def rotation_readouts(engine: Engine) -> list[dict]:
    """AG News test, listwise, in the 4 rotations of the per-item order: one packed pass per item, 4 branches."""
    import data
    key = {"model": engine.name, "adapter_sha256": engine.adapter_sha256, "prompt_sha1": prompt_fingerprint(engine),
           "splits_sha256": data.splits_fingerprint(), "orders": "4 rotations of random.Random(source_index) order"}
    path = readout_dir(engine) / "ag_news_test_rotations.json"
    rows = data.load_split("ag_news", "test")
    if path.exists() and json.loads(path.read_text())["key"] == key:
        return json.loads(path.read_text())["items"]
    keys, items, t0 = list(AG_OPTIONS), [], time.perf_counter()
    for i, r in enumerate(rows):
        base = random.Random(r["source_index"]).sample(range(len(keys)), len(keys))
        orders = [base[j:] + base[:j] for j in range(len(keys))]
        qs = {f"r{j}": {"type": "choice", "instructions": AG_QUESTION, "choice_mode": "listwise",
                        "criteria": {keys[c]: AG_OPTIONS[keys[c]] for c in o}} for j, o in enumerate(orders)}
        raw, _ = raw_scores(engine, {"state": r["text"], "questions": qs})
        items.append({"source_index": r["source_index"], "y": r["y"], "orders": orders,
                      "logits": [[raw[f"r{j}"]["logits"][o.index(c)] for c in range(len(keys))] for j, o in enumerate(orders)],
                      "pos_logits": [raw[f"r{j}"]["logits"] for j in range(len(orders))]})
        if (i + 1) % 100 == 0:
            print(f"  rotations {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "items": items}))
    return items


def flip_metrics(items: list[dict]) -> dict:
    """Share of items whose listwise winner changes with the option order, accuracy over all orders, and how often
    the model picks each letter position (the right answer is at each position 25% of the time)."""
    winners = [[max(range(4), key=z.__getitem__) for z in it["logits"]] for it in items]
    flips = [len(set(w)) > 1 for w in winners]
    acc = [statistics.mean(w == it["y"] for w in ws) for ws, it in zip(winners, items)]
    picks = [statistics.mean(max(range(4), key=z.__getitem__) == j for it in items for z in it["pos_logits"])
             for j in range(4)]
    return {"flip_rate": statistics.mean(flips), "flip_rate_ci95": bootstrap_ci(flips, statistics.mean),
            "accuracy_all_orders": statistics.mean(acc), "accuracy_all_orders_ci95": bootstrap_ci(acc, statistics.mean),
            "position_picks": dict(zip("ABCD", picks))}


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Mean of b - a over the same items, with a 95% bootstrap interval over items."""
    d = [y - x for x, y in zip(a, b)]
    return {"mean": statistics.mean(d), "ci95": bootstrap_ci(d, statistics.mean)}


def lora(engine: Engine) -> dict:
    """E20. engine carries the adapter (--adapter); the base model is loaded next to it."""
    import data
    assert engine.adapter, "run with --adapter adapters/<model>/epoch-<n>"
    data.ACCESS.clear()
    base = Engine(engine.name, attn=engine.model.config._attn_implementation)
    models = {"base": base, "lora": engine}
    out: dict = {"adapter": engine.adapter, "adapter_sha256": engine.adapter_sha256}
    per_item: dict = {}
    for name, eng in models.items():
        bv, bt = readout_cache(eng, "boolq", "val"), readout_cache(eng, "boolq", "test")
        av, at = readout_cache(eng, "ag_news", "val"), readout_cache(eng, "ag_news", "test")
        a_t, _ = fit_affine([it["z"] for it in bv], [it["y"] for it in bv], slope_only=True)
        zs, ys = [it["z"] for it in bt], [it["y"] for it in bt]
        noul = {"raw": binary_entry(zs, ys, [0.0] * len(ys), 0),
                "val_temperature": binary_entry([a_t * v for v in zs], ys, [0.0] * len(ys), 0), "T": 1 / a_t}
        for m in (noul["raw"], noul["val_temperature"]):
            for k in ("mean_s", "mean_s_ci95", "output_tokens", "parse_failures", "has_probs"):
                m.pop(k, None)
        yat, choice = [it["y"] for it in at], {}
        for m in CHOICE_MODES:
            t = fit_temperature_multiclass([it[m] for it in av], [it["y"] for it in av])
            choice[m] = {"raw": with_ci([softmax_list(it[m]) for it in at], yat),
                         "val_temperature": with_ci([softmax_list([v / t for v in it[m]]) for it in at], yat),
                         "nll_raw": nll_multi([it[m] for it in at], yat), "T": t}
        rot = rotation_readouts(eng)
        out[name] = {"noul": noul, "choice": choice, "order_flips": flip_metrics(rot)}
        per_item[name] = {
            "noul_correct": [float((z > 0) == bool(y)) for z, y in zip(zs, ys)],
            "noul_nll": [math.log1p(math.exp(-z)) if y else math.log1p(math.exp(z)) for z, y in zip(zs, ys)],
            "listwise_correct": [float(max(range(4), key=it["listwise"].__getitem__) == it["y"]) for it in at],
            "flip": [float(len({max(range(4), key=z.__getitem__) for z in it["logits"]}) > 1) for it in rot]}
    out["delta_lora_minus_base"] = {k: paired_delta(per_item["base"][k], per_item["lora"][k]) for k in per_item["base"]}
    # Control: the base model with one bias per topic and a temperature, fitted on val (E22 has the same control).
    av, at = readout_cache(base, "ag_news", "val"), readout_cache(base, "ag_news", "test")
    t, bias = fit_bias_temperature([it["listwise"] for it in av], [it["y"] for it in av])
    ctrl = [float(max(range(4), key=lambda c: it["listwise"][c] / t + bias[c]) == it["y"]) for it in at]
    out["base_bias_control"] = {"ag_news_listwise_accuracy": statistics.mean(ctrl), "T": t, "bias": bias,
                                "delta_lora_minus_control": paired_delta(ctrl, per_item["lora"]["listwise_correct"])}
    out["splits_accessed"] = list(data.ACCESS)
    out["use_of_splits"] = {"train": "LoRA training (train_lora.py)", "val": "epoch choice, temperatures, bias control",
                            "test": "report only"}
    log = Path(engine.adapter).parent / "train_log.json"
    if log.exists():
        t = json.loads(log.read_text())
        out["training"] = {k: v for k, v in t.items() if k != "epochs"} | {
            "val_nll_per_epoch": [e["val_nll"] for e in t["epochs"]],
            "loss_per_step": [x for e in t["epochs"] for x in e["train_loss_per_step"]],
            "hours": t["epochs"][-1]["seconds"] / 3600}
    out["base_val_nll"] = {  # the untrained model on the same val items, so the per-epoch values have a start point
        "boolq": nll([it["z"] for it in readout_cache(base, "boolq", "val")], [it["y"] for it in readout_cache(base, "boolq", "val")], 1.0, 0.0),
        "ag_news": nll_multi([it["listwise"] for it in readout_cache(base, "ag_news", "val")], [it["y"] for it in readout_cache(base, "ag_news", "val")])}

    print(f"\n{engine.name} + LoRA ({engine.adapter}); test split. [..] = 95% bootstrap interval")
    for name in models:
        n, f = out[name]["noul"], out[name]["order_flips"]
        print(f"{name:5s} BoolQ raw acc {n['raw']['accuracy']:.3f} ECE {n['raw']['ece']:.3f} NLL {n['raw']['nll']:.3f} | "
              f"val-T {n['T']:.2f}: ECE {n['val_temperature']['ece']:.3f} NLL {n['val_temperature']['nll']:.3f}")
        for m, r in out[name]["choice"].items():
            print(f"      AG {m:9s} raw acc {r['raw']['accuracy']:.3f} ECE {r['raw']['ece']:.3f} NLL {r['nll_raw']:.3f} | "
                  f"val-T {r['T']:.2f}: ECE {r['val_temperature']['ece']:.3f}")
        print(f"      flips {f['flip_rate']:.3f} [{f['flip_rate_ci95'][0]:.2f},{f['flip_rate_ci95'][1]:.2f}]  "
              f"acc over 4 orders {f['accuracy_all_orders']:.3f}  picks " +
              " ".join(f"{k} {v:.2f}" for k, v in f["position_picks"].items()))
    for k, d in out["delta_lora_minus_base"].items():
        print(f"delta {k:16s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    c = out["base_bias_control"]
    d = c["delta_lora_minus_control"]
    print(f"base + val bias AG listwise acc {c['ag_news_listwise_accuracy']:.3f}; "
          f"lora - control {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    return out


def fit_bias_temperature(logits: list[list[float]], labels: list[int]) -> tuple[float, list[float]]:
    """T and one bias per class that minimize the NLL of softmax(z / T + b); b[0] = 0. A no-training control: it can
    move the argmax (a temperature alone cannot), so it corrects a shift of the whole scale, such as "too positive"."""
    z, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(z.shape[1] - 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp() + torch.cat([b.new_zeros(1), b]), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item(), [0.0] + b.tolist()


def transfer(engine: Engine) -> dict:
    """E21 and E22: base model against an adapter on SST-5 Scores (test) and the documented Jev cases (agreement with
    Jev, not accuracy). engine carries the adapter.
    E21, the E20 adapter (BoolQ + AG News): SST-5 is a task it was not trained on; temperatures are fitted on SST-5 train.
    E22, a per-task SST-5 adapter: it has seen SST-5 train, so temperatures are fitted on val for both models, and
    BoolQ and AG News test show what the adapter costs on the tasks it was not trained on."""
    import data
    assert engine.adapter, "run with --adapter adapters/<model>/epoch-<n>"
    log = Path(engine.adapter).parent / "train_log.json"
    task = json.loads(log.read_text()).get("task", "mixed") if log.exists() else "mixed"
    fit_split = "val" if task == "sst5" else "train"
    data.ACCESS.clear()
    base = Engine(engine.name, attn=engine.model.config._attn_implementation)
    models = {"base": base, "lora": engine}
    variants = ("pointwise", "listwise")
    out: dict = {"adapter": engine.adapter, "adapter_sha256": engine.adapter_sha256, "adapter_task": task}
    per_item: dict = {}
    for name, eng in models.items():
        train, test = score_readouts(eng, fit_split), score_readouts(eng, "test")
        y = [it["y"] for it in test]
        res, per_item[name] = {}, {}
        for v in variants:
            t = fit_temperature_multiclass([it[v] for it in train], [it["y"] for it in train])
            raw = [softmax_list(it[v]) for it in test]
            cal = [softmax_list([z / t for z in it[v]]) for it in test]
            res[v] = {"raw": ordinal_metrics(raw, y), "train_temperature": ordinal_metrics(cal, y), "T": t}
            pred = [max(range(len(p)), key=p.__getitem__) for p in raw]
            per_item[name][f"{v}_correct"] = [float(a == b) for a, b in zip(pred, y)]
            per_item[name][f"{v}_adjacent_or_exact"] = [float(abs(a - b) <= 1) for a, b in zip(pred, y)]
            per_item[name][f"{v}_nll_calibrated"] = [-math.log(max(p[b], 1e-12)) for p, b in zip(cal, y)]
        out[name] = {"sst5": res, "jevdocs": jevdocs(eng)["summary"]}
        if task == "sst5":  # the tasks this adapter was not trained on
            bt, at = readout_cache(eng, "boolq", "test"), readout_cache(eng, "ag_news", "test")
            per_item[name]["boolq_correct"] = [float((it["z"] > 0) == bool(it["y"])) for it in bt]
            per_item[name]["ag_news_listwise_correct"] = [float(max(range(4), key=it["listwise"].__getitem__) == it["y"])
                                                          for it in at]
            out[name]["other_tasks"] = {k: statistics.mean(per_item[name][k])
                                        for k in ("boolq_correct", "ag_news_listwise_correct")}
    out["delta_lora_minus_base"] = {k: paired_delta(per_item["base"][k], per_item["lora"][k]) for k in per_item["base"]}
    if task == "sst5":  # control: the base model with one bias per level and a temperature, fitted on val, no training
        val, test = score_readouts(base, "val"), score_readouts(base, "test")
        y, ctrl, per_ctrl = [it["y"] for it in test], {}, {}
        for v in variants:
            t, b = fit_bias_temperature([it[v] for it in val], [it["y"] for it in val])
            probs = [softmax_list([z / t + bi for z, bi in zip(it[v], b)]) for it in test]
            ctrl[v] = {**{k: m for k, m in ordinal_metrics(probs, y).items()}, "T": t, "bias": b}
            pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
            per_ctrl[f"{v}_correct"] = [float(a == c) for a, c in zip(pred, y)]
            per_ctrl[f"{v}_adjacent_or_exact"] = [float(abs(a - c) <= 1) for a, c in zip(pred, y)]
            per_ctrl[f"{v}_nll_calibrated"] = [-math.log(max(p[c], 1e-12)) for p, c in zip(probs, y)]
        out["base_bias_control"] = ctrl
        out["delta_lora_minus_bias_control"] = {k: paired_delta(per_ctrl[k], per_item["lora"][k]) for k in per_ctrl}
    out["splits_accessed"] = list(data.ACCESS)
    out["use_of_splits"] = ({"train": "adapter training", "val": "epoch choice; temperature per model and variant; "
                             "the bias control",
                             "test": "report only"} if task == "sst5" else
                            {"train": "temperature per model and variant", "val": "not used", "test": "report only"})
    out["note"] = "Jev cases: exploratory, documented cases, not held out; agreement with Jev, not accuracy"

    print(f"\n{engine.name} + LoRA ({engine.adapter}); SST-5 test. [..] = 95% bootstrap interval")
    for name in models:
        for v, r in out[name]["sst5"].items():
            print(f"{name:5s} {v:9s} raw acc {r['raw']['accuracy']:.3f} MAE {r['raw']['mae_expected']:.3f} "
                  f"NLL {r['raw']['nll']:.3f} | {fit_split}-T {r['T']:.2f}: acc {r['train_temperature']['accuracy']:.3f} "
                  f"NLL {r['train_temperature']['nll']:.3f}")
    for k, d in out["delta_lora_minus_base"].items():
        print(f"delta {k:28s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    for v, m in out.get("base_bias_control", {}).items():
        print(f"base + val bias {v:9s} acc {m['accuracy']:.3f} within-1 {1 - m['far_error']:.3f} NLL {m['nll']:.3f} "
              f"mean level {m['mean_level_predicted']:.2f}")
    for k, d in out.get("delta_lora_minus_bias_control", {}).items():
        print(f"lora - control {k:28s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    return out


EXPERIMENTS = {"demo": demo, "tree": tree, "latency": latency, "jevdocs": jevdocs, "permutation": permutation,
               "calibration": calibration, "llm_vs_minijev": llm_vs_minijev,
               "fanout": lambda engine: {"rows": fan_out(engine, 500)}, "quality": quality, "same_format": same_format, "order_bias": order_bias, "heldout": heldout,
               "template_sensitivity": template_sensitivity, "contrastive_levels": contrastive_levels,
               "opposite_pairs": opposite_pairs, "criteria_ablation": criteria_ablation,
               "needle": needle, "lora": lora, "transfer": transfer}
# transfer writes transfer.json (E21) or transfer_sst5.json (E22), by the adapter's task: see main()
ADAPTER_EXPERIMENTS = ("lora", "transfer")  # these need --adapter; the other experiments' caches do not key on it


def tag(model: str) -> str:
    return "" if model == MODEL else "-" + model.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=[*EXPERIMENTS, "all"], help="Experiment to run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--adapter", help="LoRA directory from train_lora.py (E20 lora, E21 transfer)")
    ap.add_argument("--update-manifest", action="store_true",
                    help="Generate/update dataset manifest with checksums (W17)")
    args = ap.parse_args()

    if args.update_manifest:
        update_manifest()
        return

    if args.name is None:
        ap.error("experiment name is required (unless using --update-manifest)")

    if args.adapter and args.name not in ADAPTER_EXPERIMENTS:
        ap.error(f"--adapter is for {', '.join(ADAPTER_EXPERIMENTS)} only (the other experiments' caches do not key on it)")
    engine = Engine(args.model, attn=args.attn, adapter=args.adapter)
    RESULTS.mkdir(exist_ok=True)
    for name in [n for n in EXPERIMENTS if n not in ADAPTER_EXPERIMENTS] if args.name == "all" else [args.name]:
        print(f"\n=== {name} ({args.model}) ===", flush=True)
        t0 = time.perf_counter()
        result = EXPERIMENTS[name](engine)
        result["model"], result["seconds_total"] = args.model, time.perf_counter() - t0
        suffix = f"_{result['adapter_task']}" if result.get("adapter_task", "mixed") != "mixed" else ""
        (RESULTS / f"{name}{suffix}{tag(args.model)}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
