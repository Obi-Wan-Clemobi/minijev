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
    all          everything above

Results go to poc/results/<name>[-<model>].json. Downloads are cached in poc/data/.
Everything runs on CPU in fp32; see docs/RESEARCH.md §7 for measured results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import ssl
import statistics
import time
import urllib.request
from pathlib import Path

import torch
from transformers import DynamicCache

from minijev_poc import (CONTENT_FREE_STATE, MODEL, MODES, SYSTEM, Engine, Settings, answer, ask, choice_block,
                         class_logits, noul_block, raw_scores, render_inline)

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

GDPR_REVISION = 1363040264  # the revision pinned by TypeSafe's parallel-questions cookbook
GDPR_QUESTIONS = {  # verbatim from https://docs.typesafe.ai/cookbooks/parallel_questions
    "breach_72h": {"type": "noul", "instructions": "Must a personal data breach be reported to the supervisory authority within 72 hours?"},
    "applies_non_eu": {"type": "noul", "instructions": "Does the regulation apply to organisations established outside the EU that offer goods or services to people in the EU?"},
    "dpo_all_orgs": {"type": "noul", "instructions": "Must every organisation appoint a Data Protection Officer, regardless of what data it processes?"},
    "pre_ticked_consent": {"type": "noul", "instructions": "Can valid consent be obtained through pre-ticked boxes or inactivity?"},
    "right_erasure": {"type": "noul", "instructions": "Does the regulation grant individuals a right to erasure of their personal data?"},
    "data_portability": {"type": "noul", "instructions": "Does the regulation include a right to data portability?"},
    "us_federal_law": {"type": "noul", "instructions": "Is the GDPR a United States federal law?"},
    "criminal_penalties": {"type": "noul", "instructions": "Does the GDPR itself impose criminal penalties such as imprisonment?"},
    "instrument_type": {"type": "choice", "instructions": "What kind of EU legal instrument is the GDPR?", "criteria": {
        "Regulation": "Directly binding law in all member states, no national implementation needed.",
        "Directive": "Sets goals that member states implement through national law.",
        "Treaty": "An international treaty between states.",
        "Recommendation": "Non-binding guidance."}},
    "max_fine": {"type": "choice", "instructions": "What is the maximum administrative fine for the most serious infringements?", "criteria": {
        "TwentyM_or_4pct": "Up to EUR 20 million or 4% of annual worldwide turnover, whichever is greater.",
        "TenM_or_2pct": "Up to EUR 10 million or 2% of annual worldwide turnover, whichever is greater.",
        "FixedCap": "A fixed amount not tied to turnover.",
        "NoFines": "The GDPR provides no administrative fines."}},
    "individual_rights": {"type": "score", "instructions": "How strong are the rights the GDPR grants to individuals over their data?", "criteria": [
        "None: individuals get no rights over their data.",
        "Weak: a right to be informed, but little control.",
        "Moderate: access and correction rights, but limited means to act on them.",
        "Strong: access, erasure, portability, and objection rights, with enforcement behind them."]},
    "penalty_severity": {"type": "score", "instructions": "How severe are the penalties the GDPR provides for non-compliance?", "criteria": [
        "None: no penalties of any kind.",
        "Symbolic: small fixed fines unlikely to change behavior.",
        "Substantial: fines large enough to matter to most companies.",
        "Severe: fines scaled to global revenue, material even to the largest companies."]},
    "compliance_burden": {"type": "score", "instructions": "How heavy is the compliance burden the GDPR places on organisations?", "criteria": [
        "Negligible: no meaningful obligations.",
        "Light: a few notices and disclosures.",
        "Moderate: documented processes and some dedicated roles for larger processors.",
        "Heavy: records, impact assessments, officers, and breach procedures for many organisations.",
        "Extreme: obligations so demanding that ordinary organisations cannot fully comply."]},
}
# Jev's batched means from the same cookbook (jev-1.12, full ~54k-character article).
GDPR_JEV = {"breach_72h": 0.804, "applies_non_eu": 0.990, "dpo_all_orgs": 0.030, "pre_ticked_consent": 0.040,
            "right_erasure": 0.990, "data_portability": 0.990, "us_federal_law": 0.010, "criminal_penalties": 0.108,
            "instrument_type": 1.000, "max_fine": 1.000, "individual_rights": 1.000, "penalty_severity": 1.000,
            "compliance_burden": 0.750}


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "minijev-poc/0.1"})
        # Certificates are checked. Behind a proxy that re-signs TLS, point SSL_CERT_FILE at its CA bundle.
        # MINIJEV_INSECURE_SSL=1 turns the check off; a download made that way could be tampered with.
        ssl_context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        if os.environ.get("MINIJEV_INSECURE_SSL") == "1":
            print(f"WARNING: certificate checks are off (MINIJEV_INSECURE_SSL=1) for {url}", flush=True)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ssl_context) as r:
            path.write_bytes(r.read())
    return path


def gdpr_text() -> str:
    url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
           f"&prop=extracts&explaintext=1&revids={GDPR_REVISION}")
    pages = json.loads(fetch(url, DATA / f"gdpr_{GDPR_REVISION}.json").read_text())["query"]["pages"]
    return next(iter(pages.values()))["extract"]


def gdpr_state(engine: Engine, n_tokens: int | None) -> dict:
    """The cookbook's state shape, with the article cut to its first n_tokens (CPU budget)."""
    text = gdpr_text()
    if n_tokens is not None:
        text = engine.tok.decode(engine.tok.encode(text, add_special_tokens=False)[:n_tokens])
    return {"article": {"source": f"https://en.wikipedia.org/?oldid={GDPR_REVISION}", "text": text}}


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
            # Rotation order: modes are tested in [A, B, C, A, B] pattern to avoid systematic order effects
            mode_order = [MODES[i % len(MODES)] for i in range(repeats)]
            mode_times = {mode: [] for mode in MODES}
            usage_per_mode = {}

            for mode in mode_order:
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
            "rotation_pattern": "modes rotated in [A, B, C, A, B, ...] order to mitigate systematic order effects"}


# Jev's published answers for documented inputs (docs.typesafe.ai, jev-1.13.0).
HUMAN = "Is the customer asking for a human agent?"
SEVERITY = ["Cosmetic; no impact to functionality", "Broken or degraded feature, but workaround exists",
            "Blocking issue; no workaround exists"]
PY_LEVELS = ["No experience", "Some familiarity", "Regular use in a job", "Deep expertise"]
SHOES = "Shoes arrived two weeks late and in the wrong size. Also I see two charges of $120 on my card. What are you going to do about this?"
JEV_DOC_CASES = [
    # (label, state, question, Jev's tracked answer)
    ("noul: human agent", "Thanks, that fixed it!", {"type": "noul", "instructions": HUMAN}, 0.02),
    ("noul: human agent", "How do I reset my password?", {"type": "noul", "instructions": HUMAN}, 0.07),
    ("noul: human agent", "I need this sorted today, whatever it takes.", {"type": "noul", "instructions": HUMAN}, 0.26),
    ("noul: human agent", "Are you a bot?", {"type": "noul", "instructions": HUMAN}, 0.40),
    ("noul: human agent", "Is there any way to speak to someone about my invoice?", {"type": "noul", "instructions": HUMAN}, 0.84),
    ("noul: human agent", "I have asked three times now. Can I please just talk to a real person?", {"type": "noul", "instructions": HUMAN}, 0.99),
    ("noul: repeat contact", "I have asked three times now. Can I please just talk to a real person?",
     {"type": "noul", "instructions": "Has the customer contacted support about this before?",
      "criteria": {"true": "Mentions a prior attempt, ticket, or that they have asked before", "false": "No sign of any previous contact"}}, 0.93),
    ("noul: urgency", "Help! My payouts have been failing for 3 days.", {"type": "noul", "instructions": "Does this convey urgency?"}, 0.95),
    ("noul: strong python", "My experience is in Java and Go. I have not used Python.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.03),
    ("noul: strong python", "I have used Python occasionally for small scripts alongside my main Java work.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.14),
    ("noul: strong python", "I used Python every day for two years in my last job, mostly data pipelines.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.81),
    ("noul: strong python", "I have written Python daily for eight years, including maintaining a large Django codebase.", {"type": "noul", "instructions": "Is the candidate strong in Python?"}, 0.92),
    ("noul: refund (jaggedness)", "I'm not happy with the fit. What are my options here?", {"type": "noul", "instructions": "Is the customer asking for a refund?"}, 0.22),
    ("noul: refund", "I was charged twice for the same order. Can someone look into this?", {"type": "noul", "instructions": "Is the customer asking for a refund?"}, 0.72),
    ("noul: not refund", "I was charged twice for the same order. Can someone look into this?", {"type": "noul", "instructions": "Is the customer asking for something other than a refund?"}, 0.47),
    ("score: severity", "The export button is misaligned by a few pixels on the settings page.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 0.0),
    ("score: severity", "The PDF export button does nothing when clicked. I can still export to CSV and convert it myself, but that takes ages.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.0),
    ("score: severity", "Export to PDF fails with a spinner that never finishes. Some of our team say CSV export still works for them, others say it fails too.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.11),
    ("score: severity", "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 1.43),
    ("score: severity", "Nobody on our team can log in since this morning. We get a 500 error on every attempt.", {"type": "score", "instructions": "How severe is the reported issue?", "criteria": SEVERITY}, 2.0),
    ("score: severity, numbers-only levels", "The export button is misaligned by a few pixels on the settings page.", {"type": "score", "instructions": "Rate severity from 0 to 2, where 2 is worst", "criteria": ["0", "1", "2"]}, 0.55),
    ("score: python experience", "My experience is in Java and Go. I have not used Python.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 0.0),
    ("score: python experience", "I have used Python occasionally for small scripts alongside my main Java work.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 1.0),
    ("score: python experience", "I used Python every day for two years in my last job, mostly data pipelines.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 2.05),
    ("score: python experience", "I have written Python daily for eight years, including maintaining a large Django codebase.", {"type": "score", "instructions": "How much Python experience does the candidate have?", "criteria": PY_LEVELS}, 2.89),
    ("score: frustration", "Help! My payouts have been failing for 3 days.", {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}, 1.05),
    ("choice: department", "My running shoes arrived in the wrong size. Can I swap them for a size 10?", {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "returns": "Exchanges, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "returns"),
    ("choice: department", "Help! My payouts have been failing for 3 days.", {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}}, "billing"),
    ("choice: department", SHOES, {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "returns": "Exchanges, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}}, "returns"),
    ("choice: return reason", SHOES, {"type": "choice", "instructions": "If the customer wants to return something, why?", "criteria": {
        "wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered", "damaged": "The item arrived broken or faulty",
        "changed_mind": "The item is fine, the customer no longer wants it", "other": "A return reason that fits none of the above"}}, "wrong_size"),
    ("choice: shipping issue", SHOES, {"type": "choice", "instructions": "If this is a shipping problem, which kind is it?", "criteria": {
        "not_delivered": "The package never arrived", "delayed": "The package is late but still on its way", "wrong_address": "The package went to the wrong place",
        "damaged_in_transit": "The package arrived damaged", "other": "A shipping problem that fits none of the above"}}, "delayed"),
    ("choice: resolution", SHOES, {"type": "choice", "instructions": "What does the customer want to happen?", "criteria": {
        "exchange": "Swap the item for a different one", "refund": "Money back", "replacement": "The same item sent again", "information": "Just an answer, no action needed"}}, "refund"),
    ("choice: tone", SHOES, {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}}, "frustrated"),
    ("choice: yes/no refund (jaggedness)", "I'm not happy with the fit. What are my options here?", {"type": "choice", "instructions": "Is the customer asking for a refund?", "criteria": {"yes": None, "no": None}}, "no"),
]


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

AG_OPTIONS = {  # AG News topics as a Choice, in label order 0..3
    "World": "World news, politics and international affairs",
    "Sports": "Sports",
    "Business": "Business, companies and the economy",
    "Technology": "Science and technology",
}
AG_QUESTION = "What is the topic of this news article?"


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


def options_text(criteria: dict) -> str:
    return "\n".join(f"- {k}: {d}" if d is not None else f"- {k}" for k, d in criteria.items())


def generate(engine: Engine, user: str, max_new_tokens: int) -> dict:
    """A chat completion: greedy decoding with a KV cache, as any LLM API would do it."""
    ids = engine.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                         add_generation_prompt=True)
    x = torch.tensor([ids])
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = engine.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                    repetition_penalty=1.0, pad_token_id=engine.tok.eos_token_id)
    seconds = time.perf_counter() - t0
    new = out[0, len(ids):]
    return {"text": engine.tok.decode(new, skip_special_tokens=True), "prompt_tokens": len(ids),
            "new_tokens": len(new), "seconds": seconds}


@torch.inference_mode()
def generate_logprobs(engine: Engine, ids: list[int], cache: DynamicCache | None = None) -> torch.Tensor:
    """What an LLM API with logprobs returns: generate one token, keep the next-token log-probs.

    This is the same computation as a readout. With a cache that holds a prefix of ids, only the
    rest of ids is prefilled.
    """
    x = torch.tensor([ids])
    out = engine.model.generate(x, attention_mask=torch.ones_like(x), past_key_values=cache, max_new_tokens=1,
                                do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                output_logits=True, return_dict_in_generate=True,
                                pad_token_id=engine.tok.eos_token_id)
    return torch.log_softmax(out.logits[0][0].float(), dim=-1)


@torch.inference_mode()
def generate_batched(engine: Engine, prefix: list[int], suffixes: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Greedy generation for all suffixes at once, as a serving stack batches concurrent requests.

    The prefix is prefilled once and its KV cache is copied to every row. Each suffix is
    left-padded (between the prefix and the suffix), and the pads are masked out. Each decode
    step then produces one token for every row with one load of the weights.
    """
    rows, p, width = len(suffixes), len(prefix), max(map(len, suffixes))
    cache = DynamicCache()
    engine.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache.batch_repeat_interleave(rows)
    pad = engine.tok.eos_token_id
    x = torch.tensor([[pad] * (width - len(s)) + s for s in suffixes])
    real = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in suffixes])
    mask = torch.cat([torch.ones(rows, p, dtype=torch.long), real], dim=1)
    pos = (p + real.cumsum(1) - 1).clamp(min=p)
    out = engine.model(x, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                       logits_to_keep=1)
    stop = set(engine.model.generation_config.eos_token_id or []) | {engine.tok.eos_token_id}
    new: list[list[int]] = [[] for _ in suffixes]
    done = [False] * rows
    nxt_pos = pos[:, -1:] + 1
    for _ in range(max_new_tokens):
        nxt = out.logits[:, -1].argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                new[i].append(t)
                done[i] = t in stop
        if all(done):
            break
        mask = torch.cat([mask, torch.ones(rows, 1, dtype=torch.long)], dim=1)
        out = engine.model(nxt[:, None], attention_mask=mask, position_ids=nxt_pos, past_key_values=cache,
                           use_cache=True)
        nxt_pos = nxt_pos + 1
    return new


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


def match_option(text: str, keys: list[str]) -> str | None:
    """Parse a written answer back into an option key. None = unparseable."""
    t = text.strip().strip("\"'`*.").lower()
    for k in keys:
        if t == k.lower():
            return k
    for k in sorted(keys, key=len, reverse=True):
        if t.startswith(k.lower()):
            return k
    return None


def number(value) -> float:
    """A written probability ("0.7", 0.7, "70%") as a float; anything else counts as 0."""
    try:
        return float(str(value).strip().rstrip("%")) / (100 if str(value).strip().endswith("%") else 1)
    except ValueError:
        return 0.0


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start:end + 1]) if start >= 0 < end else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def option_key(name: str, keys: list[str]) -> str | None:
    """Like match_option, but also accepts a truncated key ("Sport" for "Sports")."""
    hit = match_option(name, keys)
    if hit is None:
        t = name.strip().lower()
        hit = next((k for k in keys if len(t) >= 3 and k.lower().startswith(t)), None)
    return hit


def parse_probs_lenient(text: str, keys: list[str]) -> list[float] | None:
    """Written probabilities → a distribution over keys. None = nothing usable.

    Accepts the requested shape {"World": 0.1, ...} with loose key spelling, and the shape the
    0.5B model usually writes instead: {"topic": "Business", "probability": 0.7}. Mass that the
    reply does not assign is spread evenly over the options it does not name.
    """
    obj = parse_json(text)
    pairs: list[tuple[str, object]] = list(obj.items()) if obj else []
    if not obj:  # broken JSON: fall back to "key": value pairs
        pairs = [(k, v) for k, v in re.findall(r'"([^"]+)"\s*:\s*"?([^",}]+)"?', text)]
    got: dict[str, float] = {}
    named_p = next((number(v) for k, v in pairs if k.lower() in ("probability", "confidence")), None)
    for k, v in pairs:
        key = option_key(k, keys)
        if key is not None and not isinstance(v, (dict, list)):
            got[key] = max(got.get(key, 0.0), max(0.0, number(v)))
        elif k.lower() in ("topic", "answer", "choice", "option", "state") and isinstance(v, str):
            key = option_key(v, keys)
            if key is not None and key not in got:
                got[key] = named_p if named_p is not None else 1.0
    if not got or sum(got.values()) <= 0:
        return None
    rest = [k for k in keys if k not in got]
    spare = max(0.0, 1 - sum(got.values()))
    probs = [got.get(k, spare / len(rest) if rest else 0.0) for k in keys]
    s = sum(probs)
    return [p / s for p in probs]


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


def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))


def softplus(t: float) -> float:
    return t + math.log1p(math.exp(-t)) if t > 0 else math.log1p(math.exp(t))


def nll(z: list[float], y: list[int], a: float, b: float) -> float:
    return sum(softplus(-(a * zi + b)) if yi else softplus(a * zi + b) for zi, yi in zip(z, y)) / len(z)


def fit_affine(z: list[float], y: list[int], slope_only: bool) -> tuple[float, float]:
    """Minimize NLL of sigmoid(a*z + b) (Platt) or sigmoid(z / T) with T = 1/a (temperature).

    Newton steps with backtracking: saturated sigmoids have almost no curvature, so a full
    Newton step can overshoot wildly. Only accept a step that lowers the NLL.
    """
    a, b = 1.0, 0.0
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for zi, yi in zip(z, y):
            s = sigmoid(a * zi + b)
            r, w = s - yi, s * (1 - s)
            ga, gb = ga + r * zi, gb + r
            haa, hab, hbb = haa + w * zi * zi, hab + w * zi, hbb + w
        if slope_only:
            da, db = ga / (haa + 1e-9), 0.0
        else:
            det = haa * hbb - hab * hab + 1e-9
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        current, step = nll(z, y, a, b), 1.0
        while step > 1e-6 and nll(z, y, a - step * da, b - step * db) > current:
            step /= 2
        if step <= 1e-6:
            break
        a, b = a - step * da, b - step * db
    return a, b


def cross_fit(z: list[float], y: list[int], slope_only: bool) -> list[float]:
    """Two-fold: fit on one half, apply to the other. Returns out-of-fold calibrated log-odds."""
    half = len(z) // 2
    out = [0.0] * len(z)
    for fit_idx, apply_idx in ((range(half, len(z)), range(half)), (range(half), range(half, len(z)))):
        a, b = fit_affine([z[i] for i in fit_idx], [y[i] for i in fit_idx], slope_only)
        for i in apply_idx:
            out[i] = a * z[i] + b
    return out


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


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def verbal_probability(text: str) -> float | None:
    """A written probability ("0.8", "80%", "Probability: 0.8") as a float in [0, 1]. None = unparseable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%?)", text)
    if not m:
        return None
    v = float(m.group(1)) / (100 if m.group(2) or float(m.group(1)) > 1 else 1)
    return v if 0 <= v <= 1 else None


def fit_temperature_multiclass(logits: list[list[float]], labels: list[int]) -> float:
    """T that minimizes the NLL of softmax(z / T), by golden-section search on log T."""
    def nll_t(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for z, y in zip(logits, labels):
            m = max(v / t for v in z)
            total -= z[y] / t - m - math.log(sum(math.exp(v / t - m) for v in z))
        return total / len(labels)
    a, b, g = -2.0, 3.0, (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if nll_t(c) < nll_t(d) else (c, b)
    return math.exp((a + b) / 2)


def cross_fit_multiclass(logits: list[list[float]], labels: list[int]) -> list[list[float]]:
    """Two-fold temperature scaling: fit T on one half, apply it to the other. Out-of-fold probabilities."""
    half, out = len(logits) // 2, [None] * len(logits)
    for fit_idx, apply_idx in ((range(half, len(logits)), range(half)), (range(half), range(half, len(logits)))):
        t = fit_temperature_multiclass([logits[i] for i in fit_idx], [labels[i] for i in fit_idx])
        for i in apply_idx:
            out[i] = [v / t for v in logits[i]]
    return [softmax_list(z) for z in out]


def softmax_list(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


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


def same_format_prompt(doc: str, qs: dict) -> tuple[str, dict]:
    """The prompt that asks the model to write minijev's response itself, and the JSON shape it must follow."""
    lines, shape = [f"STATE:\n{doc}", "", "Answer every question below about the STATE.", ""], {}
    for qid, q in qs.items():
        text = render_inline(q["instructions"])
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            lines.append(f"{qid} (yes/no): {text}")
            for side, word in (("true", "Yes"), ("false", "No")):
                if crit.get(side):
                    lines.append(f"  {word} means: {render_inline(crit[side])}")
            shape[qid] = {"noul": 0.0}
        elif q["type"] == "choice":
            lines.append(f"{qid} (pick one option): {text}")
            lines += [f"  - {k}" + (f": {render_inline(d)}" if d is not None else "") for k, d in q["criteria"].items()]
            shape[qid] = {"choice": "<option name>", "probabilities": {k: 0.0 for k in q["criteria"]}}
        else:
            lines.append(f"{qid} (scale, lowest first): {text}")
            lines += [f"  {i}: {render_inline(level)}" for i, level in enumerate(q["criteria"])]
            shape[qid] = {"score": 0.0, "probabilities": {str(i): 0.0 for i in range(len(q["criteria"]))}}
        lines.append("")
    lines += ["Reply with only a JSON object in exactly this shape. Replace every 0.0 with your number.",
              "noul = the probability of yes. The probabilities of one question add up to 1.",
              "score = the expected level: the sum of level × probability.",
              json.dumps(shape)]
    return "\n".join(lines), shape


def check_written(parsed: dict | None, qs: dict) -> dict:
    """Per question: the value the model wrote, or, in plain words, why code cannot use it."""
    out = {}
    for qid, q in qs.items():
        if parsed is None:
            out[qid] = {"ok": False, "problem": "The reply is not valid JSON, so no answer can be read from it."}
            continue
        a = parsed.get(qid)
        if not isinstance(a, dict):
            out[qid] = {"ok": False, "problem": f'The reply has no "{qid}" at the top level of the JSON. It left it out, '
                                               "or put it inside another question."}
            continue
        try:
            if q["type"] == "noul":
                v = float(a["noul"])
                assert 0 <= v <= 1, f'"noul" is {v}, but a probability must be between 0 and 1.'
                out[qid] = {"ok": True, "noul": v}
            else:
                probs = a["probabilities"]
                if not isinstance(probs, dict):
                    raise ValueError('"probabilities" is not a list of name: number pairs.')
                probs = {str(k): float(v) for k, v in probs.items()}
                if q["type"] == "choice":
                    names = set(q["criteria"])
                    assert set(probs) == names, f'"probabilities" must name exactly the options {sorted(names)}; it has {sorted(probs)}.'
                    assert a["choice"] in names, f'"choice" is "{a["choice"]}", which is not one of the options.'
                    out[qid] = {"ok": True, "choice": a["choice"], "probabilities": probs,
                                "sums_to_1": abs(sum(probs.values()) - 1) < 0.02}
                else:
                    levels = {str(i) for i in range(len(q["criteria"]))}
                    assert set(probs) == levels, f'"probabilities" must name the levels {sorted(levels)}; it has {sorted(probs)}.'
                    out[qid] = {"ok": True, "score": float(a["score"]), "probabilities": probs,
                                "sums_to_1": abs(sum(probs.values()) - 1) < 0.02}
        except KeyError as e:
            out[qid] = {"ok": False, "problem": f"The answer is missing the field {e}."}
        except (TypeError, ValueError) as e:
            out[qid] = {"ok": False, "problem": str(e) if str(e).startswith('"') else "A value that must be a number is not a number."}
        except AssertionError as e:
            out[qid] = {"ok": False, "problem": str(e)}
    return out


class StructuredWriter:
    """Structured output, the way an AI API with an enforced JSON format works.

    The program writes every fixed part of the JSON (braces, keys, quotes) itself; those tokens are fed to the
    model in one pass per piece. The model decides only the content, one token per step, and only among valid
    tokens: the digits of a number, or the tokens of one of the option names. So the reply is always valid.
    """

    def __init__(self, engine: Engine, prompt: str):
        self.e, self.cache, self.text, self.decided, self.forced = engine, DynamicCache(), "", 0, 0
        ids = engine.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                             add_generation_prompt=True)
        self.prompt_tokens = len(ids)
        self._run(ids)
        self.digits = [engine.tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]

    def _run(self, ids: list[int]) -> None:
        self.logits = self.e.model(torch.tensor([ids]), past_key_values=self.cache, use_cache=True).logits[0, -1]

    def force(self, text: str) -> None:
        ids = self.e.tok.encode(text, add_special_tokens=False)
        self._run(ids)
        self.forced += len(ids)
        self.text += text

    def pick(self, allowed: list[int]) -> int:
        best = max(allowed, key=lambda t: self.logits[t].item())
        self._run([best])
        self.decided += 1
        self.text += self.e.tok.decode([best])
        return best

    def number(self, max_first: int = 1) -> float:
        """A number d.dd, first digit 0..max_first. For a probability (max_first 1), 1 is written as 1.00."""
        first = self.digits.index(self.pick(self.digits[:max_first + 1]))
        if max_first == 1 and first == 1:
            self.force(".00")
            return 1.0
        self.force(".")
        d1 = self.digits.index(self.pick(self.digits))
        d2 = self.digits.index(self.pick(self.digits))
        return first + d1 / 10 + d2 / 100

    def option(self, names: list[str]) -> str:
        """One of the names, token by token; ends with the closing quote."""
        seqs = {n: self.e.tok.encode(n + '"', add_special_tokens=False) for n in names}
        step, alive = 0, list(names)
        while True:
            t = self.pick(sorted({seqs[n][step] for n in alive if len(seqs[n]) > step}))
            alive = [n for n in alive if len(seqs[n]) > step and seqs[n][step] == t]
            step += 1
            done = [n for n in alive if len(seqs[n]) == step]
            if done:
                return done[0]


@torch.inference_mode()
def generate_structured(engine: Engine, prompt: str, qs: dict) -> dict:
    """The model writes minijev's response JSON with the format enforced. Returns the parsed JSON and the cost."""
    t0 = time.perf_counter()
    w = StructuredWriter(engine, prompt)
    out: dict = {}
    w.force("{")
    for i, (qid, q) in enumerate(qs.items()):
        w.force(("" if i == 0 else ", ") + json.dumps(qid) + ": {")
        if q["type"] == "noul":
            w.force('"noul": ')
            out[qid] = {"noul": w.number()}
        elif q["type"] == "choice":
            w.force('"choice": "')
            choice = w.option(list(q["criteria"]))
            probs = {}
            for j, k in enumerate(q["criteria"]):
                w.force((', "probabilities": {' if j == 0 else ", ") + json.dumps(k) + ": ")
                probs[k] = w.number()
            out[qid] = {"choice": choice, "probabilities": probs}
        else:
            n = len(q["criteria"])
            w.force('"score": ')
            score = w.number(max_first=n - 1)
            probs = {}
            for j in range(n):
                w.force((', "probabilities": {' if j == 0 else ", ") + f'"{j}": ')
                probs[str(j)] = w.number()
            out[qid] = {"score": score, "probabilities": probs}
        w.force("}" if q["type"] == "noul" else "}}")
    w.force("}")
    return {"parsed": out, "text": w.text, "seconds": time.perf_counter() - t0, "decided_tokens": w.decided,
            "forced_tokens": w.forced, "prompt_tokens": w.prompt_tokens}


def same_format_run(engine: Engine, body: dict, settings: Settings | None = None, mode: str = "packed") -> dict:
    """One request two ways: minijev's readout, and the same model writing minijev's response JSON itself."""
    doc = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"], indent=2, ensure_ascii=False)
    prompt, shape = same_format_prompt(doc, body["questions"])
    budget = int(len(engine.tok.encode(json.dumps(shape))) * 1.5) + 24  # room for every digit, and a little more
    t0 = time.perf_counter()
    readout = ask(engine, body, mode, settings=settings or Settings())
    t_readout = time.perf_counter() - t0
    g = generate(engine, prompt, budget)
    parsed = parse_json(g["text"])
    written = check_written(parsed, body["questions"])
    st = generate_structured(engine, prompt, body["questions"])
    structured = check_written(st["parsed"], body["questions"])

    def agreement(written: dict) -> dict:
        out = {}
        for qid, q in body["questions"].items():
            a, w = readout["answers"][qid], written[qid]
            if not w["ok"]:
                out[qid] = {"agree": False, "note": w["problem"]}
            elif q["type"] == "noul":
                out[qid] = {"agree": (a["noul"] > 0.5) == (w["noul"] > 0.5), "note": "same side of 0.5"}
            elif q["type"] == "choice":
                out[qid] = {"agree": a["choice"] == w["choice"], "note": "same option"}
            else:
                out[qid] = {"agree": round(a["score"]) == round(w["score"]), "note": "same rounded level"}
        return out
    compare = agreement(written)
    return {
        "model": engine.name,
        "readout": {"seconds": t_readout, "output_tokens": 0, "response": readout["answers"],
                    "input_tokens": readout["usage"]["input_tokens"]},
        "written": {"seconds": g["seconds"], "output_tokens": g["new_tokens"], "prompt_tokens": g["prompt_tokens"],
                    "token_budget": budget, "text": g["text"], "valid_json": parsed is not None,
                    "answers": written, "usable": sum(w["ok"] for w in written.values())},
        "structured": {"seconds": st["seconds"], "output_tokens": st["decided_tokens"], "forced_tokens": st["forced_tokens"],
                       "prompt_tokens": st["prompt_tokens"], "text": st["text"], "valid_json": True,
                       "answers": structured, "usable": sum(w["ok"] for w in structured.values())},
        "compare": compare, "compare_structured": agreement(structured), "prompt": prompt,
    }


SUPPORT_TICKET = {
    "state": "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
            "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages", "sales": "Pricing, new plans"}},
        "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
    },
}


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


def readout_cache(engine: Engine, dataset: str, split: str) -> list[dict]:
    """Raw readout logits for one split, cached. A readout never uses the labels, so computing it for val and test
    is not tuning on them. The cache is keyed by model, prompt fingerprint and splits file."""
    import data
    key = {"model": engine.name, "prompt_sha1": prompt_fingerprint(engine), "choice_block_sha1":
           hashlib.sha1(choice_block({"instructions": "?", "criteria": AG_OPTIONS}).encode()).hexdigest()[:12],
           "splits_sha256": data.splits_fingerprint()}
    path = RESULTS / "readouts" / f"{tag(engine.name).lstrip('-') or 'Qwen2.5-0.5B-Instruct'}" / f"{dataset}_{split}.json"
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
            q = {"type": "choice", "instructions": AG_QUESTION, "criteria": AG_OPTIONS}
            raw, _ = raw_scores(engine, {"state": r["text"], "questions": {m: {**q, "choice_mode": m} for m in CHOICE_MODES}})
            items.append({"source_index": r["source_index"], "y": r["y"], **{m: raw[m]["logits"] for m in CHOICE_MODES},
                          "mass": min(min(raw[m]["mass"]) for m in CHOICE_MODES)})
        if (i + 1) % 100 == 0:
            print(f"  {dataset}/{split} {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "items": items}))
    return items


def nll_multi(logits: list[list[float]], y: list[int], t: float = 1.0) -> float:
    total = 0.0
    for z, yi in zip(logits, y):
        m = max(v / t for v in z)
        total -= z[yi] / t - m - math.log(sum(math.exp(v / t - m) for v in z))
    return total / len(y)


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
    print(f"Choice (AG News): T = " + ", ".join(f"{m} {t:.2f}" for m, t in choice_t.items()) + f"; val picks {choice_pick}")
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

EXPERIMENTS = {"demo": demo, "tree": tree, "latency": latency, "jevdocs": jevdocs, "permutation": permutation,
               "calibration": calibration, "llm_vs_minijev": llm_vs_minijev,
               "fanout": lambda engine: {"rows": fan_out(engine, 500)}, "quality": quality, "same_format": same_format, "order_bias": order_bias, "heldout": heldout}


def tag(model: str) -> str:
    return "" if model == MODEL else "-" + model.split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=[*EXPERIMENTS, "all"], help="Experiment to run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--update-manifest", action="store_true",
                    help="Generate/update dataset manifest with checksums (W17)")
    args = ap.parse_args()

    if args.update_manifest:
        update_manifest()
        return

    if args.name is None:
        ap.error("experiment name is required (unless using --update-manifest)")

    engine = Engine(args.model, attn=args.attn)
    RESULTS.mkdir(exist_ok=True)
    for name in EXPERIMENTS if args.name == "all" else [args.name]:
        print(f"\n=== {name} ({args.model}) ===", flush=True)
        t0 = time.perf_counter()
        result = EXPERIMENTS[name](engine)
        result["model"], result["seconds_total"] = args.model, time.perf_counter() - t0
        (RESULTS / f"{name}{tag(args.model)}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
