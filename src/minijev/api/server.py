"""minijev HTTP API for the web playground.

    minijev serve --port 8000          (or: uvicorn minijev.api.server:app)

One Engine is loaded at startup. A lock serializes forward passes: the model runs on the CPU, one request at a time.
Run it from poc/ (or set MINIJEV_ENV_FILE, MINIJEV_CALIBRATION_DIR, MINIJEV_DATA_DIR and MINIJEV_RESULTS_DIR):
results/, calibration/, data/ and minijev.env are read from the current directory.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import DynamicCache

from ..engine import Engine, StateCache
from ..fixtures import GDPR_QUESTIONS, JEV_DOC_CASES, SHOES, gdpr_state
from ..generation import generate, generate_logprobs, match_option, options_text, parse_json, same_format_run
from ..judge import ask, branches_for, validate, with_modes
from ..primitives import answer, class_logits
from ..prompt import choice_block, render_inline
from ..settings import Settings
from .schema import CompareReq, ModelReq, Req

RESULTS = Path(os.environ.get("MINIJEV_RESULTS_DIR", Path.cwd() / "results"))

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

app = FastAPI(title="minijev", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.RLock()  # re-entrant: engine() may load the model while the lock is held
_engine: Engine | None = None


def cached(e: Engine) -> Engine:
    """Give the engine the state cache that minijev.env asks for (MINIJEV_STATE_CACHE entries; 0 = off)."""
    e.state_cache = StateCache(Settings.load().state_cache)
    return e


def engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = cached(Engine())  # model, threads and attention from minijev.env
    return _engine


def settings_for(overrides: dict) -> Settings:
    base = Settings.load()
    known = {f.name for f in dataclasses.fields(Settings)} - {"fitted", "state_cache"}  # the cache is per server
    bad = sorted(set(overrides) - known)
    if bad:
        raise HTTPException(400, f"unknown settings {bad}; known: {sorted(known)}")
    try:
        return dataclasses.replace(base, **overrides).with_calibration()
    except AssertionError as e:
        raise HTTPException(400, str(e))


def checked(req: Req, s: Settings) -> dict:
    """The request with readout modes filled in, validated like the POC does."""
    if req.mode not in ("naive", "kv", "packed"):
        raise HTTPException(400, f"mode must be naive, kv or packed, not {req.mode!r}")
    body = {"state": req.state, "questions": {qid: with_modes(q, s) for qid, q in req.questions.items()}}
    try:
        validate(body)
    except (AssertionError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e) or type(e).__name__)
    return body


# ---------------------------------------------------------------------------
# Endpoints


@app.get("/v1/health")
def health():
    e = engine()
    return {"ok": True, "model": e.name, "state_cache": e.state_cache.stats()}


@app.post("/v1/ask")
def v1_ask(req: Req):
    """The Jev contract, plus raw logits so the page can re-score with new temperatures."""
    s = settings_for(req.settings)
    body = checked(req, s)
    with _lock:
        resp = ask(engine(), body, req.mode, settings=s, debug=True)
    resp["settings"] = dataclasses.asdict(s)
    resp["questions"] = body["questions"]  # with modes filled in, so the page knows the branch layout
    return resp


def branch_labels(qid: str, q: dict) -> list[str]:
    if q["type"] == "score" and q.get("score_mode") == "pointwise":
        return [f"{qid} · {render_inline(level)}" for level in q["criteria"]]
    if q["type"] == "choice" and q.get("choice_mode") == "pointwise":
        return [f"{qid} · {key}" for key in q["criteria"]]
    if q["type"] == "choice" and q.get("choice_mode") == "averaged":
        keys = list(q["criteria"])
        return [f"{qid} · {keys[r]} first" for r in range(len(keys))]
    return [qid]


@app.post("/v1/tree")
def v1_tree(req: Req):
    """The packed layout: prefix, shared question heads and branch tokens, and their positions."""
    s = settings_for(req.settings)
    body = checked(req, s)
    e = engine()
    prefix = e.prefix_ids(body["state"])
    branches, heads, start = [], [], len(prefix)
    for qid, q in body["questions"].items():
        bs = branches_for(e, qid, q, s.share_question)
        head = bs[0].head if bs else ()
        if head:  # the two-level tree: the question text once, then one short branch per item
            heads.append({"question": qid, "length": len(head), "start": start,
                          "positions": [len(prefix), len(prefix) + len(head) - 1],
                          "tokens": [e.tok.decode([t]) for t in head]})
            start += len(head)
        for b, label in zip(bs, branch_labels(qid, q)):
            first = len(prefix) + len(b.head)
            branches.append({"question": qid, "label": label, "type": q["type"], "pointwise": b.pointwise,
                             "head": len(heads) - 1 if b.head else None, "length": len(b.ids), "start": start,
                             "positions": [first, first + len(b.ids) - 1],
                             "tokens": [e.tok.decode([t]) for t in b.ids]})
            start += len(b.ids)
    # Where the user's state sits inside the prefix: after the chat template and "STATE:\n", before the blank line.
    head = e.tok.encode(f"{e._head}STATE:\n", add_special_tokens=False)
    lo = len(head) if prefix[:len(head)] == head else 0
    hi = len(prefix)
    while hi > lo and not e.tok.decode(prefix[hi - 1:hi]).strip():
        hi -= 1
    return {"prefix": {"length": len(prefix), "tokens": [e.tok.decode([t]) for t in prefix], "state_span": [lo, hi]},
            "heads": heads, "branches": branches, "total": start,
            "max_position": max((b["positions"][1] for b in branches), default=len(prefix) - 1)}


def to_choice(q: dict) -> dict:
    """Every question as a plain Choice, so every method answers the same thing."""
    if q["type"] == "noul":
        return {"type": "choice", "instructions": q["instructions"], "criteria": {"yes": None, "no": None}}
    if q["type"] == "score":
        return {"type": "choice", "instructions": q["instructions"],
                "criteria": {render_inline(level): None for level in q["criteria"]}}
    return {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}


def by_name(q: dict) -> str:
    return (f"QUESTION: {render_inline(q['instructions'])}\nOPTIONS:\n{options_text(q['criteria'])}\n"
            "Answer with the name of the best option only.")


@app.post("/v1/compare")
def v1_compare(req: CompareReq):
    """The same questions (as Choices) through the readout and through generation. CPU-heavy: seconds."""
    known = {"readout", "logprobs_cached", "generate_cached", "generate_uncached", "generate_json"}
    if set(req.methods) - known:
        raise HTTPException(400, f"methods must be in {sorted(known)}")
    qs = {qid: to_choice(q) for qid, q in req.questions.items()}
    body = checked(Req(state=req.state, questions=qs), settings_for({}))
    qs = {qid: {k: v for k, v in q.items() if k != "choice_mode"} for qid, q in body["questions"].items()}
    e, state = engine(), body["state"]
    doc = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
    out = {}
    with _lock, torch.inference_mode():
        prefix = e.prefix_ids(state)

        def timed(fn):
            t0 = time.perf_counter()
            answers, tokens = fn()
            return {"seconds": time.perf_counter() - t0, "output_tokens": tokens, "answers": answers}

        def readout():
            from ..judge import raw_scores
            raw, _ = raw_scores(e, {"state": state, "questions": qs}, "packed")
            return {qid: answer(q, raw[qid]["logits"])["choice"] for qid, q in qs.items()}, 0

        def logprobs_cached():
            cache, ans = DynamicCache(), {}
            e.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            for qid, q in qs.items():
                lp = generate_logprobs(e, prefix + e.suffix_ids(choice_block(q)), cache)
                ans[qid] = answer(q, class_logits(lp, e.letters[:len(q["criteria"])])[0])["choice"]
                cache.crop(len(prefix))
            return ans, len(qs)

        def generate_cached():
            cache, ans, tokens = DynamicCache(), {}, 0
            e.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            for qid, q in qs.items():
                x = torch.tensor([prefix + e.suffix_ids(by_name(q))])
                new = e.model.generate(x, attention_mask=torch.ones_like(x), past_key_values=cache, max_new_tokens=12,
                                       do_sample=False, temperature=None, top_p=None, top_k=None,
                                       repetition_penalty=1.0, pad_token_id=e.tok.eos_token_id)[0, x.shape[1]:]
                ans[qid] = match_option(e.tok.decode(new, skip_special_tokens=True), list(q["criteria"]))
                tokens += len(new)
                cache.crop(len(prefix))
            return ans, tokens

        def generate_uncached():
            ans, tokens = {}, 0
            for qid, q in qs.items():
                g = generate(e, f"STATE:\n{doc}\n\n{by_name(q)}", 12)
                ans[qid], tokens = match_option(g["text"], list(q["criteria"])), tokens + g["new_tokens"]
            return ans, tokens

        def generate_json():
            listing = "\n\n".join(f"q{i}: {render_inline(q['instructions'])}\nOPTIONS:\n{options_text(q['criteria'])}"
                                  for i, q in enumerate(qs.values(), start=1))
            g = generate(e, f"STATE:\n{doc}\n\nAnswer each question below by choosing one of its options.\n\n{listing}"
                            "\n\nReply with a JSON object that maps each question id (q1, q2, ...) to the name of the "
                            "chosen option. Reply with the JSON object only.", 24 * len(qs) + 40)
            parsed = parse_json(g["text"]) or {}
            return ({qid: match_option(str(parsed.get(f"q{i}", "")), list(q["criteria"]))
                     for i, (qid, q) in enumerate(qs.items(), start=1)}, g["new_tokens"])

        fns = {"readout": readout, "logprobs_cached": logprobs_cached, "generate_cached": generate_cached,
               "generate_uncached": generate_uncached, "generate_json": generate_json}
        for name in ["readout"] + [m for m in req.methods if m != "readout"]:
            out[name] = timed(fns[name])
    picks = out["readout"]["answers"]
    for name, r in out.items():
        r["agrees_with_readout"] = sum(r["answers"].get(k) == picks[k] for k in qs) / len(qs)
        r["parse_failures"] = sum(v is None for v in r["answers"].values())
    return {"model": e.name, "questions": qs, "methods": out}


@app.post("/v1/same_format")
def v1_same_format(req: Req):
    """minijev's readout vs the same model writing minijev's exact response as JSON. Same state, same questions."""
    s = settings_for(req.settings)
    body = checked(req, s)
    with _lock:
        return same_format_run(engine(), body, s, req.mode)


@app.get("/v1/presets")
def v1_presets():
    shoes = {}
    for label, state, q, jev in JEV_DOC_CASES:
        if state == SHOES:
            shoes[label.split(": ")[1].replace(" ", "_")] = {**q, "jev": jev}
    try:
        gdpr = gdpr_state(engine(), 500)
    except Exception:  # no network and no cached copy
        gdpr = None
    presets = [
        {"id": "support", "name": "Support ticket",
         "state": "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
         "questions": {
             "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
             "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
                 "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages",
                 "sales": "Pricing, new plans"}},
             "tone": {"type": "score", "instructions": "How upset is the customer?",
                      "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]}}},
        {"id": "shoes", "name": "Jev doc cases", "state": SHOES,
         "questions": {k: {kk: vv for kk, vv in q.items() if kk != "jev"} for k, q in shoes.items()},
         "jev": {k: q["jev"] for k, q in shoes.items()}},
    ]
    if gdpr is not None:
        presets.append({"id": "gdpr", "name": "GDPR · 13 questions", "state": gdpr["article"]["text"], "questions": GDPR_QUESTIONS})
    return presets


@app.get("/v1/results")
def v1_results():
    """The committed measurements, read only."""
    def load(name):
        p = RESULTS / name
        return json.loads(p.read_text()) if p.exists() else None

    out = {}
    for tag, suffix in (("0.5B", ""), ("1.5B", "-Qwen2.5-1.5B-Instruct")):
        cal = load(f"calibration{suffix}.json")
        out[tag] = {
            "calibration": cal and {k: cal[k] for k in ("n", "base_rate_yes", "temperature_all", "platt_all", "metrics")},
            "fanout": (load(f"fanout{suffix}.json") or {}).get("rows"),
            "permutation": (load(f"permutation{suffix}.json") or {}).get("summary"),
            "jevdocs": (load(f"jevdocs{suffix}.json") or {}).get("summary"),
            "order_bias": load(f"order_bias{suffix}.json"),
            "template_sensitivity": load(f"template_sensitivity{suffix}.json"),
            "lora": load(f"lora{suffix}.json"),
            "quality": {task: {k: v for k, v in r.items() if k != "rows"}
                        for task, r in ((load(f"quality{suffix}.json") or {}).get("tasks") or {}).items()} or None,
        }
    e10 = load("llm_vs_minijev.json") or {}
    out["single_decision"] = {k: e10.get(k) for k in ("n", "single_decision", "generation_costs",
                                                        "label_vs_readout_agreement")}
    out["demo"] = load("demo.json")
    return out


@app.get("/v1/criteria")
def v1_criteria():
    """The contrastive criteria library for vague yes/no questions (W8)."""
    from ..criteria import LIBRARY
    return LIBRARY


@app.get("/v1/data")
def v1_data():
    """The data card's facts, computed from the files now, and every claim of data.py check with its result."""
    path = RESULTS.parent / "data.py"  # poc/data.py, next to poc/results
    if not path.exists():
        raise HTTPException(404, "The data checks need poc/data.py: run the API from poc/.")
    if "data" not in sys.modules:
        spec = importlib.util.spec_from_file_location("data", path)
        sys.modules["data"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["data"])
    data = sys.modules["data"]
    claims: list = []
    ok = data.check(claims, quiet=True)
    heldout = {}
    for tag_, suffix in (("0.5B", ""), ("1.5B", "-Qwen2.5-1.5B-Instruct")):
        p = RESULTS / f"heldout{suffix}.json"
        if p.exists():
            r = json.loads(p.read_text())
            heldout[tag_] = {k: r[k] for k in ("calibration", "val", "test", "splits_accessed", "use_of_splits", "base_rates")}
    return {"summary": data.summary(), "claims": claims, "all_pass": ok, "heldout": heldout}


@app.get("/v1/calibration")
def v1_calibration():
    """The fitted calibration for the loaded model, with its provenance, or null when none is fitted."""
    s = dataclasses.replace(Settings.load(), model=engine().name).with_calibration()
    return {"model": engine().name, "mode": s.calibration, "fitted": s.fitted or None,
            "selected_choice_mode": s.resolved_choice_mode() if s.fitted else None}


@app.get("/v1/model")
def v1_model():
    return {"model": engine().name, "available": MODELS}


@app.post("/v1/model")
def v1_set_model(req: ModelReq):
    global _engine
    if req.name not in MODELS:
        raise HTTPException(400, f"model must be one of {MODELS}")
    with _lock:
        if _engine is None or _engine.name != req.name:
            _engine = cached(Engine(req.name))
    return {"model": _engine.name}
