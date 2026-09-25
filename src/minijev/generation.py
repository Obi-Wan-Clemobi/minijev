"""Generation baselines: the same model writing its answer, freely or with an enforced format."""

from __future__ import annotations

import json
import re
import time

import torch
from transformers import DynamicCache

from .engine import Engine
from .judge import ask
from .prompt import SYSTEM, render_inline
from .settings import Settings


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


def verbal_probability(text: str) -> float | None:
    """A written probability ("0.8", "80%", "Probability: 0.8") as a float in [0, 1]. None = unparseable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%?)", text)
    if not m:
        return None
    v = float(m.group(1)) / (100 if m.group(2) or float(m.group(1)) > 1 else 1)
    return v if 0 <= v <= 1 else None


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
