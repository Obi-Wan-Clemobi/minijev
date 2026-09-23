"""minijev POC: Jev-shaped typed decisions read from label-token probabilities.

A request is evaluated as a prefix tree (docs/RESEARCH.md §3.1):

    [system + STATE]  ── encoded once
         ├── Noul branch            → P(Yes) vs P(No) at the branch's last token
         ├── Choice branch (listwise, letter labels) → softmax over the letters
         └── Score: one branch per level (pointwise) → yes/no log-odds per level,
                                                       softmax across levels
    (choice_mode="pointwise" gives a Choice one branch per option, like a Score:
     Jev's stage 1 for large option sets, and exactly order-invariant.)

Each branch sees the state and itself, never another branch. There are three ways to
compute the same tree, and all three should give the same numbers:
    naive   re-encode the state for every branch
    kv      encode the state once, run each branch against the cached keys/values
    packed  one forward pass over [state][b1][b2]..., with a block attention mask and
            position ids that restart after the state
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are a precise classifier. You read a STATE and answer one QUESTION about it."
CONTENT_FREE_STATE = "N/A"
# "I" is skipped: it is a common first word of an answer, so it attracts unrelated mass.
LETTERS = list("ABCDEFGHJKLMNOPQRSTUVWXYZ")
YES, NO = ["Yes", "yes", "YES"], ["No", "no", "NO"]
MODES = ("naive", "kv", "packed")


def render(value) -> str:
    """Text for a state, instruction or description. Structure stays as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_inline(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass
class Branch:
    """One path from the end of the state to a readout position."""

    ids: list[int]
    classes: list[list[int]]  # per answer class: the token ids (label variants) it owns
    question: str  # question id this branch belongs to
    pointwise: bool = False  # one of several yes/no branches (a Score level or a Choice option)


class Engine:
    def __init__(self, model: str = MODEL, attn: str = "eager", threads: int = 6):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float32, attn_implementation=attn
        ).eval()
        self.name = model
        self.yes = self._variants(YES)
        self.no = self._variants(NO)
        self.letters = [self._variants([c]) for c in LETTERS]
        sentinel = "\x00SPLIT\x00"
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self._head, self._tail = text.split(sentinel)

    def _variants(self, words: list[str]) -> list[int]:
        """Token ids for each word with and without a leading space. Each must be one token."""
        ids = []
        for w in words:
            for form in (w, " " + w):
                t = self.tok.encode(form, add_special_tokens=False)
                assert len(t) == 1, f"label {form!r} is {len(t)} tokens"
                ids.append(t[0])
        return sorted(set(ids))

    # ---- prompt pieces: tokenized separately, never as one joined string ----
    def prefix_ids(self, state) -> list[int]:
        text = f"{self._head}STATE:\n{render(state)}\n\n"
        return self.tok.encode(text, add_special_tokens=False)

    def suffix_ids(self, block: str) -> list[int]:
        return self.tok.encode(block + self._tail, add_special_tokens=False)

    # ---- the three ways to evaluate a prefix tree ----
    @torch.inference_mode()
    def readouts(self, prefix: list[int], branches: list[Branch], mode: str = "packed") -> torch.Tensor:
        """Next-token log-probs (full vocab) at the last token of every branch: [n_branches, vocab]."""
        if mode == "naive":
            rows = [self.model(torch.tensor([prefix + b.ids]), logits_to_keep=1).logits[0, -1] for b in branches]
        elif mode == "kv":
            cache = DynamicCache()
            self.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            rows = []
            for b in branches:
                out = self.model(torch.tensor([b.ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
                rows.append(out.logits[0, -1])
                cache.crop(len(prefix))  # back to the shared state for the next branch
        elif mode == "packed":
            pos, mask, last = self.pack(len(prefix), [len(b.ids) for b in branches])
            out = self.model(
                torch.tensor([prefix + [t for b in branches for t in b.ids]]),
                attention_mask=mask,
                position_ids=pos,
                logits_to_keep=last,
            )
            rows = list(out.logits[0])
        else:
            raise ValueError(mode)
        return torch.log_softmax(torch.stack(rows).float(), dim=-1)

    def pack(self, n_prefix: int, lengths: list[int]):
        """Position ids, additive 4D mask and readout indices for [prefix][b1][b2]...

        Token i may attend to token j iff j <= i and (j is in the prefix, or i and j are in
        the same branch). Positions restart at n_prefix in every branch, so the largest
        position is n_prefix + longest branch: Jev's "state plus the longest question".
        """
        total = n_prefix + sum(lengths)
        segment = torch.zeros(total, dtype=torch.long)  # 0 = prefix, k = branch k
        pos = torch.arange(total)
        last, start = [], n_prefix
        for k, n in enumerate(lengths, start=1):
            segment[start : start + n] = k
            pos[start : start + n] = torch.arange(n_prefix, n_prefix + n)
            last.append(start + n - 1)
            start += n
        causal = torch.ones(total, total, dtype=torch.bool).tril()
        visible = causal & ((segment[None, :] == 0) | (segment[None, :] == segment[:, None]))
        mask = torch.zeros(total, total, dtype=torch.float32)
        mask.masked_fill_(~visible, torch.finfo(torch.float32).min)
        return pos[None, :], mask[None, None], torch.tensor(last)


# ---------------------------------------------------------------------------
# Questions → branches


def noul_block(q: dict) -> str:
    lines = [f"QUESTION: {render(q['instructions'])}"]
    crit = q.get("criteria") or {}
    if crit.get("true") is not None:
        lines.append(f"Yes means: {render_inline(crit['true'])}")
    if crit.get("false") is not None:
        lines.append(f"No means: {render_inline(crit['false'])}")
    lines.append("Answer with Yes or No.")
    return "\n".join(lines)


def choice_block(q: dict) -> str:
    lines = [f"QUESTION: {render(q['instructions'])}", "OPTIONS:"]
    for letter, (key, desc) in zip(LETTERS, q["criteria"].items()):
        lines.append(f"{letter}) {key}: {render_inline(desc)}" if desc is not None else f"{letter}) {key}")
    lines.append("Answer with the letter of the best option.")
    return "\n".join(lines)


def proposal_block(q: dict, proposed: str) -> str:
    """Pointwise: judge one Score level (or one Choice option) on its own, without the others."""
    return "\n".join(
        [
            f"QUESTION: {render(q['instructions'])}",
            f"PROPOSED ANSWER: {proposed}",
            "Is the proposed answer correct for this state? Answer with Yes or No.",
        ]
    )


def score_listwise_block(q: dict) -> str:
    lines = [f"QUESTION: {render(q['instructions'])}", "LEVELS:"]
    for letter, level in zip(LETTERS, q["criteria"]):
        lines.append(f"{letter}) {render_inline(level)}")
    lines.append("Answer with the letter of the level that fits best.")
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Readouts → typed answers


def class_logits(logprobs: torch.Tensor, classes: list[list[int]]) -> tuple[list[float], float]:
    """Combined log-prob per class (logsumexp over label variants), and the total label mass."""
    z = [torch.logsumexp(logprobs[ids], 0).item() for ids in classes]
    mass = sum(math.exp(v) for v in z)
    return z, mass


def softmax(z: list[float], temperature: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((v - m) / temperature) for v in z]
    s = sum(e)
    return [v / s for v in e]


# Both confidence statistics follow TypeSafe's own definitions in
# github.com/typesafe-ai/system-one-adapter-python (_utils/confidence_metrics.py, MIT), which
# reproduce all 16 documented examples (docs/RESEARCH.md §3.3).


def choice_confidence(p: list[float]) -> float:
    """Peak probability rescaled from uniform (0) to certain (1)."""
    u = 1 / len(p)
    return (max(p) - u) / (1 - u)


def score_confidence(p: list[float]) -> float:
    """Ordinal: 1 − E|level − mode| / E|level − centre| under a uniform distribution, floored at 0.

    Probability on levels far from the mode costs more than on neighbouring levels.
    """
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    spread = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    uniform_spread = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
    return max(0.0, 1 - spread / uniform_spread)


def raw_scores(engine: Engine, req: dict, mode: str = "packed") -> tuple[dict, dict]:
    """Per question: the uncalibrated logits the answer is built from, plus label mass and usage."""
    validate(req)
    prefix = engine.prefix_ids(req["state"])
    branches = [b for qid, q in req["questions"].items() for b in branches_for(engine, qid, q)]
    t0 = time.perf_counter()
    lp = engine.readouts(prefix, branches, mode)
    elapsed = time.perf_counter() - t0
    raw: dict = {qid: {"logits": [], "mass": []} for qid in req["questions"]}
    for b, row in zip(branches, lp):
        z, mass = class_logits(row, b.classes)
        if b.pointwise:
            raw[b.question]["logits"].append(z[0] - z[1])  # this level's (or option's) yes/no log-odds
        else:
            raw[b.question]["logits"] = z
        raw[b.question]["mass"].append(mass)
    usage = {
        "input_tokens": len(prefix) + sum(len(b.ids) for b in branches),
        "output_tokens": 0,
        "latency_ms": round(elapsed * 1000),  # minijev extension; not in Jev's usage object
    }
    return raw, usage


def answer(q: dict, logits: list[float], temperature: float = 1.0) -> dict:
    if q["type"] == "noul":
        return {"type": "noul", "noul": 1 / (1 + math.exp(-(logits[0] - logits[1]) / temperature))}
    p = softmax(logits, temperature)
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return {
            "type": "choice",
            "choice": keys[max(range(len(p)), key=p.__getitem__)],
            "probabilities": dict(zip(keys, p)),
            "confidence": choice_confidence(p),
        }
    return {
        "type": "score",
        "score": sum(i * pi for i, pi in enumerate(p)),
        "legend": {str(i): level for i, level in enumerate(q["criteria"])},
        "probabilities": {str(i): pi for i, pi in enumerate(p)},
        "confidence": score_confidence(p),
    }


def rounded(x):
    if isinstance(x, float):
        return round(x, 2)
    if isinstance(x, dict):
        return {k: v if k == "legend" else rounded(v) for k, v in x.items()}
    return x


def ask(engine: Engine, req: dict, mode: str = "packed", temperatures: dict | None = None, debug: bool = False) -> dict:
    """The Jev contract: {state, questions} in, {model, answers, usage} out."""
    raw, usage = raw_scores(engine, req, mode)
    temps = temperatures or {}
    answers, warnings = {}, []
    for qid, q in req["questions"].items():
        a = answer(q, raw[qid]["logits"], temps.get(q["type"], 1.0))
        answers[qid] = a if debug else rounded(a)
        low = min(raw[qid]["mass"])
        if low < 0.5:  # the model wanted to say something other than a label: prompt problem
            warnings.append(f"{qid}: only {low:.2f} of next-token mass is on the labels")
    resp = {"model": f"minijev-poc ({engine.name})", "answers": answers, "usage": usage}
    if debug:
        resp["debug"] = {"raw": raw, "warnings": warnings}
    return resp
