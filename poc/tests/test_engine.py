"""Slow tests: load Qwen2.5-0.5B-Instruct. Run `uv run pytest -m "not model"` to skip them."""

import pytest
import torch
from transformers import DynamicCache

from experiments import GDPR_QUESTIONS, as_choice, generate_batched, generate_logprobs
from minijev import MODEL, MODES, Engine, choice_block, raw_scores

pytestmark = pytest.mark.model

STATE = "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP."
QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
        "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages", "sales": "Pricing"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
}


@pytest.fixture(scope="module")
def engine():
    return Engine(MODEL, attn="eager", threads=6)  # fixed: ignore minijev.env


def gap(a: dict, b: dict) -> float:
    return max(abs(x - y) for qid in a for x, y in zip(a[qid]["logits"], b[qid]["logits"]))


def test_labels_are_single_tokens(engine):
    # Engine._variants asserts this at load time; check the result is non-empty and disjoint.
    assert engine.yes and engine.no and not set(engine.yes) & set(engine.no)
    assert len({t for group in engine.letters for t in group}) == sum(map(len, engine.letters))


def test_three_modes_agree(engine):
    req = {"state": STATE, "questions": QUESTIONS}
    raws = {m: raw_scores(engine, req, m)[0] for m in MODES}
    assert gap(raws["naive"], raws["kv"]) < 1e-3
    assert gap(raws["naive"], raws["packed"]) < 1e-3


def test_batched_equals_single(engine):
    together = raw_scores(engine, {"state": STATE, "questions": QUESTIONS})[0]
    for qid, q in QUESTIONS.items():
        alone = raw_scores(engine, {"state": STATE, "questions": {qid: q}})[0]
        assert gap({qid: together[qid]}, alone) < 1e-3


def test_one_generated_token_with_logprobs_is_a_readout(engine):
    q = QUESTIONS["team"]
    prefix, suffix = engine.prefix_ids(STATE), engine.suffix_ids(choice_block(q))
    readout = raw_scores(engine, {"state": STATE, "questions": {"q": q}}, "kv")[0]["q"]["logits"]
    lp = generate_logprobs(engine, prefix + suffix)
    via_api = [torch.logsumexp(lp[ids], 0).item() for ids in engine.letters[:3]]
    assert max(abs(a - b) for a, b in zip(readout, via_api)) < 1e-3


def test_batched_decode_matches_sequential_greedy(engine):
    prefix = engine.prefix_ids(STATE)
    qs = [as_choice(q) for q in list(GDPR_QUESTIONS.values())[:4]]
    suffixes = [engine.suffix_ids(choice_block(q)) for q in qs]  # different lengths: exercises the padding
    assert len(set(map(len, suffixes))) > 1
    batched = generate_batched(engine, prefix, suffixes, 6)
    for s, got in zip(suffixes, batched):
        x = torch.tensor([prefix + s])
        with torch.inference_mode():
            ref = engine.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=6, do_sample=False,
                                        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                        pad_token_id=engine.tok.eos_token_id)[0, x.shape[1]:].tolist()
        assert got == ref


def test_averaged_choice_ignores_the_option_order(engine):
    # An ambiguous case, so that the listwise answer can depend on the order.
    q = {"type": "choice", "instructions": "What is the topic of this news article?", "choice_mode": "averaged",
         "criteria": {"World": "Politics", "Sports": "Sports", "Business": "Companies", "Technology": "Science"}}
    state = "The league's owners approved a new streaming deal with a tech giant worth billions."
    keys = list(q["criteria"])
    probs = []
    for r in range(len(keys)):
        order = keys[r:] + keys[:r]
        qq = {**q, "criteria": {k: q["criteria"][k] for k in order}}
        z = raw_scores(engine, {"state": state, "questions": {"q": qq}})[0]["q"]["logits"]
        probs.append({k: torch.tensor(v).exp().item() for k, v in zip(order, z)})
    for p in probs[1:]:
        assert max(abs(p[k] - probs[0][k]) for k in keys) < 1e-4


SHARED = {**QUESTIONS,
          "team_pw": {**QUESTIONS["team"], "choice_mode": "pointwise"},
          "team_avg": {**QUESTIONS["team"], "choice_mode": "averaged"}}


@pytest.mark.parametrize("mode", MODES)
def test_shared_question_head_keeps_the_logits(engine, mode):
    # Task 4.1: the two-level tree (state -> question -> item) gives the same logits as one flat branch per item.
    req = {"state": STATE, "questions": SHARED}
    flat, flat_usage = raw_scores(engine, req, mode)
    tree, tree_usage = raw_scores(engine, req, mode, share_question=True)
    assert gap(flat, tree) < 1e-3
    assert tree_usage["input_tokens"] < flat_usage["input_tokens"]


def test_pack_tree_mask(engine):
    # prefix 2, one head of 2 with two children (1 and 2 tokens), then a flat branch of 1 token.
    ids_len = 2 + 2 + 1 + 2 + 1
    pos, mask, last = engine.pack_tree(2, [(2, [1, 2]), (0, [1])])
    m = mask.reshape(ids_len, ids_len)
    see = lambda i, j: bool(m[i, j] == 0) if m.dtype.is_floating_point else bool(m[i, j])
    assert see(4, 2) and see(4, 3) and see(5, 3) and see(6, 3)   # children see their head
    assert not see(5, 4) and not see(4, 5)                       # siblings do not see each other
    assert not see(7, 2) and see(7, 0)                           # the flat branch sees the state, not the head
    assert pos[0].tolist() == [0, 1, 2, 3, 4, 4, 5, 2] and last.tolist() == [4, 6, 7]


@pytest.mark.parametrize("mode", ["kv", "packed"])
def test_state_cache_keeps_the_logits(engine, mode):
    # Task 4.2: a state from the cache gives the same logits as a state prefilled again.
    from minijev.engine import StateCache
    req = {"state": STATE, "questions": SHARED}
    plain = raw_scores(engine, req, mode, share_question=True)[0]
    engine.state_cache = StateCache(4)
    try:
        first = raw_scores(engine, req, mode, share_question=True)[0]   # miss: prefill and store
        second = raw_scores(engine, req, mode, share_question=True)[0]  # hit: no prefill
        stats = engine.state_cache.stats()
    finally:
        engine.state_cache = StateCache(0)
    assert gap(plain, first) < 1e-3 and gap(plain, second) < 1e-3
    assert (stats["hits"], stats["misses"], stats["entries"]) == (1, 1, 1)


def test_state_cache_evicts_the_oldest():
    from minijev.engine import StateCache
    c = StateCache(2)
    for k in [(1,), (2,), (1,), (3,)]:  # (1,) is used again, so (2,) is the oldest when (3,) arrives
        if c.get(k) is None:
            c.put(k, "kv")
    assert list(c.entries) == [(1,), (3,)] and c.hits == 1
