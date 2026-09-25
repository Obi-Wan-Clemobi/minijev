"""The model, and the three ways to evaluate a prefix tree: naive, kv and packed."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .prompt import LETTERS, NO, SYSTEM, YES, render
from .settings import Settings

MODES = ("naive", "kv", "packed")


@dataclass
class Branch:
    """One path from the end of the state to a readout position."""

    ids: list[int]
    classes: list[list[int]]  # per answer class: the token ids (label variants) it owns
    question: str  # question id this branch belongs to
    pointwise: bool = False  # one of several yes/no branches (a Score level or a Choice option)
    order: list[int] | None = None  # averaged mode: the option index shown at each letter position
    head: tuple[int, ...] = ()  # shared question tokens (two-level tree); sibling branches of a question share it


def groups(branches: list[Branch]) -> list[tuple[tuple[int, ...], list[Branch]]]:
    """Consecutive branches of one question that share a head, as (head, branches); unshared branches alone."""
    out: list = []
    for b in branches:
        if b.head and out and out[-1][0] == b.head and out[-1][1][0].question == b.question:
            out[-1][1].append(b)
        else:
            out.append((b.head, [b]))
    return out


class StateCache:
    """The key/value tensors of recent states, so a repeated state is not prefilled again (Task 4.2, W13).

    Keyed by the prefix token ids. Least recently used entries go first. size 0 turns it off. The stored tensors are
    never changed: the model appends to a copy (torch.cat) and crop() only slices.
    Memory: Qwen2.5-0.5B keeps about 24 KB per token (24 layers x 2 x 2 heads x 64 x 4 bytes), so a 600-token state
    is about 15 MB; Qwen2.5-1.5B keeps about 57 KB per token.
    """

    def __init__(self, size: int = 0):
        self.size, self.entries = size, OrderedDict()
        self.hits = self.misses = self.tokens_saved = 0

    def get(self, key: tuple[int, ...]):
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            self.tokens_saved += len(key)
            return self.entries[key]
        self.misses += 1
        return None

    def put(self, key: tuple[int, ...], layers) -> None:
        self.entries[key] = layers
        self.entries.move_to_end(key)
        while len(self.entries) > self.size:
            self.entries.popitem(last=False)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"enabled": self.size > 0, "size": self.size, "entries": len(self.entries), "hits": self.hits,
                "misses": self.misses, "hit_rate": self.hits / n if n else None, "tokens_saved": self.tokens_saved}


class Engine:
    def __init__(self, model: str | None = None, attn: str | None = None, threads: int | None = None,
                 adapter: str | None = None):
        """adapter: a LoRA directory (poc/train_lora.py). It is merged into the weights, so every mode runs as before.
        Needs peft (uv group "train")."""
        s = Settings.load() if None in (model, attn, threads) else Settings()
        model, attn, threads = model or s.model, attn or s.attn, threads or s.threads
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float32, attn_implementation=attn
        ).eval()
        self.name = model
        self.adapter = self.adapter_sha256 = None
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().eval()
            weights = next(p for p in (Path(adapter) / "adapter_model.safetensors", Path(adapter) / "adapter_model.bin")
                           if p.exists())
            self.adapter, self.adapter_sha256 = str(adapter), hashlib.sha256(weights.read_bytes()).hexdigest()
        self.state_cache = StateCache(0)  # off; the server sets the size from MINIJEV_STATE_CACHE
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
            rows = [self.model(torch.tensor([prefix + list(b.head) + b.ids]), logits_to_keep=1).logits[0, -1]
                    for b in branches]
        elif mode == "kv":
            cache = self.prefilled(prefix)
            rows = []
            for head, group in groups(branches):
                if head:  # the shared question: once, then every level on top of it
                    self.model(torch.tensor([list(head)]), past_key_values=cache, use_cache=True, logits_to_keep=1)
                for b in group:
                    out = self.model(torch.tensor([b.ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    rows.append(out.logits[0, -1])
                    cache.crop(len(prefix) + len(head))  # back to state (+ question) for the next branch
                cache.crop(len(prefix))
        elif mode == "packed":
            tree = groups(branches)
            pos, mask, last = self.pack_tree(len(prefix), [(len(h), [len(b.ids) for b in g]) for h, g in tree])
            tokens = prefix + [t for h, g in tree for t in list(h) + [t for b in g for t in b.ids]]
            if self.state_cache.size:  # the state from the cache (or prefilled once and stored); the tree on top
                n = len(prefix)
                out = self.model(torch.tensor([tokens[n:]]), past_key_values=self.prefilled(prefix), use_cache=True,
                                 attention_mask=mask[:, :, n:, :], position_ids=pos[:, n:], logits_to_keep=last - n)
            else:
                out = self.model(torch.tensor([tokens]), attention_mask=mask, position_ids=pos, logits_to_keep=last)
            rows = list(out.logits[0])
        else:
            raise ValueError(mode)
        return torch.log_softmax(torch.stack(rows).float(), dim=-1)

    def prefilled(self, prefix: list[int]) -> DynamicCache:
        """A cache that holds the state's keys and values: from the state cache when it has them, else computed."""
        key = tuple(prefix)
        layers = self.state_cache.get(key) if self.state_cache.size else None
        if layers is not None:
            return DynamicCache.from_legacy_cache(layers)
        cache = DynamicCache()
        self.model(torch.tensor([prefix]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        if self.state_cache.size:
            self.state_cache.put(key, cache.to_legacy_cache())
        return cache

    def pack_tree(self, n_prefix: int, tree: list[tuple[int, list[int]]]):
        """Position ids, additive 4D mask and readout indices for [prefix][head1][b1a][b1b]...[head2][b2a]...

        tree: per group, (length of its shared head, lengths of its branches); a head of length 0 means no sharing.
        Token i may attend to token j iff j <= i and (j is in the prefix, or in i's own node, or in the head that is
        the parent of i's branch). A head's positions start at n_prefix; its branches' positions continue after it.
        """
        total = n_prefix + sum(h + sum(ls) for h, ls in tree)
        segment = torch.zeros(total, dtype=torch.long)  # 0 = prefix; every head and branch has its own id
        parent = [0]  # parent[node id] = the node it may also see (0 = the prefix only)
        pos = torch.arange(total)
        last, start = [], n_prefix
        for h, lengths in tree:
            head_id = 0
            if h:
                head_id = len(parent)
                parent.append(0)
                segment[start : start + h] = head_id
                pos[start : start + h] = torch.arange(n_prefix, n_prefix + h)
                start += h
            for n in lengths:
                node = len(parent)
                parent.append(head_id)
                segment[start : start + n] = node
                pos[start : start + n] = torch.arange(n_prefix + h, n_prefix + h + n)
                last.append(start + n - 1)
                start += n
        par = torch.tensor(parent)[segment]
        causal = torch.ones(total, total, dtype=torch.bool).tril()
        visible = causal & ((segment[None, :] == 0) | (segment[None, :] == segment[:, None]) | (segment[None, :] == par[:, None]))
        mask = torch.zeros(total, total, dtype=torch.float32)
        mask.masked_fill_(~visible, torch.finfo(torch.float32).min)
        return pos[None, :], mask[None, None], torch.tensor(last)

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
