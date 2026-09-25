"""The model, and the three ways to evaluate a prefix tree: naive, kv and packed."""

from __future__ import annotations

from dataclasses import dataclass

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


class Engine:
    def __init__(self, model: str | None = None, attn: str | None = None, threads: int | None = None):
        s = Settings.load() if None in (model, attn, threads) else Settings()
        model, attn, threads = model or s.model, attn or s.attn, threads or s.threads
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
