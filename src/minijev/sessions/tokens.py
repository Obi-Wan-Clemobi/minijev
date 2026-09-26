"""Where the tokens go: the token use of the assistant messages, per kind of work and per chain pattern.

Each assistant message records its usage: fresh input, cache read, cache write (cache creation) and output tokens. A
message's tokens are split evenly over the calls it makes, and each call counts toward its kind of work (patterns.py).
A message without calls (a text answer) counts as "answer". A turn's **chain pattern** is the sequence of the kinds of
its calls, with repeats merged ("inspect > change > run"). Cache reads cost much less than fresh input, so they are
reported apart.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .dataset import Paths, sessions
from .logs import TOKEN_FIELDS
from .patterns import work_kind

SHORT = {"input_tokens": "input", "cache_read_input_tokens": "cache_read", "cache_creation_input_tokens": "cache_write",
         "output_tokens": "output"}


def chain(kinds: list[str]) -> str:
    out = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
    return " > ".join(out) if out else "(no calls)"


def breakdown(all_sessions: list[dict], top: int = 15) -> dict:
    by_kind = defaultdict(Counter)
    by_turn = defaultdict(Counter)     # (session, turn) -> tokens
    kinds_of_turn = defaultdict(list)
    for s in all_sessions:
        for c in s["calls"]:
            if c["turn"] is not None:
                kinds_of_turn[(s["session"], c["turn"])].append(work_kind(c))
        for m in s.get("messages", []):
            t = {SHORT[k]: m["tokens"].get(k, 0) for k in TOKEN_FIELDS}
            names = [work_kind(s["calls"][k]) for k in m["calls"]] or ["answer"]
            for name in names:
                by_kind[name].update({k: v / len(names) for k, v in t.items()})
                by_kind[name]["calls"] += 1 if name != "answer" else 0
            if m["turn"] is not None:
                by_turn[(s["session"], m["turn"])].update(t)
    by_chain = defaultdict(Counter)
    for key, t in by_turn.items():
        c = chain(kinds_of_turn.get(key, []))
        by_chain[c].update(t)
        by_chain[c]["turns"] += 1
    total = Counter()
    for t in by_kind.values():
        total.update({k: v for k, v in t.items() if k != "calls"})
    chains = sorted(by_chain.items(), key=lambda kv: -(kv[1]["input"] + kv[1]["cache_write"] + kv[1]["output"]))
    return {"total": dict(total), "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
            "by_chain": {k: dict(v) for k, v in chains[:top]}, "chains_total": len(by_chain), "turns": len(by_turn)}


def report(paths: Paths) -> dict:
    out = breakdown(sessions(paths))
    folder = paths.out / "results"
    folder.mkdir(exist_ok=True)
    (folder / "tokens.json").write_text(json.dumps(out, indent=1))
    cols = ("input", "cache_read", "cache_write", "output")
    print(f"{'kind':12} {'calls':>6} " + " ".join(f"{c:>12}" for c in cols))
    for k, t in sorted(out["by_kind"].items(), key=lambda kv: -kv[1].get("cache_read", 0)):
        print(f"{k:12} {int(t.get('calls', 0)):6} " + " ".join(f"{int(t.get(c, 0)):12,}" for c in cols))
    print(f"{'total':12} {'':6} " + " ".join(f"{int(out['total'].get(c, 0)):12,}" for c in cols))
    print(f"\n{out['turns']} turns, {out['chains_total']} chain patterns; the most expensive "
          f"(fresh input + cache write + output):")
    for c, t in out["by_chain"].items():
        print(f"  {t['turns']:4} turns  in {int(t['input']):>9,}  cw {int(t['cache_write']):>11,}  "
              f"out {int(t['output']):>9,}  cr {int(t['cache_read']):>13,}  {c[:90]}")
    print(f"\nwrote {folder / 'tokens.json'}")
    return out
