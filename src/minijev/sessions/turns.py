"""Turn cost (openspec/changes/add-turn-cost-routing): how many assistant messages a turn will take, asked when the
user message arrives.

The **cost of a turn** is its number of assistant messages; a **long turn** has 31 or more. The levels follow the share
of cache reads they hold (Measured on 256 interactive turns, 2026-09-26): 0–2 (1.9%), 3–8 (7.4%), 9–30 (29.1%),
31 or more (61.7%).

The state holds only what exists when the user message arrives: the user message, the previous user message, the end
of the previous turn's last assistant text, the number of earlier turns, the level of the previous turn, and the
context size (the cache read of the last assistant message before the turn). Nothing of the turn itself, and no project
name. Other questions leave assistant text out because text before a call can name the call; this text comes from an
earlier turn, and holds the plan that a short "go" approves. Interactive sessions only: in an SDK session a program
writes the turns.
"""

from __future__ import annotations

from collections import Counter

from .questions import USER_CHARS, Question, register

LEVELS = ["0-2", "3-8", "9-30", "31+"]
CUTS = (2, 8, 30)                  # the highest message count of each level but the last
CONTEXT_CHARS = 300                # the previous user message and the previous assistant text, each


def level_of(messages: int) -> int:
    return next((k for k, cut in enumerate(CUTS) if messages <= cut), len(CUTS))


def _cut(text: str, n: int) -> str:
    return text[:n] + (" …" if len(text) > n else "")


def _tail(text: str, n: int) -> str:
    return ("… " if len(text) > n else "") + text[-n:]


def messages_per_turn(s: dict) -> Counter:
    return Counter(m["turn"] for m in s.get("messages", []) if m["turn"] is not None)


def context_before(s: dict) -> dict[int, int]:
    """For each turn, the cache read of the last assistant message before it (0 at the session start)."""
    out, last = {}, 0
    by_turn: dict[int, int] = {}
    for m in s.get("messages", []):
        if m["turn"] is not None:
            by_turn[m["turn"]] = m["tokens"].get("cache_read_input_tokens", 0)   # messages are in order
    for t in s["turns"]:
        out[t["i"]] = last
        last = by_turn.get(t["i"], last)
    return out


def state(s: dict, i: int, counts: Counter, context: dict[int, int]) -> str:
    t, prev = s["turns"][i], s["turns"][i - 1] if i > 0 else None
    lines = ["User message:", _cut(t["text"], USER_CHARS)]
    if prev is None:
        lines += ["Earlier turns: none (the session starts here)"]
    else:
        lines += ["Previous user message:", _cut(prev["text"], CONTEXT_CHARS)]
        if prev.get("last_text"):
            lines += ["End of the previous answer:", _tail(prev["last_text"], CONTEXT_CHARS)]
        lines += [f"Earlier turns: {i}", f"Previous turn: {LEVELS[level_of(counts[prev['i']])]} assistant messages"]
    lines.append(f"Context size: {round(context[t['i']] / 1000)} thousand tokens")
    return "\n".join(lines)


def _rows(s):
    if not s.get("interactive", False):   # unknown counts as not interactive, as in logs.parse
        return
    counts, context = messages_per_turn(s), context_before(s)
    for t in s["turns"]:
        yield f"{s['session']}:t{t['i']}", state(s, t["i"], counts, context), level_of(counts[t["i"]])


def previous_level(text: str) -> str:
    """For the previous-turn baseline: the previous turn's level as the state states it, or "none"."""
    line = next((l for l in text.split("\n") if l.startswith("Previous turn: ")), None)
    return line.split(": ", 1)[1].split(" ")[0] if line else "none"


register(Question(
    "turn_cost", "score", LEVELS,
    {"instructions": "How many assistant messages will the coding agent need for this user message?",
     "criteria": ["0 to 2", "3 to 8", "9 to 30", "31 or more"]},
    _rows, condition=previous_level))
