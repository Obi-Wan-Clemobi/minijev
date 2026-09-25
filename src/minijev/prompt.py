"""Prompt pieces: the system line, the answer labels, and one template per question type."""

from __future__ import annotations

import hashlib
import json

SYSTEM = "You are a precise classifier. You read a STATE and answer one QUESTION about it."


CONTENT_FREE_STATE = "N/A"


# "I" is skipped: it is a common first word of an answer, so it attracts unrelated mass.
LETTERS = list("ABCDEFGHJKLMNOPQRSTUVWXYZ")


YES, NO = ["Yes", "yes", "YES"], ["No", "no", "NO"]


def render(value) -> str:
    """Text for a state, instruction or description. Structure stays as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_inline(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


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


def template_fingerprint() -> str:
    """SHA-256 of every prompt template as the code renders it, on fixed probe inputs, plus the labels and the
    system line. templates/v1-preregistration.json records it; tests/test_templates.py fails when it changes."""
    probe = {
        "system": SYSTEM, "content_free_state": CONTENT_FREE_STATE, "letters": LETTERS, "yes": YES, "no": NO,
        "noul": noul_block({"instructions": "<Q>"}),
        "noul_criteria": noul_block({"instructions": "<Q>", "criteria": {"true": "<T>", "false": "<F>"}}),
        "choice": choice_block({"instructions": "<Q>", "criteria": {"<a>": "<A>", "<b>": None}}),
        "proposal": proposal_block({"instructions": "<Q>"}, "<P>"),
        "score_listwise": score_listwise_block({"instructions": "<Q>", "criteria": ["<l0>", "<l1>"]}),
    }
    return hashlib.sha256(json.dumps(probe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
