"""Questions: what a model learns to answer from session data.

A Question is a name, a type (noul or choice), labels, a readout prompt, and a function that turns one parsed
session into rows (row id, state text, label). For a Noul, label 0 is "no" and label 1 is "yes".

label_source says where labels come from:
- "log": the label is what happened in the session (a real label).
- "external": rows() gives decision points with label None. Labels come from ROOT/sessions/labels/<name>.jsonl, one
  {"id", "y", "by"} per line, written by a person or by a model. Only labelled points become rows. `minijev
  sessions points NAME` writes the unlabelled points for a labeller.

A state holds only what existed before the decision. Tool results and Claude's text are never in a state: the results
are the largest source of secrets, and Claude's text before a call often names the call. A call from the same
assistant message as the decision is a parallel call: its status shows as "pending".

Add a question in a Python file and pass it with --questions FILE:

    from minijev.sessions import Question, register, state

    def rows(s):
        for k, c in enumerate(s["calls"]):
            if c["turn"] is not None and c["tool"] == "Bash":
                turn = s["turns"][c["turn"]]
                yield f"{s['session']}:{k}", state(s, turn, s["calls"][:k], c, msg=c["msg"]), int("pytest" in c["summary"])

    register(Question("runs_tests", "noul", ["no", "yes"], {"instructions": "Is this call a test run?"}, rows))
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

USER_CHARS, CALLS_KEPT = 600, 4   # state budget: mean 238, p90 375 tokens on our v1 data (Measured). 1500, 8 gave 404, 661

Row = tuple[str, str, "int | None"]


@dataclass(frozen=True)
class Question:
    name: str
    type: str                                   # "noul" or "choice"
    labels: list[str]
    prompt: dict                                # the minijev question for the zero-shot readout
    rows: Callable[[dict], Iterable[Row]]
    label_source: str = "log"                   # "log" or "external"
    condition: Callable[[str], str] | None = field(default=None)   # for the previous-call baseline: state -> condition

    def spec(self) -> dict:
        return {"type": self.type, "labels": self.labels, "prompt": self.prompt, "label_source": self.label_source}


QUESTIONS: dict[str, Question] = {}


def register(q: Question) -> Question:
    assert q.type in ("noul", "choice") and (q.type != "noul" or len(q.labels) == 2), q.name
    QUESTIONS[q.name] = q
    return q


def load(path: str | Path) -> None:
    """Import a file that calls register()."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def state(s: dict, turn: dict | None, before: list[dict], call: dict | None = None, msg: str | None = None) -> str:
    """The state text: the project (left out if s["project"] is None), the user turn, and the last calls of that turn
    before the decision. A call's status shows only if its result existed at decision time; calls from assistant
    message msg show "pending"."""
    lines = ([f"Project: {s['project']}"] if s.get("project") is not None else []) + ["User request:",
             (turn["text"][:USER_CHARS] + (" …" if len(turn["text"]) > USER_CHARS else "")) if turn else "(none)"]
    own = [c for c in before if turn is not None and c["turn"] == turn["i"]][-CALLS_KEPT:]
    lines.append("Calls so far in this turn:" if own else "Calls so far in this turn: none")
    lines += [f"- [{c['tool']}] {c['summary']} -> {'pending' if c['msg'] == msg else c['status']}" for c in own]
    if call is not None:
        lines += ["Next call:", f"[{call['tool']}] {call['summary']}"]
    return "\n".join(lines)


def previous_call(text: str) -> tuple[str, str, str]:
    """(tool, summary, status) of the last call that a state lists, or ("none", "", "none")."""
    listed = [l for l in text.split("\nNext call:")[0].split("\n") if l.startswith("- [")]
    if not listed:
        return "none", "", "none"
    tool, _, rest = listed[-1][3:].partition("] ")
    summary, _, status = rest.rpartition(" -> ")
    return tool, summary, status


def _calls(s: dict, keep: Callable[[dict], bool]) -> Iterable[tuple[str, dict, list[dict]]]:
    for k, c in enumerate(s["calls"]):
        if c["turn"] is not None and keep(c):
            yield f"{s['session']}:{k}", c, s["calls"][:k]


# Built-in questions ---------------------------------------------------------------------------------------------------

from .logs import bash_kind, tool_group  # noqa: E402

NEXT_TOOL = ["bash", "edit", "read", "web", "browser", "ask", "mcp", "other"]
BASH_KIND = ["inspect", "run", "git", "remote", "other"]


def _next_tool(s):
    for rid, c, before in _calls(s, lambda c: True):
        yield rid, state(s, s["turns"][c["turn"]], before, msg=c["msg"]), NEXT_TOOL.index(c["group"])


def _bash_kind(s):
    for rid, c, before in _calls(s, lambda c: c["tool"] == "Bash"):
        yield rid, state(s, s["turns"][c["turn"]], before, msg=c["msg"]), BASH_KIND.index(c["kind"])


def _will_fail(s):
    for rid, c, before in _calls(s, lambda c: c["status"] in ("ok", "error") and c["tool"] != "AskUserQuestion"):
        yield rid, state(s, s["turns"][c["turn"]], before, c, msg=c["msg"]), int(c["status"] == "error")


def _decision_points(s):
    """Every call, with the call in the state: the moment before the agent acts. No label."""
    for rid, c, before in _calls(s, lambda c: c["tool"] != "AskUserQuestion"):
        yield rid, state(s, s["turns"][c["turn"]], before, c, msg=c["msg"]), None


def _last_group(text):
    tool = previous_call(text)[0]
    return tool_group(tool) if tool != "none" else "none"


def _last_bash_kind(text):
    bash = [l for l in text.split("\n") if l.startswith("- [Bash] ")]
    return bash_kind(bash[-1][9:].rsplit(" -> ", 1)[0]) if bash else "none"


def _status_and_tool(text):
    return previous_call(text)[2] + "|" + text.split("\nNext call:\n[")[1].split("]")[0]


register(Question(
    "next_tool", "choice", NEXT_TOOL,
    {"instructions": "Which kind of tool will the coding agent call next?", "criteria": {
        "bash": "Run a shell command", "edit": "Edit or write a file", "read": "Read or search files",
        "web": "Search the web or fetch a web page", "browser": "Control the web browser",
        "ask": "Ask the user a question", "mcp": "Call a project-specific service tool",
        "other": "Another tool (skills, subagents, tool search, tasks)"}},
    _next_tool, condition=_last_group))
register(Question(
    "bash_kind", "choice", BASH_KIND,
    {"instructions": "What kind of shell command will the coding agent run next?", "criteria": {
        "inspect": "Look at files or system state without changing them",
        "run": "Run a program, tests or a build, or change files", "git": "Use git or GitHub",
        "remote": "Call a remote service or a cloud command-line tool", "other": "Another kind of command"}},
    _bash_kind, condition=_last_bash_kind))
register(Question(
    "will_fail", "noul", ["no", "yes"], {"instructions": "Will the next call fail with an error?"},
    _will_fail, condition=_status_and_tool))
register(Question(
    "needs_approval", "noul", ["no", "yes"],
    {"instructions": "Should the agent ask the user for approval before it makes the next call?"},
    _decision_points, label_source="external", condition=_status_and_tool))
