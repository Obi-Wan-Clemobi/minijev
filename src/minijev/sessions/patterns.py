"""Decision-pattern questions (openspec/changes/add-decision-patterns): what kind of work a call does, which tool does
it, and which calls are waste. Labels come from the log by the fixed table below, never from labellers.

The table was written on 2026-09-26 from the inventory of tool names in our logs, before any fold result existed. A
change to it gives a new version.

Kind of work, from the call:
- inspect      read without changing: Read, Grep, Glob; Bash inspect; git status/diff/log/show/fetch; an MCP tool whose
               verb reads (get, list, load, lookup, search, read, find, view); a browser read (find, page text,
               console, tabs context)
- change       change local files or local app data: Edit, Write; git add/commit/stash/checkout/rm/branch; an MCP
               tool whose verb writes (add, update, remove, create, delete, set, save)
- run          run a program, tests or a build, or change files by shell: Bash run, and every Bash command whose
               first word the Bash kind list does not know (59 of 2922 Bash calls in our logs, Measured)
- remote       act on a remote machine or cloud service by shell: Bash remote (ssh, gcloud, curl, …)
- research     find information outside the project: WebSearch, WebFetch, a knowledge MCP tool (Wolfram)
- browse       act in a web browser: browser clicks, navigation, tabs
- publish      make something visible to others: git push, gh (pr, api, repo, release), send a message or email, share
               a file, publish an artifact, send a file to the user
- ask          ask the user: AskUserQuestion
- orchestrate  manage the agent's own work: Agent, SendMessage, Skill, ToolSearch, Monitor, TaskStop, plan mode, …

A tool id is the tool name, except that every MCP tool is "mcp" and every browser tool is "browser": server names
such as "travel" name a domain, and the goal is patterns that hold across domains.
"""

from __future__ import annotations

import re

from .logs import bash_kind, strip_bash
from .questions import USER_CHARS, Question, previous_call, register, state

WORK_KINDS = ["inspect", "change", "run", "remote", "research", "browse", "publish", "ask", "orchestrate"]
READ_VERBS = re.compile(r"(^|_)(get|list|load|lookup|search|read|find|view|query|fetch)(_|$)")
WRITE_VERBS = re.compile(r"(^|_)(add|update|remove|create|delete|set|save|upload|move|copy)(_|$)")
PUBLISH_VERBS = re.compile(r"(^|_)(send|share|publish|post|reply|forward)(_|$)")
BROWSER_READS = {"find", "get_page_text", "read_page", "read_console_messages", "read_network_requests",
                 "tabs_context_mcp", "list_connected_browsers"}
KNOWLEDGE_MCP = re.compile(r"wolfram|context7|hugging_face|docs", re.I)
GIT_READS = {"status", "diff", "log", "show", "fetch", "blame", "--no-pager", "ls-files", "rev-parse"}
GIT_PUBLISH = re.compile(r"^(git\s+push|gh\s+(pr|api|repo|release|issue))\b")
PUBLISH_TOOLS = {"Artifact", "SendUserFile", "PushNotification"}

# The tools that can do each kind of work. A kind with one tool has no tool question.
TOOLS_BY_KIND = {
    "inspect": ["Read", "Grep", "Glob", "Bash", "mcp", "browser"],
    "change": ["Edit", "Write", "Bash", "mcp"],
    "research": ["WebSearch", "WebFetch", "mcp"],
    "publish": ["Bash", "mcp", "Artifact", "SendUserFile", "PushNotification"],
    "orchestrate": ["Agent", "SendMessage", "Skill", "ToolSearch", "Monitor", "other"],
}


def tool_id(tool: str) -> str:
    if tool.startswith("mcp__claude-in-chrome"):
        return "browser"
    if tool.startswith("mcp__"):
        return "mcp"
    return {"MultiEdit": "Edit", "NotebookEdit": "Edit"}.get(tool, tool)


def work_kind(call: dict) -> str:
    """The kind of work of one call (a call dict of logs.parse), by the table in the module docstring."""
    tool, summary = call["tool"], call["summary"]
    if tool == "AskUserQuestion":
        return "ask"
    if tool in ("Read", "Grep", "Glob"):
        return "inspect"
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        return "change"
    if tool in ("WebSearch", "WebFetch"):
        return "research"
    if tool in PUBLISH_TOOLS:
        return "publish"
    if tool.startswith("mcp__claude-in-chrome"):
        return "inspect" if tool.split("__")[-1] in BROWSER_READS else "browse"
    if tool.startswith("mcp__"):
        verb = tool.split("__")[-1].lower()
        if PUBLISH_VERBS.search(verb):
            return "publish"
        if KNOWLEDGE_MCP.search(tool):
            return "research"
        if WRITE_VERBS.search(verb):
            return "change"
        return "inspect"   # READ_VERBS, and unknown verbs: an MCP call with no write or publish verb reads
    if tool == "Bash":
        command = strip_bash(summary)
        if GIT_PUBLISH.match(command):
            return "publish"
        if call["kind"] == "git":
            words = command.split()
            return "inspect" if len(words) > 1 and words[1] in GIT_READS else "change"
        return {"inspect": "inspect", "run": "run", "remote": "remote"}.get(call["kind"], "run")
    return "orchestrate"


def _calls(s):
    for k, c in enumerate(s["calls"]):
        if c["turn"] is not None:
            yield f"{s['session']}:{k}", c, s["calls"][:k]


def _work_kind_rows(s):
    for rid, c, before in _calls(s):
        yield rid, state(s, s["turns"][c["turn"]], before, msg=c["msg"]), WORK_KINDS.index(work_kind(c))


def _tool_rows(kind: str):
    options = TOOLS_BY_KIND[kind]

    def rows(s):
        for rid, c, before in _calls(s):
            if work_kind(c) != kind:
                continue
            t = tool_id(c["tool"])
            yield rid, state(s, s["turns"][c["turn"]], before, msg=c["msg"]), options.index(t if t in options else options[-1])
    return rows


def _waste_rows(s):
    """A call is waste if it was rejected; or it failed and the next call of the turn uses the same tool (a retry); or
    it reads a file that the turn already read, with no edit of that file between."""
    calls = s["calls"]
    for rid, c, before in _calls(s):
        k = int(rid.split(":")[1])
        nxt = calls[k + 1] if k + 1 < len(calls) and calls[k + 1]["turn"] == c["turn"] else None
        retry = c["status"] == "error" and nxt is not None and nxt["tool"] == c["tool"]
        reread = False
        if c["tool"] == "Read":
            for b in reversed([b for b in before if b["turn"] == c["turn"]]):
                if b["summary"] == c["summary"] and b["tool"] in ("Edit", "Write", "MultiEdit"):
                    break
                if b["tool"] == "Read" and b["summary"] == c["summary"]:
                    reread = True
                    break
        y = int(c["status"] == "rejected" or retry or reread)
        yield rid, state(s, s["turns"][c["turn"]], before, c, msg=c["msg"]), y


def _last_kind(text: str) -> str:
    """For the previous-call baseline: the kind of work of the last call that the state lists."""
    tool, summary, _ = previous_call(text)
    if tool == "none":
        return "none"
    return work_kind({"tool": tool, "summary": summary, "kind": bash_kind(summary) if tool == "Bash" else None})


def _last_tool(text: str) -> str:
    return tool_id(previous_call(text)[0])


register(Question(
    "work_kind", "choice", WORK_KINDS,
    {"instructions": "What kind of work will the coding agent do next?", "criteria": {
        "inspect": "Read or search without changing anything", "change": "Change local files or local app data",
        "run": "Run a program, tests or a build", "remote": "Act on a remote machine or cloud service",
        "research": "Find information outside the project", "browse": "Act in a web browser",
        "publish": "Make something visible to other people", "ask": "Ask the user a question",
        "orchestrate": "Manage its own work: agents, skills, tool search, monitors"}},
    _work_kind_rows, condition=_last_kind))

for _kind, _tools in TOOLS_BY_KIND.items():
    register(Question(
        f"tool_{_kind}", "choice", _tools,
        {"instructions": f"The coding agent will {_kind} next. Which tool will it use?",
         "criteria": {t: None for t in _tools}},
        _tool_rows(_kind), condition=_last_tool))

register(Question(
    "waste", "noul", ["no", "yes"],
    {"instructions": "Will this next call be wasted (rejected, retried after a failure, or a repeated read)?"},
    _waste_rows, condition=lambda text: text.split("\nNext call:\n[")[1].split("]")[0]))


AREAS = ["frontend", "backend", "data", "infra", "ml", "docs", "agent-setup", "personal-admin", "other"]


def _cut(text: str, n: int) -> str:
    return text[:n] + (" …" if len(text) > n else "")


def _area_rows(s):
    """One decision point per turn of an interactive session (a program writes the turns of an SDK session). The
    state is the request, and the previous request of the session for context (a reply such as "go" names no area
    by itself); no project line and no calls, so a labeller (and a model) judges the area of work, not the domain."""
    if not s.get("interactive", False):   # unknown counts as not interactive, as in logs.parse
        return
    previous = None
    for t in s["turns"]:
        context = f"Previous request:\n{_cut(previous, 300)}\n" if previous else ""
        yield f"{s['session']}:t{t['i']}", context + "User request:\n" + _cut(t["text"], USER_CHARS), None
        previous = t["text"]


register(Question(
    "area", "choice", AREAS,
    {"instructions": "Which technical area is this request about?", "criteria": {
        "frontend": "User interface, pages, styling, mockups", "backend": "Server code, APIs, command-line tools",
        "data": "Databases, data files, migrations, queries", "infra": "Deploys, cloud, servers, CI, networking",
        "ml": "Models, training, evaluation, prompts", "docs": "Documents, specs, plans, reports",
        "agent-setup": "The coding agent's own settings, hooks, skills, memory",
        "personal-admin": "Email, calendar, files, accounts, home devices", "other": "None of these"}},
    _area_rows, label_source="external"))
