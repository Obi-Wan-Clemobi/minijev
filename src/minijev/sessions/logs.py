"""Parse Claude Code session logs into turns and calls.

A session log is one .jsonl file in ~/.claude/projects/<folder>/. Its format has no public documentation and can change
between Claude Code versions, so parse() counts what it could not use (Diagnostics). A large "unknown status" or
"no turns" count means that the format changed.

Only the main thread is used. Subagent transcripts (a `subagents` folder, or entries with isSidechain) and meta
entries (command wrappers, system reminders, hook output) are left out.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

CLAUDE_PROJECTS = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"
SKIP_PREFIXES = ("<command-", "<local-command", "<system-reminder", "Caveat:", "<task-notification",
                 "[Request interrupted", "<bash-", "<user-prompt-submit-hook")
REJECTED = "The user doesn't want to proceed with this tool use"
CALL_CHARS = 200
INTERACTIVE = {"cli", "claude-desktop"}   # a person types the turns; "sdk-py", "sdk-cli", … are programs
TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")

TOOL_GROUPS = {"Bash": "bash", "Edit": "edit", "Write": "edit", "MultiEdit": "edit", "NotebookEdit": "edit",
               "Read": "read", "Grep": "read", "Glob": "read", "WebSearch": "web", "WebFetch": "web",
               "AskUserQuestion": "ask"}
BASH_KINDS = {
    "inspect": {"cat", "ls", "grep", "rg", "sed", "head", "tail", "find", "wc", "jq", "diff", "echo", "tree", "du",
                "stat", "file", "sort", "awk", "which", "pwd", "for", "while", "test", "[", "lsof", "ps", "open", "pgrep",
                "printf", "sqlite3", "pdftoppm", "command"},
    "run": {"python", "python3", "uv", "npm", "npx", "node", "pytest", "make", "docker", "docker-compose", "tilt",
            "bash", "sh", "zsh", "pip", "timeout", "kill", "pkill", "sleep", "until", "uvx", "brew", "openspec",
            "mkdir", "cp", "mv", "rm", "ln", "chmod", "touch"},   # file changes count as run: they act, not inspect
    "git": {"git", "gh"},
    "remote": {"curl", "wget", "gcloud", "ssh", "scp", "herdr", "gsutil", "firebase"},
}


def tool_group(name: str) -> str:
    if name in TOOL_GROUPS:
        return TOOL_GROUPS[name]
    if name.startswith("mcp__claude-in-chrome"):
        return "browser"
    if name.startswith("mcp__"):
        return "mcp"
    return "other"


def strip_bash(command: str) -> str:
    """The command without leading parts that say nothing about its kind: `cd X &&` (or a newline), `NAME=value`,
    `export`, `set -e`, `exec 2>&1`, `time` and an opening parenthesis."""
    c = command.strip()
    while True:
        m = re.match(r"^(?:\(|(?:cd|export|set)\s+\S+\s*(?:&&|;|\n)\s*|exec\s+\S+\s*|time\s+|"
                     r"[A-Za-z_][A-Za-z0-9_]*=\S*\s*(?:&&|;|\n)?\s*)", c)
        if not m or not m.group(0):
            return c
        c = c[m.end():]


def bash_kind(command: str) -> str:
    words = strip_bash(command).split()
    if not words:
        return "other"
    if words[0].startswith("./"):
        return "run"
    head = words[0].split("/")[-1]
    head = "python" if re.fullmatch(r"python[\d.]*", head) else head
    return next((k for k, heads in BASH_KINDS.items() if head in heads), "other")


def call_summary(name: str, inp: dict) -> str:
    if name == "Bash":
        s = strip_bash(inp.get("command", ""))
    elif "file_path" in inp:
        s = inp["file_path"]
    else:
        s = json.dumps(inp, ensure_ascii=False)
    s = " ".join(s.split())
    return s[:CALL_CHARS] + (" …" if len(s) > CALL_CHARS else "")


def user_text(entry: dict) -> str | None:
    """The real text a person typed, or None for meta entries, tool results and command wrappers."""
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isSidechain"):
        return None
    content = (entry.get("message") or {}).get("content")
    parts = [content] if isinstance(content, str) else [b.get("text", "") for b in content or []
                                                         if isinstance(b, dict) and b.get("type") == "text"]
    parts = [p.strip() for p in parts if p and p.strip() and not p.lstrip().startswith(SKIP_PREFIXES)]
    return "\n".join(parts) or None


class Diagnostics(Counter):
    """Counts of what parse() could not use. Printed by extract."""


def log_files(raw: Path) -> list[Path]:
    return sorted(p for p in raw.rglob("*.jsonl") if "subagents" not in p.parts)


def folders(path: Path) -> set[str]:
    """Every working folder that the log records."""
    out = set()
    for line in path.open():
        m = re.search(r'"cwd":\s*"([^"]+)"', line)
        if m:
            out.add(m.group(1))
    return out


def parse(path: Path, diag: Diagnostics | None = None) -> dict:
    """One session: {session, project, start, cwd, turns, calls}. Each call knows its turn, its assistant message
    (calls from one message are parallel) and its status: ok, error, rejected or unknown."""
    diag = diag if diag is not None else Diagnostics()
    entries = []
    for line in path.open():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            diag["bad json lines"] += 1
    results = {}
    for e in entries:
        content = (e.get("message") or {}).get("content")
        if e.get("type") == "user" and isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    s = json.dumps(b.get("content"), ensure_ascii=False)
                    results[b.get("tool_use_id")] = "rejected" if REJECTED in s else "error" if b.get("is_error") else "ok"
    cwd = next((e["cwd"] for e in entries if e.get("cwd")), "")
    entrypoint = next((e["entrypoint"] for e in entries if e.get("entrypoint")), "unknown")
    diag[f"sessions from entrypoint {entrypoint}"] += 1
    calls, turns, turn, seen = [], [], None, set()
    messages: dict[str, dict] = {}   # assistant message id -> {turn, calls, tokens}; streamed entries repeat an id
    for e in entries:
        if e.get("isSidechain"):
            diag["sidechain entries"] += 1
            continue
        text = user_text(e)
        if text is not None:
            turn = {"i": len(turns), "text": text, "ts": e.get("timestamp", "")}
            turns.append(turn)
            continue
        if e.get("type") != "assistant":
            continue
        mid = (e.get("message") or {}).get("id") or e.get("uuid")
        m = messages.setdefault(mid, {"turn": turn["i"] if turn else None, "calls": [], "tokens": {}})
        usage = (e.get("message") or {}).get("usage") or {}
        for k in TOKEN_FIELDS:   # a repeated entry carries the same or a later count: keep the largest
            m["tokens"][k] = max(m["tokens"].get(k, 0), usage.get(k) or 0)
        for b in (e.get("message") or {}).get("content") or []:
            if not (isinstance(b, dict) and b.get("type") == "tool_use") or b.get("id") in seen:
                continue
            seen.add(b.get("id"))   # a streamed assistant message can repeat in several entries
            inp = b.get("input") or {}
            call = {"msg": (e.get("message") or {}).get("id") or e.get("uuid"), "tool": b["name"],
                    "group": tool_group(b["name"]), "summary": call_summary(b["name"], inp),
                    "kind": bash_kind(inp.get("command", "")) if b["name"] == "Bash" else None,
                    "status": results.get(b.get("id"), "unknown"), "turn": turn["i"] if turn else None,
                    "ts": e.get("timestamp", "")}
            diag[f"calls with status {call['status']}"] += 1
            m["calls"].append(len(calls))
            calls.append(call)
    if not turns:
        diag["sessions without turns"] += 1
    return {"session": path.stem, "project": project_name(cwd), "start": turns[0]["ts"] if turns else "", "cwd": cwd,
            "entrypoint": entrypoint, "interactive": entrypoint in INTERACTIVE,
            "turns": turns, "calls": calls, "messages": list(messages.values())}


def project_name(cwd: str) -> str:
    """The working folder relative to home, without its first folder: ~/Code/app -> "app",
    ~/Code/app/web -> "app/web", ~ -> "home". A folder outside home gives its base name."""
    path, home = Path(cwd), Path.home()
    if not cwd or path == home:
        return "home"
    if not path.is_relative_to(home):
        return path.name
    parts = path.relative_to(home).parts
    return "/".join(parts[1:] if len(parts) > 1 else parts)


def sync(source: Path, archive: Path) -> int:
    """Copy new and grown logs from source to archive. It never deletes: the archive keeps logs that Claude Code
    removes after cleanupPeriodDays. Returns the number of files copied."""
    n = 0
    for src in source.rglob("*.jsonl"):
        dst = archive / src.relative_to(source)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n
