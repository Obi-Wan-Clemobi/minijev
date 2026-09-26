"""Secret removal for session data. Two independent layers:

1. Patterns (SECRETS): key formats, emails, home paths, long random strings. scrub() replaces every match.
2. Literals: strings that no pattern knows about. These are the user's name, username and email (from git config and the
   home folder), every value in a .env file in a folder that a session worked in, and every line of
   ROOT/private-strings.txt. The user adds names, network names and addresses to that file after they review a
   sample (`minijev sessions sample`).

check() in dataset.py counts both layers in the output. Zero pattern hits only shows that scrub() ran. Zero literal
hits shows that no known private string passed.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

HOME = str(Path.home())
MIN_NAME, MIN_ENV_VALUE = 3, 8   # shorter literals would replace parts of normal words

SECRETS = [  # (name, pattern)
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("anthropic/openai key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("github token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("assigned secret", re.compile(r"(?i)(\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*\s*[=:]\s*['\"]?)"
                                   r"(?!<redacted>)[^\s'\"<]{6,}")),
    ("billing account", re.compile(r"\b[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}\b")),
    ("home path", re.compile(re.escape(HOME))),
    ("long hex", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
    ("random string", re.compile(r"\b(?=[A-Za-z0-9+_-]*\d)(?=[A-Za-z0-9+_-]*[A-Z])(?=[A-Za-z0-9+_-]*[a-z])"
                                 r"[A-Za-z0-9+_-]{32,}")),
]
LITERAL = "literal (private string)"


def _git(key: str) -> str:
    try:
        return subprocess.run(["git", "config", "--global", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


PLAIN_VALUE = re.compile(r"^(?:[A-Za-z]+|[\d.]+|https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?/?[\w/.-]*)$")


def env_values(folders: set[str]) -> set[str]:
    """Every value of MIN_ENV_VALUE characters or more in a .env file in these folders or one level below, except
    plain values (a word such as "development", a number, a local URL without credentials): as literals, these would
    replace normal text in every row."""
    out = set()
    for folder in folders:
        root = Path(folder)
        if not root.is_dir() or str(root) == HOME:
            continue   # the home folder itself: its .env files are not a project's
        for env in [*root.glob(".env*"), *root.glob("*/.env*")]:
            if not env.is_file():
                continue
            for line in env.read_text(errors="ignore").splitlines():
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                value = line.partition("=")[2].split("#")[0].strip().strip("'\"")
                if len(value) >= MIN_ENV_VALUE and not PLAIN_VALUE.match(value):
                    out.add(value)
    return out


def literals(folders: set[str], private_strings: Path | None) -> dict[str, str]:
    """The private strings for this machine (see the module docstring), each with its source: user, env or file."""
    name = _git("user.name")
    sources = {"user": {Path.home().name, _git("user.email"), name, *name.split()}, "env": env_values(folders)}
    sources["file"] = {line.strip() for line in private_strings.read_text().splitlines()
                       if line.strip() and not line.startswith("#")} if private_strings and private_strings.exists() else set()
    return {v: src for src, values in sources.items() for v in values if len(v) >= MIN_NAME}


def literal_pattern(values) -> re.Pattern | None:
    """One pattern for all literals, longest first, that matches only whole words: a username "dev" leaves "device"."""
    if not values:
        return None
    alts = "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])")


def scrub(text: str, literal: re.Pattern | None = None, counts: Counter | None = None) -> str:
    """The text with secrets, private strings and home paths replaced. counts, if given, counts each replaced literal."""
    text = re.sub(r"(?:/private)?/tmp/claude-\d+/[^\s'\"]*?/scratchpad", "<scratch>", text)
    text = text.replace(HOME, "~")
    if literal is not None:
        def replace(m):
            if counts is not None:
                counts[m.group(0)] += 1
            return "<private>"
        text = literal.sub(replace, text)
    for name, pattern in SECRETS:
        if name == "home path":
            continue
        if name == "assigned secret":
            text = pattern.sub(r"\1<redacted>", text)
        else:
            text = pattern.sub(f"<{name.split()[0]}>", text)
    return text


def hits(text: str, literal: re.Pattern | None = None) -> list[str]:
    out = [name for name, pattern in SECRETS if pattern.search(text)]
    return out + [LITERAL] if literal is not None and literal.search(text) else out
