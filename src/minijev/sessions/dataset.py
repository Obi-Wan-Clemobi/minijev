"""From raw logs to frozen, split datasets. All files live under ROOT, outside any git work tree:

    ROOT/sessions-raw/                 the archive of raw logs (minijev sessions sync)
    ROOT/private-strings.txt           private strings that the scrub replaces, one per line (the user writes it)
    ROOT/sessions/events.jsonl         one parsed and scrubbed session per line (minijev sessions extract)
    ROOT/sessions/labels/<q>.jsonl     external labels: {"id", "y", "by"} per line
    ROOT/sessions/<version>/           frozen rows per question, and manifest.json (minijev sessions freeze)

ROOT is ~/.minijev-private, or the MINIJEV_PRIVATE environment variable.

Splits are by session start time inside each project: train until 70% of the project's calls, val until 85%, then test.
Rows from one session are always in one split. A global time split would put whole projects (and the tools they use)
in one split. A cut by session count gives tiny val and test splits, because a few long sessions hold most calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from . import logs, scrub
from .questions import QUESTIONS, Question

ROOT = Path(os.environ.get("MINIJEV_PRIVATE", Path.home() / ".minijev-private"))
SPLITS = (("train", 0.70), ("val", 0.85), ("test", 1.0))
LABEL_FIELDS = ("by", "sample", "unsure", "status")   # external label fields a row keeps. Not "why": it is free text that
                                            # the labeller wrote, and neither the scrub nor check covers it.
ACCESS: list[dict] = []   # every load_split call: {question, split, version, n}. Results and training logs record it.


class Paths:
    def __init__(self, root: Path = ROOT, allow_git: bool = False):
        self.root = Path(root).expanduser()
        self.raw = self.root / "sessions-raw"
        self.out = self.root / "sessions"
        self.events = self.out / "events.jsonl"
        self.labels = self.out / "labels"
        self.private_strings = self.root / "private-strings.txt"
        if not allow_git and inside_git(self.root):
            raise SystemExit(f"{self.root} is inside a git work tree. Session data must not go into git. "
                             "Choose another folder, or pass --allow-git.")


def inside_git(path: Path) -> bool:
    return any((p / ".git").exists() for p in [path.resolve(), *path.resolve().parents])


def literals(paths: Paths) -> dict[str, str]:
    folders = set().union(*(logs.folders(f) for f in logs.log_files(paths.raw))) if paths.raw.exists() else set()
    return scrub.literals(folders, paths.private_strings)


def literal_pattern(paths: Paths):
    return scrub.literal_pattern(literals(paths))


def extract(paths: Paths) -> logs.Diagnostics:
    """Parse and scrub every archived log into events.jsonl. Returns the counts of what could not be used."""
    files = logs.log_files(paths.raw)
    if not files:
        raise SystemExit(f"no logs in {paths.raw}. Run `minijev sessions sync` first.")
    sources = literals(paths)
    literal, replaced = scrub.literal_pattern(sources), Counter()
    diag = logs.Diagnostics(files=len(files))
    paths.out.mkdir(parents=True, exist_ok=True)
    with paths.events.open("w") as f:
        for path in files:
            s = logs.parse(path, diag)
            if not s["turns"]:
                continue
            s["cwd"], s["project"] = scrub.scrub(s["cwd"], literal, replaced), scrub.scrub(s["project"], literal, replaced)
            s["turns"] = [{**t, "text": scrub.scrub(t["text"], literal, replaced)} for t in s["turns"]]
            s["calls"] = [{**c, "summary": scrub.scrub(c["summary"], literal, replaced)} for c in s["calls"]]
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            diag["sessions written"] += 1
    for src in ("user", "env", "file"):   # a large count from one short literal means it replaces normal words
        diag[f"literals from {src}"] = sum(v == src for v in sources.values())
        diag[f"literal replacements from {src}"] = sum(n for v, n in replaced.items() if sources.get(v) == src)
    return diag


def sessions(paths: Paths) -> list[dict]:
    return [json.loads(line) for line in paths.events.open()]


def external_labels(paths: Paths, name: str) -> dict[str, dict]:
    path = paths.labels / f"{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def rows(paths: Paths, q: Question, all_sessions: list[dict] | None = None) -> list[dict]:
    """The rows of one question: {id, session, project, text, y} (and "by" for external labels)."""
    labels = external_labels(paths, q.name) if q.label_source == "external" else {}
    out = []
    for s in all_sessions if all_sessions is not None else sessions(paths):
        for rid, text, y in q.rows(s):
            row = {"id": rid, "session": s["session"], "project": s["project"], "text": text, "y": y}
            if q.label_source == "external":
                if rid not in labels or labels[rid].get("status") == "contested":
                    continue   # contested: no consensus, so the label is not trusted for training or evaluation
                row |= {"y": int(labels[rid]["y"])} | {k: labels[rid][k] for k in LABEL_FIELDS if k in labels[rid]}
            out.append(row)
    return out


def points(paths: Paths, q: Question, n: int, seed: int = 0) -> list[dict]:
    """n random unlabelled decision points of an external question, as {id, text}, for a labeller."""
    done = external_labels(paths, q.name)
    todo = [{"id": rid, "text": text} for s in sessions(paths) for rid, text, _ in q.rows(s) if rid not in done]
    return random.Random(seed).sample(todo, min(n, len(todo)))


def split_of(all_sessions: list[dict]) -> dict[str, str]:
    """Session id -> split: see the module docstring."""
    by_project = defaultdict(list)
    for s in all_sessions:
        by_project[s["project"]].append(s)
    out = {}
    for group in by_project.values():
        total, seen = sum(len(s["calls"]) for s in group), 0
        for s in sorted(group, key=lambda s: s["start"]):
            out[s["session"]] = next(name for name, cut in SPLITS if seen < cut * total or cut == 1.0)
            seen += len(s["calls"])
    return out


def check(paths: Paths, quiet: bool = False) -> bool:
    """Zero pattern hits and zero literal hits in every text of events.jsonl and every row. True if all counts are
    zero. (That a state never lists its own call or a result from the future is a property of questions.state, tested
    in tests/test_sessions.py: identical repeated calls make it impossible to check from the text.)"""
    literal = literal_pattern(paths)
    all_sessions = sessions(paths)
    counts = Counter()
    for s in all_sessions:
        for text in [s["cwd"], s["project"], *(t["text"] for t in s["turns"]), *(c["summary"] for c in s["calls"])]:
            counts.update(scrub.hits(text, literal))
    for q in QUESTIONS.values():
        for r in rows(paths, q, all_sessions):
            counts.update(scrub.hits(r["text"], literal))
    names = [n for n, _ in scrub.SECRETS] + [scrub.LITERAL]
    if not quiet:
        for name in names:
            print(f"{'FAIL' if counts[name] else 'ok  '} {name}: {counts[name]}")
    return not any(counts[n] for n in names)


def stats(paths: Paths) -> dict:
    all_sessions = sessions(paths)
    split = split_of(all_sessions)
    out = {}
    for q in QUESTIONS.values():
        per = defaultdict(Counter)
        for r in rows(paths, q, all_sessions):
            per[split[r["session"]]][q.labels[r["y"]]] += 1
        out[q.name] = {sp: dict(per[sp]) for sp, _ in SPLITS}
    return out


def next_version(paths: Paths) -> str:
    """One more than the highest existing version, so a deleted version's name is never reused."""
    numbers = [int(p.name[1:]) for p in paths.out.glob("v*") if p.name[1:].isdigit()]
    return f"v{max(numbers, default=0) + 1}"


def freeze(paths: Paths, version: str | None = None, diag: dict | None = None) -> Path:
    """Write the rows of every question with their split to ROOT/sessions/<version>/. An existing version is never
    rewritten. The check must pass first."""
    version = version or next_version(paths)
    folder = paths.out / version
    if folder.exists():
        raise SystemExit(f"{folder} exists. A frozen version is never rewritten; choose a new version.")
    if not check(paths):
        raise SystemExit("check failed: nothing written")
    all_sessions = sessions(paths)
    split = split_of(all_sessions)
    folder.mkdir(parents=True)
    manifest = {"version": version, "provenance": provenance(paths),
                "sessions": {sp: sorted(s for s, v in split.items() if v == sp) for sp, _ in SPLITS},
                "extract": diag or {}, "questions": {}}
    for q in QUESTIONS.values():
        path = folder / f"{q.name}.jsonl"
        with path.open("w") as f:
            for r in rows(paths, q, all_sessions):
                f.write(json.dumps({**r, "split": split[r["session"]]}, ensure_ascii=False) + "\n")
        manifest["questions"][q.name] = {**q.spec(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return folder


def provenance(paths: Paths) -> dict:
    """What a reader needs to reproduce or audit a version: the code revision, the state budget, the split cuts, the
    number of private strings per source (never their values), and the hashes of the external labels and rubrics."""
    from . import questions
    repo = Path(__file__).resolve().parents[3]

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    sources = Counter(literals(paths).values())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    code = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "src/minijev/sessions")),
            "code_sha256": code,   # of src/minijev/sessions/*.py: it identifies the code when git_dirty is true
            "state_budget": {"user_chars": questions.USER_CHARS, "calls_kept": questions.CALLS_KEPT,
                             "call_chars": logs.CALL_CHARS},
            "split_cuts": dict(SPLITS), "literals_per_source": dict(sources),
            # every file of the labelling process (label files, rounds, reviews, facts, rubrics), so a reader can rerun
            # `minijev sessions consensus` and compare. Queue files are left out: consensus rewrites them.
            "label_files": {str(p.relative_to(paths.labels)): sha(p) for p in sorted(paths.labels.rglob("*"))
                            if p.is_file() and not p.name.startswith("queue-") and "todo" not in p.parts}
                           if paths.labels.exists() else {}}


def load_split(question: str, split: str, version: str, paths: Paths | None = None) -> list[dict]:
    """The rows of one frozen split, each with "text" and "y". Fails loudly if the file changed after the freeze.
    Every call is logged in ACCESS, so a result can show which splits it read."""
    folder = (paths or Paths()).out / version
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{question}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["questions"][question]["sha256"]:
        raise ValueError(f"{path} changed after the freeze")
    out = [r for r in map(json.loads, path.open()) if r["split"] == split]
    ACCESS.append({"question": question, "split": split, "version": version, "n": len(out)})
    return out


def manifest(version: str, paths: Paths | None = None) -> dict:
    return json.loads(((paths or Paths()).out / version / "manifest.json").read_text())
