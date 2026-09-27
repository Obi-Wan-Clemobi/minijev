"""From raw logs to frozen, split datasets. All files live under ROOT, outside any git work tree:

    ROOT/sessions-raw/                 the archive of raw logs (minijev sessions sync)
    ROOT/private-strings.txt           private strings that the scrub replaces, one per line (the user writes it)
    ROOT/sessions/events.jsonl         one parsed and scrubbed session per line (minijev sessions extract)
    ROOT/sessions/labels/<q>.jsonl     external labels: {"id", "y", "by"} per line
    ROOT/sessions/<version>/           frozen rows per question, and manifest.json (minijev sessions freeze)

ROOT is ~/.minijev-private, or the MINIJEV_PRIVATE environment variable.

Split strategies (rows from one session are always in one split):
- time-per-project: sessions by start time inside each project; train until 70% of the project's calls, val until
  85%, then test. A global time split would put whole projects in one split; a cut by session count gives tiny val and
  test splits, because a few long sessions hold most calls.
- leave-one-project-out: one fold per project group with at least MIN_FOLD_CALLS calls. In a fold, the held-out group
  is test; in every other group the sessions that reach into the newest 15% of calls are val (never the group's first
  session) and the rest is train; groups with fewer calls are always train. A project group is the first part of the project name ("app/web" belongs to "app").
- before-cutoff: every session that starts before the cutoff, in one split, "dev" (development data).
- after-cutoff: every session that starts from the cutoff to the end date, in one split, "test" (a prospective test).
  It cannot be frozen before the end date. Every other reader skips the sessions from SEALED on.
A session starts at its first user turn (logs.parse). A session without a user turn has no start and no rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import logs, scrub
from .questions import QUESTIONS, Question

ROOT = Path(os.environ.get("MINIJEV_PRIVATE", Path.home() / ".minijev-private"))
SPLITS = (("train", 0.70), ("val", 0.85), ("test", 1.0))
STRATEGIES = ("time-per-project", "leave-one-project-out", "before-cutoff", "after-cutoff")
WINDOWED = {"before-cutoff": "dev", "after-cutoff": "test"}   # one split each, chosen by session start time
# The prospective test of openspec/changes/add-plan-cost. Every reader skips the sessions that start from SEALED on.
# Only `check` and an after-cutoff freeze on or after SEALED_UNTIL can read them. SEALED_UNTIL is 2026-11-22 00:00
# America/Toronto (EST, UTC-5).
SEALED, SEALED_UNTIL = "2026-09-27T04:00:00Z", "2026-11-22T05:00:00Z"
MIN_FOLD_CALLS, LOPO_VAL = 50, 0.15
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
            s["turns"] = [{**t, "text": scrub.scrub(t["text"], literal, replaced),
                           **({"last_text": scrub.scrub(t["last_text"], literal, replaced)} if "last_text" in t else {})}
                          for t in s["turns"]]
            s["calls"] = [{**c, "summary": scrub.scrub(c["summary"], literal, replaced)} for c in s["calls"]]
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            diag["sessions written"] += 1
    for src in ("user", "env", "file"):   # a large count from one short literal means it replaces normal words
        diag[f"literals from {src}"] = sum(v == src for v in sources.values())
        diag[f"literal replacements from {src}"] = sum(n for v, n in replaced.items() if sources.get(v) == src)
    return diag


def sealed(s: dict) -> bool:
    """True for a session of the prospective test window or later. A session without a start counts as sealed."""
    try:
        return not s.get("start") or when(s["start"]) >= when(SEALED)
    except ValueError:   # a start without a time zone: fail closed
        return True


def sessions(paths: Paths, include_sealed: bool = False) -> list[dict]:
    """The extract. Sessions from SEALED on are left out unless include_sealed (only `check` and the after-cutoff
    freeze pass it)."""
    out = [json.loads(line) for line in paths.events.open()]
    return out if include_sealed else [s for s in out if not sealed(s)]


def external_labels(paths: Paths, name: str) -> dict[str, dict]:
    path = paths.labels / f"{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def rows(paths: Paths, q: Question, all_sessions: list[dict] | None = None, show_project: bool = True) -> list[dict]:
    """The rows of one question: {id, session, project, text, y} (and "by" for external labels). With show_project
    false, the states leave out the project line; the row still names its project, for splits and reports."""
    labels = external_labels(paths, q.name) if q.label_source == "external" else {}
    out = []
    for s in all_sessions if all_sessions is not None else sessions(paths):
        for rid, text, y in q.rows(s if show_project else {**s, "project": None}):
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


def group_of(project: str) -> str:
    return project.split("/")[0]


def folds(all_sessions: list[dict]) -> list[str]:
    """The project groups that are a fold of leave-one-project-out: those with at least MIN_FOLD_CALLS calls."""
    calls = Counter()
    for s in all_sessions:
        calls[group_of(s["project"])] += len(s["calls"])
    return sorted(g for g, n in calls.items() if n >= MIN_FOLD_CALLS)


def when(ts: str) -> datetime:
    """An ISO time ("2026-09-27T04:00:00Z" or with an offset) as an aware datetime."""
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"{ts} has no time zone")
    return t


def split_of(all_sessions: list[dict], strategy: str = "time-per-project", held_out: str | None = None,
             cutoff: str | None = None, until: str | None = None) -> dict[str, str]:
    """Session id -> split: see the module docstring. A windowed strategy leaves out the sessions outside its window."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy}")
    if strategy in WINDOWED:
        if cutoff is None or (strategy == "after-cutoff" and until is None):
            raise ValueError(f"{strategy} needs --cutoff" + (" and --until" if strategy == "after-cutoff" else ""))
        lo, hi = when(cutoff), when(until) if until else None
        inside = (lambda t: t < lo) if strategy == "before-cutoff" else (lambda t: lo <= t < hi)
        return {s["session"]: WINDOWED[strategy] for s in all_sessions if inside(when(s["start"]))}
    lopo = strategy == "leave-one-project-out"
    if lopo and held_out not in folds(all_sessions):
        raise ValueError(f"{held_out} is not a fold; folds: {folds(all_sessions)}")
    by_group = defaultdict(list)
    for s in all_sessions:
        by_group[group_of(s["project"]) if lopo else s["project"]].append(s)
    out = {}
    for group, members in by_group.items():
        total, seen = sum(len(s["calls"]) for s in members), 0
        for s in sorted(members, key=lambda s: s["start"]):
            if not lopo:
                out[s["session"]] = next(name for name, cut in SPLITS if seen < cut * total or cut == 1.0)
            elif group == held_out:
                out[s["session"]] = "test"
            else:
                # val: the sessions whose calls reach past the newest LOPO_VAL of the group's calls, never the first
                # session (a cut by start time leaves val empty when the last session is long)
                late = seen > 0 and seen + len(s["calls"]) > (1 - LOPO_VAL) * total
                out[s["session"]] = "val" if late and total >= MIN_FOLD_CALLS else "train"
            seen += len(s["calls"])
    return out


def check(paths: Paths, quiet: bool = False) -> bool:
    """Zero pattern hits and zero literal hits in every text of events.jsonl and every row. True if all counts are
    zero. (That a state never lists its own call or a result from the future is a property of questions.state, tested
    in tests/test_sessions.py: identical repeated calls make it impossible to check from the text.)"""
    literal = literal_pattern(paths)
    all_sessions = sessions(paths, include_sealed=True)   # the privacy scan covers every session
    counts = Counter()
    for s in all_sessions:
        for text in [s["cwd"], s["project"], *(t["text"] for t in s["turns"]), *(t.get("last_text", "") for t in s["turns"]),
                     *(c["summary"] for c in s["calls"])]:
            counts.update(scrub.hits(text, literal))
    for q in QUESTIONS.values():
        for r in rows(paths, q, all_sessions):
            counts.update(scrub.hits(r["text"], literal))
    names = [n for n, _ in scrub.SECRETS] + [scrub.LITERAL]
    if not quiet:
        for name in names:
            print(f"{'FAIL' if counts[name] else 'ok  '} {name}: {counts[name]}")
    return not any(counts[n] for n in names)


def stats(paths: Paths, strategy: str = "time-per-project", held_out: str | None = None,
          interactive_only: bool = False, cutoff: str | None = None) -> dict:
    if strategy == "after-cutoff":
        raise SystemExit("stats does not count the labels of a prospective test before its freeze")
    all_sessions = select(sessions(paths), interactive_only)
    split = split_of(all_sessions, strategy, held_out, cutoff)
    all_sessions = [s for s in all_sessions if s["session"] in split]
    names = [WINDOWED[strategy]] if strategy in WINDOWED else [sp for sp, _ in SPLITS]
    out = {}
    for q in QUESTIONS.values():
        per = defaultdict(Counter)
        for r in rows(paths, q, all_sessions):
            per[split[r["session"]]][q.labels[r["y"]]] += 1
        out[q.name] = {sp: dict(per[sp]) for sp in names}
    return out


def next_version(paths: Paths) -> str:
    """One more than the highest version that exists or ever existed (the ledger versions.txt), so the name of a
    deleted version is never reused."""
    ledger = paths.out / "versions.txt"
    names = [p.name for p in paths.out.glob("v*")] + (ledger.read_text().split() if ledger.exists() else [])
    numbers = [int(n[1:]) for n in names if n[1:].isdigit()]
    return f"v{max(numbers, default=0) + 1}"


def select(all_sessions: list[dict], interactive_only: bool) -> list[dict]:
    """With interactive_only, the sessions a person typed (logs.INTERACTIVE entry points); SDK sessions, where a
    program writes the turns, are left out. A session without the flag (an extract older than the flag) counts as not
    interactive, as logs.parse counts an unknown entry point."""
    return [s for s in all_sessions if s.get("interactive", False)] if interactive_only else all_sessions


def freeze(paths: Paths, version: str | None = None, diag: dict | None = None,
           strategy: str = "time-per-project", show_project: bool = True, interactive_only: bool = False,
           cutoff: str | None = None, until: str | None = None, now: datetime | None = None) -> Path:
    """Write the rows of every question with their split to ROOT/sessions/<version>/. An existing version is never
    rewritten. The check must pass first. With leave-one-project-out, each row gets "splits": {fold: split} for every
    fold, instead of "split"."""
    if strategy in WINDOWED and (cutoff is None or (strategy == "after-cutoff" and until is None)):
        raise SystemExit(f"{strategy} needs --cutoff" + (" and --until" if strategy == "after-cutoff" else ""))
    if strategy == "after-cutoff" and (now or datetime.now(timezone.utc)) < when(until):
        raise SystemExit(f"the prospective test ends at {until}: it cannot be frozen before then")
    version = version or next_version(paths)
    folder = paths.out / version
    if folder.exists():
        raise SystemExit(f"{folder} exists. A frozen version is never rewritten; choose a new version.")
    if not check(paths):
        raise SystemExit("check failed: nothing written")
    # only the fixed end date opens the seal, never the caller's --until
    opened = strategy == "after-cutoff" and (now or datetime.now(timezone.utc)) >= when(SEALED_UNTIL)
    all_sessions = select(sessions(paths, include_sealed=opened), interactive_only)
    lopo = strategy == "leave-one-project-out"
    fold_names = folds(all_sessions) if lopo else [None]
    splits = {f: split_of(all_sessions, strategy, f, cutoff, until) for f in fold_names}
    if strategy in WINDOWED:
        all_sessions = [s for s in all_sessions if s["session"] in splits[None]]
        if not all_sessions:
            raise SystemExit(f"no readable session starts in the {strategy} window: nothing written")
    names = [WINDOWED[strategy]] if strategy in WINDOWED else [sp for sp, _ in SPLITS]
    folder.mkdir(parents=True)
    manifest = {"version": version, "provenance": provenance(paths),
                "split": {"strategy": strategy, "folds": fold_names if lopo else [], "show_project": show_project,
                          "interactive_only": interactive_only, "cutoff": cutoff, "until": until},
                "sessions": {str(f): {sp: sorted(s for s, v in splits[f].items() if v == sp) for sp in names}
                             for f in fold_names} if lopo else
                            {sp: sorted(s for s, v in splits[None].items() if v == sp) for sp in names},
                "extract": diag or {}, "questions": {}}
    for q in QUESTIONS.values():
        path = folder / f"{q.name}.jsonl"
        with path.open("w") as f:
            for r in rows(paths, q, all_sessions, show_project):
                where = {"splits": {fo: splits[fo][r["session"]] for fo in fold_names}} if lopo else \
                    {"split": splits[None][r["session"]]}
                f.write(json.dumps({**r, **where}, ensure_ascii=False) + "\n")
        manifest["questions"][q.name] = {**q.spec(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with (paths.out / "versions.txt").open("a") as f:
        f.write(version + "\n")
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


def load_split(question: str, split: str, version: str, paths: Paths | None = None, fold: str | None = None) -> list[dict]:
    """The rows of one frozen split, each with "text" and "y". Fails loudly if the file changed after the freeze.
    A leave-one-project-out version needs the fold. Every call is logged in ACCESS, so a result can show which splits
    it read."""
    folder = (paths or Paths()).out / version
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{question}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["questions"][question]["sha256"]:
        raise ValueError(f"{path} changed after the freeze")
    lopo = manifest.get("split", {}).get("strategy") == "leave-one-project-out"
    if lopo and fold not in manifest["split"]["folds"]:
        raise ValueError(f"{version} is leave-one-project-out: give a fold, one of {manifest['split']['folds']}")
    if not lopo and split not in manifest["sessions"]:
        raise ValueError(f"{version} has no split {split}; it has {sorted(manifest['sessions'])}")
    out = [r for r in map(json.loads, path.open()) if (r["splits"][fold] if lopo else r["split"]) == split]
    ACCESS.append({"question": question, "split": split, "version": version, "n": len(out)} | ({"fold": fold} if lopo else {}))
    return out


def manifest(version: str, paths: Paths | None = None) -> dict:
    return json.loads(((paths or Paths()).out / version / "manifest.json").read_text())
