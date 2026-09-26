"""`minijev sessions …`: the command line of minijev.sessions (see its docstring)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import __doc__ as USAGE
from . import dataset, logs
from .questions import QUESTIONS, load


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="minijev sessions", description=USAGE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=dataset.ROOT, help="the private folder (default: %(default)s)")
    ap.add_argument("--allow-git", action="store_true", help="allow a root inside a git work tree")
    ap.add_argument("--questions", action="append", default=[], help="a Python file that registers more questions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--source", type=Path, default=logs.CLAUDE_PROJECTS)
    sub.add_parser("extract")
    s = sub.add_parser("sample")
    s.add_argument("question")
    s.add_argument("-n", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("check")
    s = sub.add_parser("stats")
    s.add_argument("--split", choices=dataset.STRATEGIES, default="time-per-project")
    s.add_argument("--held-out", help="the held-out fold, for --split leave-one-project-out")
    s = sub.add_parser("freeze")
    s.add_argument("--version", help="default: the next unused vN")
    s.add_argument("--split", choices=dataset.STRATEGIES, default="time-per-project")
    s.add_argument("--no-project", action="store_true", help="leave the project line out of every state")
    s = sub.add_parser("points")
    s.add_argument("question")
    s.add_argument("-n", type=int, default=100)
    s = sub.add_parser("consensus", help="combine labellers and reviewers (minijev.sessions.consensus)")
    s.add_argument("question")
    sub.add_parser("tokens", help="token use per kind of work and per chain pattern (tokens.py)")
    s = sub.add_parser("compare", help="the fixed adapter comparison on test (read once): SESSIONS_METHOD.md §7.1")
    s.add_argument("version")
    s.add_argument("question")
    s.add_argument("adapter", help="an adapter folder, for example ~/.minijev-private/adapters/<run>/epoch-1")
    s = sub.add_parser("baselines")
    s.add_argument("version")
    s.add_argument("questions_to_run", nargs="*", metavar="question", help="default: every question in the version")
    s.add_argument("--model", help="default: MINIJEV_MODEL")
    s.add_argument("--no-readout", action="store_true", help="count baselines only; do not load a model")
    args = ap.parse_args(argv)

    for path in args.questions:
        load(path)
    paths = dataset.Paths(args.root, args.allow_git)
    if args.cmd == "sync":
        print(f"copied {logs.sync(args.source, paths.raw)} files to {paths.raw}")
    elif args.cmd == "extract":
        diag = dataset.extract(paths)
        for k, v in sorted(diag.items()):
            print(f"  {k}: {v}")
        print(f"wrote {paths.events}")
        (paths.out / "extract.json").write_text(json.dumps(diag, indent=1))
    elif args.cmd == "sample":
        rows = dataset.rows(paths, QUESTIONS[args.question])
        for r in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
            label = QUESTIONS[args.question].labels[r["y"]] if r["y"] is not None else "?"
            print(f"=== {r['id']}  label: {label}\n{r['text']}\n")
    elif args.cmd == "check":
        sys.exit(0 if dataset.check(paths) else 1)
    elif args.cmd == "stats":
        for name, per in dataset.stats(paths, args.split, args.held_out).items():
            print(f"\n{name} ({QUESTIONS[name].type}, labels from {QUESTIONS[name].label_source})")
            for split, counts in per.items():
                print(f"  {split:5} n={sum(counts.values()):5}  " + "  ".join(f"{k}={counts.get(k, 0)}"
                                                                          for k in QUESTIONS[name].labels))
    elif args.cmd == "freeze":
        extract = paths.out / "extract.json"
        diag = json.loads(extract.read_text()) if extract.exists() else None
        print(f"wrote {dataset.freeze(paths, args.version, diag, args.split, not args.no_project)}")
    elif args.cmd == "points":
        for p in dataset.points(paths, QUESTIONS[args.question], args.n):
            print(json.dumps(p, ensure_ascii=False))
    elif args.cmd == "consensus":
        from . import consensus
        q = QUESTIONS[args.question]
        folder = paths.labels / q.name
        samples = folder / "samples.json"
        texts = {rid: text for s_ in dataset.sessions(paths) for rid, text, _ in q.rows(s_)}
        result = consensus.decide(folder, json.loads(samples.read_text()) if samples.exists() else {}, texts)
        done = not any(result["queues"].values())
        consensus.write(folder, result, texts, paths.labels / f"{q.name}.jsonl" if done else folder / "partial.jsonl")
        print(json.dumps({**result["counts"], "kappa_round1": consensus.pairwise_kappa(folder)}, indent=1))
        print("consensus complete: wrote" if done else "not complete: queues in", folder)
    elif args.cmd == "tokens":
        from .tokens import report
        report(paths)
    elif args.cmd == "compare":
        from .evaluate import compare
        compare(args.version, args.question, str(Path(args.adapter).expanduser()), paths)
    elif args.cmd == "baselines":
        from .evaluate import baselines
        names = args.questions_to_run or list(dataset.manifest(args.version, paths)["questions"])
        baselines(args.version, names, args.model, paths, readouts=not args.no_readout)
