"""The minijev command line.

    minijev ask "Help! My payouts have failed for 3 days." "Is this urgent?"
    minijev ask "…" "Which team?" --type choice --options billing technical sales
    minijev ask "…" "How upset?" --type score --levels calm annoyed angry
    minijev ask --json request.json                  # a full {state, questions} request
    minijev serve --port 8000                         # the HTTP API for the web app

Settings come from ./minijev.env and MINIJEV_* variables (run from poc/ to use its calibration).
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="minijev", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer one question (or a JSON request) with one forward pass")
    a.add_argument("state", nargs="?", help="the text the question is about")
    a.add_argument("question", nargs="?", help="the question")
    a.add_argument("--type", choices=["noul", "choice", "score"], default="noul")
    a.add_argument("--options", nargs="+", help="Choice options")
    a.add_argument("--levels", nargs="+", help="Score levels, lowest first")
    a.add_argument("--json", dest="request", help="a request file: {\"state\": …, \"questions\": {…}}")
    a.add_argument("--model", help="override MINIJEV_MODEL")
    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("minijev.api.server:app", host=args.host, port=args.port)
        return

    if args.request:
        with open(args.request) as f:
            req = json.load(f)
    else:
        if not (args.state and args.question):
            ap.error("ask needs STATE and QUESTION, or --json FILE")
        q: dict = {"type": args.type, "instructions": args.question}
        if args.type == "choice":
            if not args.options:
                ap.error("--type choice needs --options")
            q["criteria"] = {o: None for o in args.options}
        if args.type == "score":
            if not args.levels:
                ap.error("--type score needs --levels")
            q["criteria"] = args.levels
        req = {"state": args.state, "questions": {"q": q}}

    from .engine import Engine
    from .judge import ask
    engine = Engine(args.model) if args.model else Engine()
    json.dump(ask(engine, req), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
