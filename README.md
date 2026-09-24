# minijev

minijev is a small, local, open re-creation of the *idea* behind TypeSafe's Jev: text in, typed
and calibrated probabilistic decisions out. minijev gets these decisions from one forward pass of a
small transformer. We built minijev to understand the mechanism. It is not a copy of Jev.

- **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** explains what we built and measured. Start here.
- **[docs/DESIGN.md](docs/DESIGN.md)** is the design. §12 gives the implementation phases.
- **[docs/RESEARCH.md](docs/RESEARCH.md)** gives what is known about Jev, with sources and evidence levels.
- **[poc/](poc/)** is a working proof of concept: `cd poc && uv sync && uv run python experiments.py demo`
  - Calibration temperatures and other dials: `poc/minijev.env` (DESIGN.md §6).
  - Tests: `cd poc && uv run pytest`.
- **[web/](web/)** is a local web playground for the POC. See "Run the playground" below.
- **[CLAUDE.md](CLAUDE.md)** gives the project writing rules and the glossary. The general rules (modified ASD-STE100) are in `~/.claude/CLAUDE.md`.

Status: design, research, and a working proof of concept for Phases 0, 1, and 3, and part of Phase 2 (DESIGN.md §12).
The code is not packaged yet.

## Run the playground

1. Run `./setup.sh` once. It checks uv and Node 20+, then installs the Python and web dependencies.
   Add `--model` to download Qwen2.5-0.5B-Instruct (~1 GB) now, not on the first request.
2. Run `tilt up`. It starts the API (port 8000) and the web app (port 3000), and shows both logs in one dashboard.
3. Open http://localhost:3000.

Without Tilt, run `cd poc && uv run uvicorn server:app --port 8000` and `cd web && npm run dev` in two terminals.
`docker compose up --build` runs both parts in Docker. Change the calibration dials and the model in
`poc/minijev.env`.

