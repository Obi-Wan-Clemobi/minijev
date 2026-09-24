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
- **[CLAUDE.md](CLAUDE.md)** gives the project writing rules and the glossary. The general rules (modified ASD-STE100) are in `~/.claude/CLAUDE.md`.

Status: design, research, and a working proof of concept for Phases 0, 1, and 3, and part of Phase 2 (DESIGN.md §12).
The code is not packaged yet.
