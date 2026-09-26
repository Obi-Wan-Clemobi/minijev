## Context

The design, the measurements and the threats to validity are in `docs/SESSIONS_METHOD.md`. This file lists only the
decisions that the spec depends on.

## Goals / Non-Goals

**Goals:**
- Anyone can run the pipeline on their own logs, and nothing private reaches git.
- Every reported number can be traced to a frozen version and to the splits it read.

**Non-Goals:**
- Labels that are ground truth. External labels are the consensus of models under a written rubric.
- Support for log formats other than the Claude Code `.jsonl` format of 2026.

## Decisions

- **States leave out tool results and Claude's text.** Results hold most secrets; Claude's text can name the next
  call. Alternative: keep a short result summary. Rejected, because it would leak labels such as `will_fail`.
- **Two scrub layers.** Pattern hits after a scrub prove only that the scrub ran. A list of known private strings,
  built from git config, `.env` files and a user-kept file, gives an independent check.
- **Frozen versions are never rewritten.** A change gives a new version. `next_version` is the highest number plus
  one, so a deleted version's name is never reused.
- **Consensus rules are code.** `consensus.py` decides every label from the recorded opinions. A rerun reproduces the
  label file byte for byte.

## Risks / Trade-offs

- [The log format changes] → `extract` prints diagnostics; a jump in "unknown status" shows a change.
- [Residual private data] → a measured audit (1.0% of rows in v3) and a private-strings file; data stays outside git.
- [Labels from Claude about Claude] → two model families, blind rounds, audits; stated as threat T7.
