## Why

minijev needs training data about how its users work. Claude Code session logs hold that data, but they also hold
private data, and their format is not documented. This change records, as a spec, the pipeline that we built and
measured on 2026-09-25, so that later changes modify a written baseline instead of undocumented code.

## What Changes

- A `minijev sessions` command and the `minijev.sessions` package: sync, extract, sample, check, stats, freeze,
  points, consensus, baselines, compare.
- Parsing of the main-thread session logs into turns and calls, with diagnostics for what the parser could not use.
- Two independent scrub layers (patterns and private strings), and a check that must find zero hits before a freeze.
- States that hold only what existed before a decision.
- Questions with log labels (`next_tool`, `bash_kind`, `will_fail`) and with external labels (`needs_approval`), and
  a registry for user questions.
- A consensus process for external labels from several model labellers, reviewers and adjudicators.
- Frozen versions with hashes and provenance, and a log of every split read.
- Baselines and a fixed adapter comparison.

## Capabilities

### New Capabilities
- `session-data`: turn Claude Code session logs into private, frozen, labelled datasets for minijev, and evaluate
  models on them.

### Modified Capabilities

## Impact

- New package `src/minijev/sessions/`, new subcommand in `src/minijev/cli.py`.
- `src/minijev/calibrate.py` now holds `bootstrap_ci`, `multiclass_metrics` and `fit_bias_temperature`;
  `poc/experiments.py` imports them from there.
- `poc/train_lora.py` gains `--task sessions-<question>`.
- Docs: `docs/SESSIONS.md` (how-to) and `docs/SESSIONS_METHOD.md` (method and data card).
- All data stays under `~/.minijev-private`, outside git.
