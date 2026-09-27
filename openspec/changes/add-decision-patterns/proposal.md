## Why

The goal is to teach minijev how the user works, not what the user works on: decision patterns (what kind of work a
request needs, which tool fits, how steps chain) that hold across projects. minijev can then answer routine decisions
cheaply and hand a decision to the large model only when it is unsure, so that fewer large-model tokens are spent.
The current datasets use a time split inside each project and put the project name in the state, so they cannot show
whether a model learned patterns or domains.

## What Changes

- A split strategy parameter, with a new strategy: leave one project out. Frozen as v13.
- States without the `Project:` line.
- A measured token breakdown of the logs, per kind of work and per chain pattern.
- New log-label questions: `work_kind` (from a fixed table on the next call), `tool_<kind>` (the tool, one question
  per kind, with the tools that can do it), and `waste` (a failed call followed by a retry, a repeated read, a
  rejected call).
- A new external-label question: `area` (the technical area of a turn), labelled by the consensus pipeline from the
  request only. It is a feature, not a target.
- The evaluation rules of design.md, fixed before any result.

## Capabilities

### New Capabilities
- `decision-patterns`: questions, datasets and evaluation for domain-independent decision patterns and token use.

### Modified Capabilities
- `session-data`: the split becomes a parameter with a leave-one-project-out strategy, recorded in the manifest.

## Impact

- `src/minijev/sessions/dataset.py` (`split_of`), `questions.py` (state and new questions), `evaluate.py` (per-fold
  reports), a new `tokens.py` for the breakdown.
- New frozen versions for the leave-one-project-out split (v13 now). v12 serves the `needs_approval` work.
- Compute: count baselines are cheap; zero-shot readouts take about 1 hour per fold; an adapter takes about 12 hours
  per fold, so only one pilot fold is trained first.
