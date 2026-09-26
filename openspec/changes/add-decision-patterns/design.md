## Context

A **decision pattern** is a mapping from a request and the work so far to the next kind of step. It must hold across
projects. The **domain** is what a project is about (board games, travel). A model that learned the domain does not
transfer to a new project. Background, data and threats: `docs/SESSIONS_METHOD.md`.

Measured on the archive (2026-09-26): 198 of 232 sessions come from SDK entry points (the travel-planner app runs
Claude Code as a program); 34 sessions are interactive and hold 3670 of the 4156 calls. Decision-pattern versions use
interactive sessions only, because the goal is how the user works. Every assistant message records its token use
(8850 of 8850 entries);
27 of 32 `AskUserQuestion` calls have a recorded answer; 248 of 3627 assistant messages hold 2 or more tool calls.

## Goals / Non-Goals

**Goals:**
- Measure whether minijev learns patterns that transfer to a project it has not seen.
- Find where tokens go, so that the decisions worth handing to minijev are known.

**Non-Goals:**
- `needs_approval` (travel-planner dominates it, and threat T14 applies).
- "Could these calls run in parallel?" The logs show what Claude did, not what was possible.
- Training on preferences: 27 answered questions are evaluation data only.

## Decisions

### Evaluation rules (fixed before any result)

1. **Leave one project out.** Folds: minijev; travel-planner with travel-planner/frontend; board-game-event-planner;
   home. Projects with fewer than 50 calls are always in train. Val is the newest 15% of calls of
   each training project, never the held-out project. The manifest records the strategy.
2. **Report per fold, never pooled.** For each fold, first list which labels its train split supports. A class with
   fewer than 20 train rows is reported apart. Example: almost all "mcp" calls are travel-planner calls.
3. **Baselines per fold:** class prior, previous call, zero-shot readout, zero-shot plus a val-fitted bias and
   temperature. A trained model counts as better only if the paired bootstrap 95% interval of its test log-loss delta
   against the previous-call baseline is below zero.
4. **The ladder must beat the flat question.** On each held-out project: the ladder (`work_kind` then
   `tool_given_kind`), flat `next_tool`, and the previous-call baseline.
5. **Hand-off** is evaluated as a risk-coverage curve (accuracy on kept decisions against the share handed off). The
   threshold is chosen on val.
6. **Tokens saved** always states its counterfactual (what the large model would no longer do). Without one, it is an
   upper bound (Inferred).

### Amendments from the val results (2026-09-26, before any test result)

The count baselines on val of v9 (interactive sessions, leave one project out) narrow what this change claims:

- **One final report per fold reads test.** In this split the held-out project is test, so every per-project number
  reads test. `minijev sessions compare --fold` computes all fixed methods in one pass (prior, previous call, base
  and adapter each with a val-fitted bias and temperature, the val-chosen hand-off threshold applied to test, and the
  ladder against the flat question), and refuses to run twice for a version, fold and adapter.
- **Ladder against flat, on one label space.** The ladder's probability of a tool group is the sum over kinds of
  P(kind) × P(tool | kind), over the tools of that group; a kind with one tool gives it probability 1. P(tool | kind)
  is the previous-call count model of `tool_<kind>`. The groups are those of `next_tool`, and the rows are the same
  calls. The ladder is scored with P(kind) from the previous-call model and from the adapter.
- **`work_kind` is the target.** It is the one question with a signal across projects: the previous call beats the
  prior by 10 to 17 accuracy points in 3 of 4 folds (Measured, val).
- **`area` does not transfer.** The prior scores 0% to 5% on the val of each fold, and most areas have fewer than 20
  train rows (Measured). Nothing uses it as a feature. This also shows that the folds separate domains.
- **`waste`** has 7 to 12 val positives per fold: counts only, no claim. **`tool_inspect`** is Bash in 90% to 98% of
  rows: reported as trivial.
- **Pilot fold rule:** among folds with at least 500 test rows, the fold with the fewest `work_kind` classes under 20
  train rows. Result: board-game-event-planner (989 test rows; only "publish" is unsupported). Only row counts of
  the test split were read to apply the rule, not its labels.
- **Zero-shot readouts on the pilot fold only.** The base model had no usable signal on these questions before
  training (v4: zero-shot below the prior on every question, Measured), so readouts on all four folds (about 5 hours
  of CPU) would not change a decision. The pilot's final report includes the base model, calibrated on val.

### Labels

- **`work_kind` from a fixed table, not from labellers.** The table maps the next call (tool, Bash kind, git
  subcommand, MCP verb) to inspect, change, run, remote, research, browse, publish, ask or orchestrate. The kinds come
  from the inventory of tool names in our logs; each kind matches at least 30 calls (Measured). It is committed before
  any fold result. Alternative: labellers who see the next call. Rejected: their label would be a noisy copy of
  the same table.
- **`area` from labellers, from the request only.** It is the one label the logs cannot give. The consensus pipeline
  of `session-data` applies.
- **`waste` from the logs.** A failed call followed by a call of the same tool in the same turn; a Read of a file
  already read in the turn with no edit between; a rejected call.

- **One tool question per kind.** `tool_inspect`, `tool_change`, `tool_research`, `tool_publish`,
  `tool_orchestrate`, each a Choice with fixed options. The flow engine's `options_from` (enhance-flow-engine) chains
  them after `work_kind`.

### Token breakdown

Per kind of work and per chain pattern (the sequence of tool groups in a turn): fresh input, cache read, cache write
and output tokens, summed over the assistant messages of the turn. Cache reads are reported apart, because they cost
much less.

## Risks / Trade-offs

- [4 folds; each result rests on one project] → report every fold; a result that holds on 1 of 4 folds is weak.
- [`home` mixes several kinds of work] → it is still a fold, and its report says so.
- [The kind-of-work table is a coarse version of the tool] → the flat question and the previous-call baseline control
  for a ladder score that comes only from the table.
- [Removing the project line changes every state] → v8 is a new version; v6 results are not compared with v8 results.
