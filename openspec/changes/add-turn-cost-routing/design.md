## Context

Measured on the 34 interactive sessions (256 turns, one run on the extract of task 1.1, 2026-09-26): assistant
messages per turn have a median of 6, a p75 of 15 and a p90 of 36 (linear interpolation; maximum 231); one turn has 0 messages. Messages and cache reads per turn have a
Spearman correlation of 0.90.

| Messages in the turn | Turns | Share of cache reads |
|---|---|---|
| 0–2 | 72 | 1.9% |
| 3–8 | 84 | 7.4% |
| 9–30 | 67 | 29.1% |
| 31 or more | 33 | 61.7% |

A **long turn** has 31 or more assistant messages. A second cutoff, 21 or more messages, gives 53 turns and 74.4% of
cache reads. P(long | previous turn long) is 6/31
(19%), against a base rate of 33/256 (12.9%). 14 of the 33 long turns start with a user message of 20 characters or
fewer (for example "go"); 80 of all 256 turns do.

**The signal is weak.** No simple feature raises P(long) far above the base rate of 12.9% (Measured on all 256 turns):
previous turn long 19% (6/31); user message of 20 characters or fewer 17.5% (14/80); user message length terciles 18%,
8%, 13%; context-size terciles 11%, 15%, 13%. The best lift is 1.5 times the base rate. Routing on these features would
flag many turns that are not long. This change measures whether any predictor does better, and the pilot runs only if
one does (the go or no-go rule below).

The time-per-project split (v12) holds train 175 turns (27 long), val 25 (3 long) and test 56 (3 long). The leave-one-
project-out folds (v13) hold 3 to 7 long turns in val and 1 to 13 in test (Measured, label counts only).

| Split or held-out project | Train (long) | Val (long) | Test (long) |
|---|---|---|---|
| Time per project (v12) | 175 (27) | 25 (3) | 56 (3) |
| board-game-event-planner (v13) | 127 (19) | 40 (7) | 89 (7) |
| home (v13) | 158 (25) | 64 (7) | 34 (1) |
| minijev (v13) | 143 (18) | 65 (3) | 48 (12) |
| travel-planner (v13) | 114 (16) | 62 (4) | 80 (13) |

## Goals / Non-Goals

**Goals:**
- Predict, from the user message and what came before it in the session, whether a turn will be long.
- Find out whether routing works in daily use: latency, timeouts, how often it flags, how often a flag is false, and
  what the hint costs.

**Non-Goals:**
- Measuring savings in this change (see "Why the pilot cannot measure savings").
- Choosing the model or effort. The hook adds a hint; the user and Claude decide.

## Decisions

### Turns

- A turn is one user message and the assistant messages until the next user message. A rejected tool call with a
  reason stays inside its turn (it is a tool result, `logs.parse`).
- `/clear` starts a new session file: in all 6 cases its command entry comes before the first turn of its file
  (Measured), so the turn after it has no previous user message and no previous level, as any session start. A compact
  summary entry is not a user message and is not a turn (`logs.user_text`).
- Interrupted turns (2) are marked (`interrupted`); results are reported with and without them.
- Messages typed while Claude works are queue operations in the log: of 272 queued messages, 9 later appear as a user
  turn, none right after a tool call (Measured). They do not split turns, so no merge is needed.
- Each turn keeps the end of its last assistant text (at most 1000 characters); the extract scrubs it and `check()`
  scans it. It holds more private strings than user text (private-string replacements rose from 399 to 622, Measured),
  so a residual-privacy audit of a sample of it comes before any freeze (task 1.3).
- A turn with 0 assistant messages belongs to the level 0–2.

### The question, its state and its label

- `turn_cost` is a Score on 4 levels: 0–2, 3–8, 9–30, 31 or more assistant messages. The levels hold 1.9%, 7.4%, 29.1%
  and 61.7% of cache reads. The thresholds come from the distribution of all turns, before any model or baseline
  result; a sensitivity row repeats the routing results at 21 or more.
- One row per turn of an interactive session. The state holds: the user message (at most 600 characters); the
  previous user message; the last assistant text of the previous turn (at most 300 characters, scrubbed); the number
  of earlier turns; the level of the previous turn; and the context size when the message arrives (the cache read of
  the last assistant message before it, in thousands of tokens). All of these exist before the turn starts. Other
  states leave assistant text out because "text before a call can name the call" (SESSIONS_METHOD.md §2); here the
  text comes from an earlier turn, so it names nothing of this turn, and it holds the plan that a short "go" approves.
  The extract, the scrub and `check()` cover it (task 1.1); lengths are measured after the scrub.
- The label is the level of the turn's number of assistant messages (log label).

### Splits and evaluation (fixed before any result)

1. **Primary split: time per project** (`time-per-project`), because the router serves the user's future turns, mostly
   in known projects. **Secondary: leave one project out** (v9 rules), for transfer to a new project. This deviates
   from the v9 rule "leave one project out is primary" for the reason above.
2. **Baselines**, counted on train: class prior; user message length (level distribution per length tercile); short-reply
   keywords (a user message of 20 characters or fewer that holds one of a fixed list as a whole word or phrase: go, yes,
   ok, continue, sure, do it, implement, apply, fix all, push, deploy); previous turn
   (level given the previous level); context size (level per context-size tercile). Plus the zero-shot readout with a
   bias and temperature fitted on train. The baselines are also counted on train,
   so val compares them fairly: a fit on val would score the model on its own fit data.
3. **Measures:** log loss; ordinal accuracy (the share of rows whose predicted level is within one level of the true
   level); and for the routing decision "31 or more", recall and precision.
4. **Small splits:** routing measures (recall, precision, flag rate) of a split with fewer than 10 long turns are
   counts only, no claim (as `waste` in add-decision-patterns). Log loss and ordinal accuracy are reported for every
   split. So the time split gives log loss and ordinal accuracy only (3 long val turns, 3 long test turns).
   **Routing claims** come from pooled out-of-fold predictions over the leave-one-project-out folds: every turn of a
   fold project is predicted by a model that did not train on its project. The pooled set is the union of the 4 fold
   test sets, so it is computed only inside the final report (step 5). The pooled set holds 251 turns in 31
   sessions, with 33 long turns (base rate 0.131); the 5 turns of the 3 projects below 50 calls are always in train
   (Measured; these are label counts only, no model result). The pooled numbers are labelled pooled. This deviates from the v9 rule "never pooled" because every fold
   has fewer than 10 long turns in val, and 2 of 4 have fewer than 10 in test.
5. **Test is read once** per split and fold, by one final report. The final report also computes the pooled routing
   numbers, the upper bound and the go or no-go decision, because all three use the fold test sets. Everything before
   it (baselines, the choice of the deployed predictor, its threshold) uses train and val only.
6. **The upper bound:** the cache reads of the long turns that the deployed predictor flags in the pooled
   out-of-fold predictions. It is loose (it
   assumes routing removes all of their cache reads) and is reported in cache reads, never as tokens, cost or savings.

### The deployed predictor and its threshold (fixed now)

- The deployed predictor is the baseline with the lowest val log loss on the primary split. A trained or zero-shot
  model replaces it only if it beats that baseline on val: the paired session bootstrap (below) of the log-loss
  difference has a 95% interval below zero. A model that could replace the baseline also runs per fold, so that it
  has pooled out-of-fold predictions for the go or no-go test.
- The chosen predictor is refitted on all 256 turns for the pilot.
- Threshold rule: a turn is **flagged** when the predictor's score P(31 or more) is at or above the score of the
  train turn at the 80th percentile of its train scores (nearest rank: the score of a real train turn, never an
  interpolated value) (the top 20%; the report also gives the top 10%). Ties are flagged
  together, so the real flag rate can differ from 20%; the report gives the real flag rate. For the pooled numbers,
  each fold computes its threshold from its own train; for the pilot, from all 256 refitted turns. A fixed quantile
  flags turns; a fixed probability (for example 2 times the base rate) flags none with these baselines (see Context).
- A predictor without spread (one score for every turn, such as the class prior) cannot rank turns: it is a no go.
- **Go or no go:** the pilot runs only if the deployed predictor's pooled precision at the top 20% is above the pooled
  base rate (0.131), by a one-sided 95% interval from a session bootstrap: 2000 resamples of the 31 sessions with
  replacement (seed 2026), precision recomputed on the pooled predictions of each resample, the fold thresholds kept
  as fixed. Otherwise the change ends with the offline report. Inferred: about 50 flags need a precision of roughly
  23% to 25% to pass, and the best single feature reaches 19%, so a no go is the expected result.
- **Session bootstrap:** every interval in this change resamples sessions, not turns, because turns of one session
  depend on each other (momentum): 2000 resamples with replacement, seed 2026, percentile intervals. The go or no-go
  bound is the one-sided 5th percentile of the resampled precision, against the fixed pooled base rate (0.131); the
  predictor choice uses the two-sided 95% interval of the paired log-loss difference.

### Feasibility pilot

- A `UserPromptSubmit` hook, in interactive sessions only (the entry point in the transcript is `cli` or
  `claude-desktop`), builds the same state as the offline rows from the `transcript_path` in the hook input, applies the
  same scrub (with `private-strings.txt`), and sends it to the local minijev API (a warm model; the API stays running).
- For a flagged turn, a random draw with a logged seed and probability 0.5 decides: add the **hint**, or add nothing.
  Only flagged turns are drawn, and a flagged turn inside an open task episode gets no new draw (see the task episode
  below). A hint stays in the context for the rest of the session, so later turns of that session
  carry it (carry-over): the false-flag share uses only no-hint turns with no earlier hint in their session, and a
  later savings study draws per session. The hint is one line, for example "This looks like a long task: plan first, and batch independent reads into
  one message."
- The hook always exits with code 0. For `UserPromptSubmit`, exit code 2 blocks the user message (Stated, Claude Code
  hooks docs). A test forces an exception and checks the exit code.
- If the API does not answer within 1 second, the hook adds nothing, and logs the turn as "not assigned" (not as the
  no-hint arm).
- The pilot log is under `~/.minijev-private`. It stores the session id, turn index, prediction, draw and timing, not
  the text.
- Later data versions mark the turns of the pilot period, because a hinted turn does not show how the user works
  without routing.
- The pilot reports: latency (p50, p95) and the timeout rate over all turns; the flag rate; the false-flag share, from
  the flagged turns without the hint and with no earlier hint in their session (the hint aims to change turn length), a rough estimate (about ±25 points at
  15 turns, Inferred); and the tokens that the hint adds. The primary outcome for any later savings study is fixed now:
  cache reads per **task episode**: the flagged turn and the turns after it, until the next user message longer than
  20 characters, a `/clear`, or the session end. In the pilot, a flagged turn inside an open episode gets no new draw:
  it joins that episode, which keeps its first draw, and the log marks it.
- The pilot runs until 30 turns are flagged, or 8 weeks, whichever comes first. The count sets the size of the
  false-flag sample. On the first user message of a session, the transcript can lack an entry with an entry point
  (unknown); task 3.2 checks this, and the hook adds nothing when it cannot confirm an interactive session.

### Why the pilot cannot measure savings

With the standard deviation of log(cache reads) over long turns (0.83, Measured), detecting a 30% drop with 80% power
at a 0.05 level needs about 86 long turns per arm. The user has about 2 long turns per week (33 in 17 weeks), so
the pilot would need more than 80 weeks (Measured inputs, Inferred result). A **paired replay** can measure savings with
fewer turns: replay logged user messages of long turns twice, with and without the hint, on a checkout of the logged
commit, through `claude -p`, and compare cache reads per episode. It costs API tokens, the replay differs from live
work (no user in the loop), and it is a separate decision for the user.

## Risks / Trade-offs

- [256 turns; 33 long ones] → counts only below 10 long turns per split; pooled out-of-fold numbers are labelled.
- [Thresholds of the levels chosen on all turns] → they use only the label distribution, not any feature; the
  sensitivity row at 21 or more shows whether a conclusion depends on the cutoff.
- [Momentum: a long turn follows a long turn] → the previous-turn baseline measures how much momentum alone predicts.
- [False flags] → at a 13% base rate, many flags are false; each costs the hint tokens and can add planning overhead.
  The pilot counts false flags and hint tokens; it does not measure the overhead.
- ["Plan first" can split a task into a plan turn and a "go" turn] → the savings outcome is per task episode, not per
  turn.
- [A hook on every user message adds latency] → a warm model, a 1 s timeout, fail open.
- [Pilot turns in later training data] → marked, and left out of the next version.
