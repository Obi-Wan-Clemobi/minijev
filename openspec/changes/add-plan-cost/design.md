## Context

add-turn-cost-routing read the test splits of `turn_cost` in v12 and v13 (docs/SESSIONS_METHOD.md §7.3). Its facts:

- 256 interactive turns up to the v13 freeze; 33 long turns (31 or more assistant messages); base rate 0.131
  (Measured).
- No feature of the user message beats the prior by more than 0.03 in test log loss, in any split (Measured).
- At the top 10% of train scores, the keyword baseline flags only short replies such as "go": 11 of 36 flagged turns
  were long (Measured, pooled over 4 folds). This is a description: the rule did not use the top 10%.
- The user has about 2 long turns per week: 33 in 17 weeks (Measured).

Every result in this change comes from data that we have not scored yet: the held-out predictions on development
data, and the prospective test. Development data is not clean: we saw its test labels and the "go" pattern. So a
development result is exploratory, and only the prospective test decides.

## Goals / Non-Goals

**Goals:**
- Find out whether the plan text predicts a long turn, on turns that nobody has seen.
- Fix every rule before the prospective test holds any turn.

**Non-Goals:**
- A hook or a routing pilot. A go permits a later change to reuse the pilot design of add-turn-cost-routing.
- A fine-tune. Development data holds about 33 long turns, too few to train an adapter (Inferred).
- Savings. As in add-turn-cost-routing, a savings test needs a paired replay.

## Decisions

### Data

- **The cutoff** is 2026-09-27 00:00 America/Toronto (2026-09-27T04:00:00Z). A session belongs to the side of the
  cutoff where its first entry falls. A session never has turns on both sides.
- **Development data:** every interactive turn of every session that starts before the cutoff. It is frozen once, after
  a sync, as a new version with one split, `dev`.
- **The prospective test:** every interactive turn of every session that starts at or after the cutoff. It is frozen
  once, as a new version with one split, `test`, by the stopping rule below. Nothing reads it before that.
- The sessions of this change's own work (in the minijev project) are data like any other. The threats list says why.

### The question and its state

- `plan_cost` is a Noul: "Will the coding agent need 31 or more assistant messages for this user message?". The
  label comes from the log: yes if the turn has 31 or more assistant messages.
- The state holds, in this order:
  1. The end of the previous answer: the last 1000 characters of the previous turn's last assistant text, scrubbed.
     At the session start, the line "No previous answer (the session starts here)".
  2. The user message: at most 600 characters.
  3. The level of the previous turn (as in `turn_cost`), or "none".
- The state holds nothing of the turn itself and no project name.

### Baselines (fixed now)

Each baseline maps a turn to a category, and predicts P(long) of that category on the training data (add-one
smoothing). Tercile cuts come from the training data.

| Baseline | Category |
|---|---|
| prior | one category |
| keywords | a short reply with a keyword, or not (the rule of `turn_eval.is_keyword_reply`) |
| plan_length | the length of the plan text, in terciles; "none" without a previous answer |
| plan_steps | the number of list lines in the plan text (lines that start with `-`, `*`, or a number and `.`), in terciles; "none" without a previous answer |
| plan_question | the last 200 characters of the plan text hold a `?`, or not; "none" without a previous answer |
| keywords_steps | keywords crossed with plan_steps |

The model is the zero-shot Noul readout of the base model, with a bias and temperature fitted on the training data.

### Choice on development data (fixed now)

1. **Cross-validation:** a **CV group** is one of 5 groups of whole sessions. Sort the session ids, shuffle them with
   `random.Random(2026)`, and deal them in turn into groups 0 to 4. Each turn gets an held-out prediction from a
   predictor fitted on the other 4 CV groups. (A CV group is not a fold: a fold holds out one project.)
2. **The measure** is the log loss of the held-out predictions.
3. **The deployed predictor** is the baseline with the lowest held-out log loss. The model replaces it only if the
   two-sided 95% paired session-bootstrap interval (2000 resamples, seed 2026) of the log-loss difference is below
   zero.
4. **The threshold:** a turn is flagged when its score is at or above the score of the training turn at the 80th
   percentile of the training scores (nearest rank; ties flagged together). This is the go rule of
   add-turn-cost-routing, so the two results stay comparable.
5. **The tie guard:** a predictor whose threshold flags more than 30% of its own training turns cannot rank turns
   (the failure of the keyword baseline in §7.3). The choice skips it and takes the next best by rule 3.
6. **The screen:** the pooled held-out precision at the threshold must exceed the development base rate by a
   one-sided 95% session-bootstrap bound. If it does not, the change ends with a no go now, and no prospective test
   is collected.
7. The chosen predictor is refitted on all development turns, and its threshold comes from all development scores.
   Both are stored with a sha256 before the prospective freeze.

### The prospective test (fixed now)

- **The stopping rule:** freeze the test on 2026-11-22 (8 weeks after the cutoff), with the sessions that started
  before that date. The date does not depend on the outcome, so the bootstrap treats the test as a fixed sample. At
  about 2 long turns per week, the test holds about 16 long turns (Inferred). Nothing counts long turns in the test
  before the freeze.
- A residual-privacy audit of 100 random plan texts from the test sessions comes before the freeze (as in
  add-turn-cost-routing task 1.3).
- **Go or no go:** the frozen predictor and threshold run once on the test. The report gives the precision and its
  one-sided bound. Go if the precision of the flagged turns
  exceeds the test base rate by a one-sided 95% session-bootstrap bound (2000 resamples, seed 2026; the base rate is
  fixed, not resampled). With fewer than 10 long turns, the report gives counts only, and the
  decision is "undecided".
- The report also gives recall, the real flag rate, log loss, and the loose upper bound in cache reads (never
  savings).
- Inferred: with about 120 test turns and 24 flags, precision needs about 0.30 to pass against a base rate of 0.13.

## Risks / Trade-offs

- [The user knows the hypothesis, and can change how they approve plans] → the test measures the user's work after the
  cutoff; the report states this threat.
- [Claude's plan style changes with the model version] → the extract records the model of each assistant message,
  and the report gives the model versions of the development data and the test.
- [Few long turns] → "undecided" below 10, and the screen that can end the change early.
- [Development data is not clean] → its results are exploratory; only the prospective test decides.
- [Plan text holds more private data than user text (task 1.1 of add-turn-cost-routing)] → the audit before the freeze.
