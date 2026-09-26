## ADDED Requirements

### Requirement: Turn cost is a Score from the log

The `turn_cost` question SHALL be a Score with 4 levels: 0–2, 3–8, 9–30, and 31 or more assistant messages in the turn.
It SHALL have one row per turn of an interactive session. Its label SHALL be the level of the turn's number of
assistant messages.

#### Scenario: A long turn
- **WHEN** a turn has 40 assistant messages
- **THEN** its label is the level "31 or more"

#### Scenario: A turn without an answer
- **WHEN** a turn has 0 assistant messages
- **THEN** its label is the level "0–2"

### Requirement: Turns follow the context

A `/clear` SHALL start a new context: the next turn SHALL have no previous user message and no previous level. A
compact summary entry SHALL NOT be a turn. Interrupted turns SHALL be marked.

#### Scenario: After /clear
- **WHEN** a user message follows a `/clear`
- **THEN** its state has no previous user message and no previous level

#### Scenario: A compact summary
- **WHEN** the log holds a compact summary entry
- **THEN** it is not a turn and not a previous user message

### Requirement: The state holds only what exists when the user message arrives

The state of a `turn_cost` row SHALL hold the user message (at most 600 characters), the previous user message, the
last assistant text of the previous turn (at most 300 characters, scrubbed), the number of earlier turns, the level of
the previous turn, and the context size when the message arrives. It SHALL NOT hold any call, result or count of the
turn itself, or the project name.

#### Scenario: The first turn of a session
- **WHEN** a turn is the first of its session
- **THEN** its state has no previous user message and says that no turn came before

### Requirement: Evaluation uses fixed rules

The evaluation SHALL use the time-per-project split as primary and leave one project out as secondary. It SHALL report
the class prior, user-message-length, short-reply keyword, previous-turn and context-size baselines, counted on train,
and the zero-shot readout with a val-fitted bias and temperature. It SHALL give log loss and ordinal accuracy for every
split. Recall, precision and flag rate of "31 or more" SHALL come from pooled out-of-fold predictions over the
leave-one-project-out folds, labelled pooled; for a single split with fewer than 10 long turns they SHALL be counts
only. Every interval SHALL come from a bootstrap over sessions.

#### Scenario: A small fold
- **WHEN** a test fold holds 1 long turn
- **THEN** the report gives its counts and no recall or precision claim for that fold

### Requirement: The deployed predictor follows a fixed rule

The deployed predictor SHALL be the baseline with the lowest val log loss on the primary split, unless a model beats it
on val: the two-sided 95% paired session-bootstrap interval (2000 resamples, seed 2026) of the log-loss difference
is below zero. It SHALL flag a turn when its score is at or above the
score of the train turn at the 80th percentile of its train scores (nearest rank; ties flagged together, the real flag
rate reported). A predictor with one score for
every turn SHALL be a no go. The pilot SHALL run only if the pooled precision at that threshold exceeds the pooled base
rate by a one-sided 95% session-bootstrap interval (2000 resamples, seed 2026). The pooled numbers, the upper bound and
the go or no-go decision SHALL be computed only in the final report, because they use the fold test sets.

#### Scenario: No model beats the baselines
- **WHEN** no trained or zero-shot model beats the best baseline on val
- **THEN** the pilot serves the best baseline, refitted on all turns

#### Scenario: Too weak to pilot
- **WHEN** the pooled precision interval at the threshold includes the base rate
- **THEN** no pilot runs, and the change ends with the offline report

### Requirement: Offline figures are upper bounds

Any figure from the logs about routing SHALL be given in cache reads and labelled a loose upper bound. It SHALL NOT be
reported as tokens saved, cost saved or savings.

#### Scenario: A cache-read figure
- **WHEN** the report gives the cache reads of flagged long turns
- **THEN** it labels the number a loose upper bound on what routing could change

### Requirement: The feasibility pilot is safe and measured

A `UserPromptSubmit` hook SHALL run only in interactive sessions, build the offline state from the transcript with
the same scrub, and ask the local minijev API for a prediction. For a flagged turn, a seeded random draw with
probability 0.5 SHALL decide whether to add one hint line, except that a flagged turn inside an open task episode SHALL
get no new draw: it joins that episode's draw, and the log marks it. The hook SHALL always exit with code 0, SHALL add nothing and log "not assigned" if the API does not answer within
1 second, and SHALL log only ids, predictions, draws and timings, under `~/.minijev-private`. The pilot SHALL stop at
30 flagged turns or 8 weeks, and SHALL report latency, the timeout rate, the flag rate, the false-flag share (from
flagged turns without the hint and with no earlier hint in their session) and the hint tokens. It SHALL NOT report savings.

#### Scenario: The API is down
- **WHEN** the hook cannot reach the API within 1 second
- **THEN** the user message goes to Claude unchanged, the hook exits with code 0, and the turn is logged "not assigned"

#### Scenario: The hook fails
- **WHEN** the hook raises an exception
- **THEN** it exits with code 0 and adds nothing

#### Scenario: A flagged turn inside an open episode
- **WHEN** a turn is flagged while the episode of an earlier flagged turn is still open
- **THEN** it gets no new draw, keeps the episode's draw, and is marked in the log

#### Scenario: Hook and offline state agree
- **WHEN** the hook builds the state for a logged turn
- **THEN** it equals the offline state of that turn
