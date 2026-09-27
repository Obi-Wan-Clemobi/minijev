## ADDED Requirements

### Requirement: Plan cost is a Noul from the log

The `plan_cost` question SHALL be a Noul with one row per turn of an interactive session. Its label SHALL be yes when
the turn has 31 or more assistant messages. Its state SHALL hold three parts in this order:

1. The last 1000 characters of the previous turn's last assistant text, scrubbed.
2. The user message, at most 600 characters.
3. The level of the previous turn.

The state SHALL NOT hold anything of the turn itself or the project name.

#### Scenario: A turn after a plan
- **WHEN** the previous answer ends with a plan and the user replies "go"
- **THEN** the state starts with the end of that plan, followed by "go"

#### Scenario: The first turn of a session
- **WHEN** a turn is the first of its session
- **THEN** its state says that no previous answer exists

### Requirement: The cutoff separates development data and the prospective test

A session SHALL belong to development data if its first entry is before 2026-09-27T04:00:00Z. Otherwise it SHALL
belong to the prospective test. No session SHALL have turns in both. Nothing SHALL read the prospective test before
its freeze.

#### Scenario: A session that spans the cutoff
- **WHEN** a session starts before the cutoff and continues after it
- **THEN** all its turns are development data

### Requirement: The deployed predictor is chosen on development data by fixed rules

The choice SHALL use held-out predictions from 5 CV groups of whole sessions, dealt as design.md specifies (seed
2026). The rules SHALL be:

1. The deployed predictor is the baseline with the lowest held-out log loss.
2. The zero-shot model replaces it only if the two-sided 95% paired session-bootstrap interval of the log-loss
   difference is below zero.
3. The choice skips a predictor whose threshold flags more than 30% of its own training turns.
4. The pooled held-out precision at the threshold must exceed the development base rate by a one-sided 95%
   session-bootstrap bound. Otherwise the change ends with a no go, and nobody collects a prospective test.

#### Scenario: Ties flag too many turns
- **WHEN** a baseline's threshold flags 90% of its training turns
- **THEN** the choice skips it

#### Scenario: The screen fails
- **WHEN** the held-out precision bound is at or below the development base rate
- **THEN** the change ends with a no go before any prospective test

### Requirement: The prospective test decides, once

The prospective test SHALL hold the interactive sessions that start from the cutoff to 2026-11-22. It SHALL be frozen
on 2026-11-22, after a privacy audit of 100 random plan texts. The frozen predictor and threshold SHALL run on it
once. The report SHALL give the precision of the flagged turns and its one-sided 95% session-bootstrap bound (2000
resamples, seed 2026). The decision SHALL be go only if that bound exceeds the test base rate. With fewer than 10 long
turns, the report SHALL give counts only, and the decision SHALL be "undecided". The report SHALL label figures about
cache reads a loose upper bound, and SHALL NOT call them savings.

#### Scenario: Too few long turns
- **WHEN** the frozen test holds 8 long turns
- **THEN** the report gives counts and the decision "undecided"

#### Scenario: A second read
- **WHEN** the final report runs a second time
- **THEN** it stops before it reads the test, because the test-read ledger holds the first read
