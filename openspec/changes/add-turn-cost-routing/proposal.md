## Why

The tokens of the user's Claude Code work are concentrated in a few long turns. In the 34 interactive sessions, the
34 turns with 31 or more assistant messages (13% of 257 turns) hold 61.7% of all cache reads; the 72 turns with at most
2 messages hold 1.9% (Measured, 2026-09-26). Every assistant message reads the whole context again from the cache
(Measured, docs/SESSIONS_METHOD.md §7.2), so the cost of a turn grows with its number of messages. A hint given once,
when the user message arrives, can change a whole turn (Inferred). Predicting the next tool (add-decision-patterns)
cannot: the large model still makes each call.

Terms: the **cost of a turn** is its number of assistant messages. **Routing** is adding one hint line to the context
of a turn that minijev predicts to be long. A **user message** is what the user types into Claude Code (the "request"
of a minijev API call is a different thing).

## What Changes

- A new question, `turn_cost`: a Score on 4 levels (0–2, 3–8, 9–30, 31 or more assistant messages), asked when the user
  message arrives, with labels from the log.
- Frozen versions with a time split (primary) and leave one project out (secondary), interactive sessions only.
- Baselines that use only what exists when the user message arrives, and a deployed predictor chosen by a rule fixed
  now.
- An offline upper bound on the cache reads that routing could change.
- A go or no-go rule: simple features raise P(long turn) to at most 1.7 times the base rate (Measured), so the pilot
  runs only if the deployed predictor's pooled precision beats the base rate with an interval that excludes it.
- If it goes, a feasibility pilot through a Claude Code `UserPromptSubmit` hook: it measures whether routing works in daily use
  (latency, timeouts, flag rate, false flags, hint tokens). It does not measure savings: at about 2 long turns per week,
  a savings test needs more than 80 weeks of the user's work (Measured variance; design.md).
- Savings need a separate method, a paired replay; it costs API money, and the user decides on it separately.

## Capabilities

### New Capabilities
- `turn-cost`: predict the cost of a turn from its user message, route predicted long turns with one hint, and measure
  the feasibility of routing.

### Modified Capabilities

## Impact

- `src/minijev/sessions/`: the `turn_cost` question, turn-definition fixes (`/clear`, compact summaries), baselines.
- A hook script, and install instructions (the user installs it).
- No change to the flow engine: a flow with a `turn_cost` step and a hand-off exit serves the hook.
