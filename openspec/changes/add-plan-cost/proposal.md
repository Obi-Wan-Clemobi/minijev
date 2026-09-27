## Why

The turn-cost final report (add-turn-cost-routing, docs/SESSIONS_METHOD.md §7.3) gave no go: no feature of the user
message predicts a long turn better than the prior. One pattern stood out: turns that start with a short reply such as
"go" were long 2.3 times as often as the base rate (11 of 36, a description, not a claim). Inferred: in these turns,
the previous answer holds a plan, and the user approves it. The size of that plan can predict the length of the turn.

The test splits of `turn_cost` in v12 and v13 were read, and the "go" pattern came from them. So the 256 turns up to
the freeze can only develop this question. A clean test needs turns that nobody has seen: turns from sessions that
start after a cutoff.

Terms: a **long turn** has 31 or more assistant messages (as in add-turn-cost-routing). The **plan text** is the end
of the previous turn's last assistant text. The **cutoff** is the time after which new sessions go only into the
prospective test. A **prospective test** is a test set collected after the rules and the predictor are fixed.

## What Changes

- A new question, `plan_cost`: a Noul, "will this turn be long?", asked when the user message arrives. Its state leads
  with the plan text (up to 1000 characters), then the user message.
- The cutoff is fixed now: 2026-09-27 00:00 America/Toronto. Every session that starts before it is development data.
  Every interactive session that starts after it goes into the prospective test.
- Baselines and a zero-shot model are chosen on development data by session cross-validation, with rules fixed now.
- The deployed predictor and its threshold are frozen before anything reads the prospective test.
- A go or no-go rule on the prospective test, frozen on 2026-11-22 (8 weeks after the cutoff) and read once.
- Go means only that a later change can build the routing pilot of add-turn-cost-routing with this predictor. This
  change builds no hook.

## Capabilities

### New Capabilities
- `plan-cost`: predict a long turn from the plan text and the user message, and test the predictor prospectively.

### Modified Capabilities

## Impact

- `src/minijev/sessions/`: the `plan_cost` question, cross-validation on development data, a prospective freeze by
  session start time, and the final report.
- The SessionEnd sync hook keeps the raw logs; nothing new to install.
- No change to the flow engine or the API.
