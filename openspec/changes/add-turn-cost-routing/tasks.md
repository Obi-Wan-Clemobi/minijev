## 1. Data

- [x] 1.1 Turn fixes in `logs.parse`: `/clear` starts a new context (check whether it is a command entry, a new session
      file, or both); compact summaries are not turns; interrupted turns are marked; check whether messages typed while
      Claude works split turns, and merge them if so; keep the last assistant text of each turn, and add it to the
      scrub and to `check()`; tests
- [x] 1.2 `turn_cost` question: one row per interactive turn, state as specified (including the previous assistant text,
      scrubbed, and the context size), label from the message count; tests
- [x] 1.3 Audit a random sample of 200 last-assistant texts for residual private data (a new reviewer, as for v3), add
      what it finds to `private-strings.txt`; then freeze versions: time per project (primary) and leave one project out (secondary), interactive only, no
      project line; list the long turns per split and fold

## 2. Offline evaluation

- [x] 2.1 Baselines (prior, user message length, short-reply keywords, previous turn, context size), counted on train,
      with log loss and ordinal accuracy on val; the threshold rule (nearest rank, ties together); the session bootstrap;
      tests
- [x] 2.2 Zero-shot readout with a bias and temperature fitted on train, on the primary split; per fold as well if it can
      replace the baseline (it then needs pooled out-of-fold predictions; they are stored, not scored, until 2.4)
- [x] 2.3 Choose the deployed predictor by the fixed rule, on val only
- [x] 2.4 The final report (reads test once): per split and fold log loss and ordinal accuracy; pooled out-of-fold
      recall, precision and real flag rate at the top 20% and 10%; the sensitivity row at 21 or more; the loose upper
      bound in cache reads; the go or no-go decision; results into docs/SESSIONS_METHOD.md

## 3. Feasibility pilot

- [ ] ~~3.1~~ (not run: the final report gave no go) A flow template "turn cost" (Score step, hand-off at the threshold rule) served by the API
- [ ] ~~3.2~~ (not run: the final report gave no go) The `UserPromptSubmit` hook: interactive sessions only (check the first message of a session: no entry point
      yet means add nothing); state from `transcript_path`; seeded draw among flagged
      turns, with episode tracking (a flagged turn inside an open episode joins its draw); exit code 0 on every path; 1 s timeout logged "not assigned"; private log of ids only; tests with a fake
      API, a forced exception, and hook–offline state parity
- [ ] ~~3.3~~ (not run: the final report gave no go) Install instructions in docs/SESSIONS.md (the user installs the hook); mark pilot-period turns in later
      versions
- [ ] ~~3.4~~ (not run: the final report gave no go) The pilot report at 30 flagged turns or 8 weeks: latency, timeout rate, flag rate, false-flag share, hint
      tokens

## 4. Decision for the user

- [ ] 4.1 Paired replay for savings (costs API tokens): the user decides whether to build it
