## 1. Data

- [ ] 1.1 `plan_cost` question: state and label as specified; tests
- [ ] 1.2 Record the model of each assistant message in `logs.parse` and the extract; tests
- [ ] 1.3 Freeze by session start time: `--before` and `--after` a cutoff; a version with one split (`dev` or `test`);
      tests that a session never spans both
- [ ] 1.4 Sync, then freeze development data (sessions before the cutoff)

## 2. Choice on development data

- [ ] 2.1 The CV groups (design.md), the six baselines and the zero-shot readout; held-out log loss; tests
- [ ] 2.2 The choice rule, the tie guard and the screen; store the refitted predictor and threshold with a sha256
- [ ] 2.3 Results into docs/SESSIONS_METHOD.md, labelled exploratory; if the screen fails, the change ends here

## 3. Prospective test

- [ ] 3.1 On 2026-11-22: sync, then the privacy audit of 100 random plan texts from the test sessions
- [ ] 3.2 Freeze the test (sessions that start from the cutoff to 2026-11-22)
- [ ] 3.3 The final report (reads test once, ledger first): precision and its bound, recall, flag rate, log loss,
      the loose upper bound, the model versions; go, no go or undecided; results into docs/SESSIONS_METHOD.md
