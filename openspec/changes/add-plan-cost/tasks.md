## 1. Data

- [x] 1.1 `plan_cost` question: state and label as specified; tests
- [x] 1.2 Record the model of each assistant message in `logs.parse` and the extract; tests
- [x] 1.3 Freeze by session start time: `--split before-cutoff` or `after-cutoff`, with `--cutoff` and `--until`; a
      version with one split (`dev` or `test`); every other reader skips the sealed sessions; tests
- [ ] 1.4 Sync, then freeze development data (sessions before the cutoff)

## 2. Choice on development data

- [ ] 2.1 The CV groups (design.md), the six baselines and the zero-shot readout; held-out log loss; tests
- [ ] 2.2 The choice rule, the tie guard and the screen; store the refitted predictor and threshold with a sha256
- [ ] 2.3 Results into docs/SESSIONS_METHOD.md, labelled exploratory; if the screen fails, the change ends here

## 3. Prospective test

- [ ] 3.1 On 2026-11-22: sync, then the privacy audit of 100 random plan texts from the test sessions
- [ ] 3.2 Freeze the test: `--split after-cutoff --cutoff 2026-09-27T04:00:00Z --until 2026-11-22T05:00:00Z` (EST,
      UTC-5)
- [ ] 3.3 The final report (reads test once, ledger first): precision and its bound, recall, flag rate, log loss,
      the loose upper bound, the model versions; go, no go or undecided; results into docs/SESSIONS_METHOD.md
