## 1. Split and state

- [ ] 1.1 Add the split strategy parameter to `split_of`, with `leave-one-project-out`; record it in the manifest
- [ ] 1.2 Add the option to leave out the `Project:` line; tests for both
- [ ] 1.3 Freeze v7 (one folder per fold, or one manifest with fold ids); list train label counts per fold

## 2. Token breakdown

- [ ] 2.1 `tokens.py` and `minijev sessions tokens`: the four token counts per kind of work and per chain pattern
- [ ] 2.2 Write the Measured breakdown into `docs/SESSIONS_METHOD.md`

## 3. Questions

- [ ] 3.1 Write and commit the `work_kind` table (before any fold result)
- [ ] 3.2 `work_kind`, `tool_given_kind` and `waste` questions, with tests
- [ ] 3.3 `area` question (external); label it with the consensus pipeline, request only

## 4. Evaluation

- [ ] 4.1 Per-fold count baselines and supported-label lists
- [ ] 4.2 Per-fold zero-shot and bias-temperature baselines
- [ ] 4.3 Risk-coverage report
- [ ] 4.4 Ladder against flat `next_tool` against previous call, per held-out project
- [ ] 4.5 One pilot adapter on one fold; `compare` on its held-out project

## 5. Docs

- [ ] 5.1 Add "decision pattern", "domain" and "fold" to the CLAUDE.md glossary, and define them in `docs/SESSIONS_METHOD.md`
