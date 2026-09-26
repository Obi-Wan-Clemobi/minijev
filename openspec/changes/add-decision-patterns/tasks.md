## 1. Split and state

- [x] 1.1 Add the split strategy parameter to `split_of`, with `leave-one-project-out`; record it in the manifest
- [x] 1.2 Add the option to leave out the `Project:` line; tests for both
- [x] 1.3 Freeze v8 (one manifest with fold ids, after the questions of 3.1 and 3.2 exist); list train label counts per fold
- [x] 1.4 Record each session's entry point; `--interactive-only` for `freeze`, `stats` and `tokens`
- [x] 1.5 Freeze the decision-pattern version from interactive sessions only (after the `area` labels), and list the
      train label counts per fold (v9)

## 2. Token breakdown

- [x] 2.1 `tokens.py` and `minijev sessions tokens`: the four token counts per kind of work and per chain pattern
- [x] 2.2 Write the Measured breakdown into `docs/SESSIONS_METHOD.md`

## 3. Questions

- [x] 3.1 Write and commit the `work_kind` table (before any fold result)
- [x] 3.2 `work_kind`, `tool_<kind>` and `waste` questions, with tests
- [x] 3.3 `area` question (external, interactive turns only, with the previous request as context); label it with
      the consensus pipeline (221 of 257 turns decided; it does not transfer across projects: design.md amendments)

## 4. Evaluation

- [x] 4.1 Per-fold count baselines and supported-label lists
- [ ] 4.2 Zero-shot and bias-temperature baselines on the pilot fold (design.md amendments)
- [x] 4.3 Risk-coverage report (val curve in `baselines`; val-chosen threshold applied to test in `compare`)
- [ ] 4.4 Ladder against flat `next_tool` against previous call, in the final report of each fold (`compare --fold`)
- [ ] 4.5 One pilot `work_kind` adapter on the board-game-event-planner fold (v9); `compare --fold` on it

## 5. Docs

- [x] 5.1 Add "decision pattern", "domain" and "fold" to the CLAUDE.md glossary, and define them in `docs/SESSIONS_METHOD.md`
