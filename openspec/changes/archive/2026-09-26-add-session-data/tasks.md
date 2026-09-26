## 1. Pipeline

- [x] 1.1 Parse main-thread logs into turns and calls, with diagnostics (`sessions/logs.py`)
- [x] 1.2 Two scrub layers and `check` (`sessions/scrub.py`, `sessions/dataset.py`)
- [x] 1.3 States with `pending` for parallel calls (`sessions/questions.py`, `tests/test_sessions.py`)
- [x] 1.4 Question registry, log-label questions and the external `needs_approval` question
- [x] 1.5 Frozen versions with hashes, provenance and the split log
- [x] 1.6 `sync`, `extract`, `sample`, `check`, `stats`, `freeze`, `points` commands

## 2. Labels

- [x] 2.1 Consensus rules and command (`sessions/consensus.py`), with tests
- [x] 2.2 `needs_approval` consensus labels, frozen as v6

## 3. Evaluation

- [x] 3.1 Baselines: prior, previous call, zero-shot, zero-shot with bias and temperature (`sessions/evaluate.py`)
- [x] 3.2 Fixed adapter comparison, `compare` (smoke-tested on a synthetic root)
- [x] 3.3 `train_lora.py --task sessions-<question>`, with test blocked

## 4. Docs

- [x] 4.1 `docs/SESSIONS.md` and `docs/SESSIONS_METHOD.md`
