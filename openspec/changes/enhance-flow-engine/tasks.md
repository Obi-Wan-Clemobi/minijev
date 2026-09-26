## 1. Engine

- [ ] 1.1 Cached request: state is the request; decisions go into the question block; count prefill tokens per run
- [ ] 1.2 Re-run the existing flow templates; list the answers that change
- [ ] 1.3 Fan-out steps: several questions per step, one request; transitions test any of them; `check` rules
- [ ] 1.4 `options_from` for Choice steps; `check` fails on a missing map entry
- [ ] 1.5 Ranked options in Choice decision events
- [ ] 1.6 `ESCALATE` terminal and `escalated` status
- [ ] 1.7 Tests for 1.1 to 1.6 in `poc/tests/test_flows.py`

## 2. API and page

- [ ] 2.1 API: accept and return the new step fields and events
- [ ] 2.2 State machine page: fan-out steps, option maps, ranked options, the ESCALATE target

## 3. Example

- [ ] 3.1 A flow template for "kind of work, then tool" with a hand-off exit, using the `work_kind` table of
      `add-decision-patterns`
- [ ] 3.2 Measure prefill tokens and latency of that flow, before and after 1.1 and 1.3

## 4. Docs

- [ ] 4.1 Add "fan-out step" and "hand-off" to the CLAUDE.md glossary, and define them in the flow docs
