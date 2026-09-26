## Context

`flows.run` asks one question per step through `ask(request)`. The request is `{"state": state_for(query,
decisions), "questions": {step.id: ...}}`. The engine has a state cache (MINIJEV_STATE_CACHE) keyed by the state's
tokens, and a packed mode that answers several questions of one request in one forward pass.

## Goals / Non-Goals

**Goals:**
- Fewer prefill tokens per flow, by reusing the request's cache across steps.
- Parallel questions at the cost of one forward pass.
- Decision trees whose later options depend on earlier answers, and ranked answers for preferences.
- A clean exit to the large model when minijev is unsure.

**Non-Goals:**
- Running tool calls. A flow decides; the caller acts.
- Learning flows from data. Flow structure stays hand-written; the data work is in `add-decision-patterns`.

## Decisions

- **Earlier decisions move into the question block.** The state stays the request, so its prefill is cached after
  the first step. Each question's block starts with "Decided so far: …". Alternative: keep decisions in the state and
  cache the request as a prefix. Rejected: the state cache keys on whole states, and a prefix cache is a larger change.
- **Fan-out uses the existing multi-question request.** A fan-out step lists several questions; all go into one
  request, so packed mode answers them in one pass. Transitions can test any of the answers (`from_question`).
- **Dependent options are a map, not code.** A Choice step may set `options_from: {"step": id, "map": {answer:
  [options]}}`. `check` fails if an answer of that step has no entry.
- **Ranking is always returned.** The decision event of a Choice step lists all options sorted by probability. A
  flow does not need a new step type for it.
- **ESCALATE is a terminal like DONE.** The end event names the step and the reason (the confidence and the threshold).

## Risks / Trade-offs

- [Moving decisions into the question block changes answers] → re-run the existing templates and report the
  answers that change; the old behaviour is not kept as an option.
- [Fan-out questions are independent by construction] → a fan-out step cannot use another fan-out question's answer;
  `check` rejects such a flow.
- [Long option maps] → `check` reports steps with more than 26 options (the label letters).
