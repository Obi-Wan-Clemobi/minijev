## Why

Flows chain minijev questions into a state machine, but today every step asks one question in sequence, and every
step's state is the request plus the earlier decisions. That state changes at every step, so the state cache misses
and each step prefills the whole request again (Observed, `flows.state_for`). Decision trees such as "kind of work,
then the tool for that kind" and ranked preferences ("ASCII or HTML mockup, and when?") also need options that follow
an earlier answer, a full ranking, and a way to hand a decision to the large model.

## What Changes

- **Cached request**: the state of every step is the request only; earlier decisions go into the question block.
- **Fan-out step**: one step asks several independent questions, answered in one packed forward pass.
- **Options from an earlier answer**: a Choice step can take its options from a map keyed by an earlier step's answer.
- **Ranked output**: every Choice decision carries all options in order, with probabilities.
- **Hand-off exit**: a terminal `ESCALATE`, reached when a transition's confidence condition sends the decision to the
  large model.
- **Cost report**: each run reports the prefill tokens it used.

## Capabilities

### New Capabilities
- `flow-engine`: state machines of minijev questions, with cached requests, fan-out, dependent options, ranking and
  hand-off.

### Modified Capabilities

## Impact

- `src/minijev/flows.py`, its API route in `src/minijev/api/`, the State machine page in `web/`, and flow templates in
  `poc/flows/`.
- The cached-request change changes what each step's model reads, so flow answers can change. Existing flow templates
  are re-run and their outputs compared.
