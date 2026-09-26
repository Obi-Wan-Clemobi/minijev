## ADDED Requirements

### Requirement: Steps reuse the request's cached prefill

The state of every step SHALL be the request only. Earlier decisions SHALL be given in the question block. With the
state cache on, only the first step of a run SHALL prefill the request.

#### Scenario: A three-step flow
- **WHEN** a flow runs three steps on one request with the state cache on
- **THEN** the request is prefilled once, and the run reports fewer prefill tokens than three full prefills

### Requirement: A fan-out step asks several questions in one pass

A step MAY list several questions. They SHALL be answered in one request, and each answer SHALL be recorded as a
decision. A transition of that step MAY test the answer of any one of its questions. A fan-out question SHALL NOT
depend on another question of the same step.

#### Scenario: Two independent questions
- **WHEN** a fan-out step asks "is it urgent?" and "which team?"
- **THEN** both answers come from one forward pass, and a transition can test either answer

### Requirement: Options can follow an earlier answer

A Choice step MAY take its options from a map keyed by an earlier step's answer. `check` SHALL fail if an answer of
that earlier step has no entry in the map.

#### Scenario: Tools for a kind of work
- **WHEN** the earlier step answered "research"
- **THEN** the Choice step offers only the options mapped to "research"

### Requirement: Choice decisions are ranked

The decision event of a Choice step SHALL list every option in order of probability, with its probability.

#### Scenario: A preference
- **WHEN** a step asks "ASCII mockup or HTML mockup?"
- **THEN** the decision lists both options with their probabilities, highest first

### Requirement: A flow can hand a decision to the large model

`ESCALATE` SHALL be a terminal target like `DONE`. A run that reaches it SHALL end with status `escalated`, the step,
and the confidence that sent it there.

#### Scenario: Low confidence
- **WHEN** a transition with `confidence_lt: 0.4` targets `ESCALATE` and the step's confidence is 0.3
- **THEN** the run ends with status `escalated` at that step

### Requirement: A run reports its cost

The end event of a run SHALL report the prefill tokens used and the number of forward passes.

#### Scenario: Cost of a run
- **WHEN** a run ends
- **THEN** its end event has the prefill token count and the number of forward passes
