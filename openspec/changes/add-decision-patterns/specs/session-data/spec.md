## ADDED Requirements

### Requirement: The split strategy is a parameter

`split_of` SHALL take a split strategy. The strategies are `time-per-project` (the v1 to v6 split) and
`leave-one-project-out` with a named held-out project. For `leave-one-project-out`, the held-out project SHALL be the
test split, val SHALL be the newest 15% of calls of each training project, and projects with fewer than 50 calls SHALL
always be in train. The manifest SHALL record the strategy and the held-out project.

#### Scenario: A leave-one-project-out fold
- **WHEN** a version is frozen with strategy `leave-one-project-out` and held-out project `minijev`
- **THEN** every `minijev` row is in test, no `minijev` row is in train or val, and the manifest names the strategy

#### Scenario: A small project
- **WHEN** a project has fewer than 50 calls
- **THEN** its rows are in train in every fold

### Requirement: The state can leave out the project name

The state builder SHALL take an option that leaves out the `Project:` line. Versions for decision patterns SHALL use
it.

#### Scenario: State without project
- **WHEN** a version is frozen for decision patterns
- **THEN** no state contains a `Project:` line
