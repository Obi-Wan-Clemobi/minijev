## ADDED Requirements

### Requirement: Session data stays outside git

The system SHALL write all session data (the raw archive, extracts, labels, versions, results and adapters) under one
private root, `~/.minijev-private` or `MINIJEV_PRIVATE`. It SHALL refuse a root inside a git work tree unless the user
passes `--allow-git`.

#### Scenario: Root inside a repository
- **WHEN** the private root is inside a git work tree and `--allow-git` is not given
- **THEN** the command stops with an error and writes nothing

### Requirement: Logs are archived without loss

`minijev sessions sync` SHALL copy new and grown `.jsonl` logs from the Claude Code projects folder to the archive,
and SHALL never delete a file from the archive.

#### Scenario: Claude Code removed an old log
- **WHEN** a log is gone from `~/.claude/projects` but exists in the archive
- **THEN** the archive keeps it

### Requirement: Parsing uses the main thread only and reports what it could not use

Extraction SHALL use only real user turns and main-thread tool calls. It SHALL leave out meta entries, command
wrappers, system reminders and subagent transcripts. It SHALL count bad lines, calls with unknown status and sessions
without turns.

#### Scenario: A log format change
- **WHEN** most calls in an extract have status "unknown"
- **THEN** the extract diagnostics show that count

### Requirement: A state holds only what existed before the decision

A state SHALL hold the user's turn and the calls of that turn before the decision, with their status. It SHALL NOT
hold tool results, Claude's text, or the decided call itself, except for questions whose decision is about that call.
A call from the same assistant message as the decision SHALL show the status `pending`.

#### Scenario: Parallel calls
- **WHEN** two calls come from one assistant message
- **THEN** the state of the second call shows the first call as `pending`, not as ok or error

### Requirement: Private data is removed and checked before a freeze

Extraction SHALL replace pattern matches (keys, tokens, emails, home paths, billing IDs, random strings) and private
strings (the user's names, `.env` values of plain form excluded, and every line of `private-strings.txt`), matching
private strings as whole words only. `freeze` SHALL refuse to write unless `check` finds zero pattern hits and zero
private-string hits in every text.

#### Scenario: A private string remains
- **WHEN** a row still contains a string from `private-strings.txt`
- **THEN** `check` fails and `freeze` writes nothing

### Requirement: Versions are frozen, hashed and traceable

`freeze` SHALL write a new version folder that it never rewrites, with a manifest that holds the sha256 of each
question file, the sessions per split, the extract diagnostics and the provenance (git commit, dirty flag, code hash,
state budget, split cuts, number of private strings per source, and the sha256 of every labelling file). `load_split`
SHALL fail on a changed file and SHALL log every read.

#### Scenario: A frozen file was edited
- **WHEN** a question file differs from the hash in its manifest
- **THEN** `load_split` raises an error

#### Scenario: A result shows what it read
- **WHEN** a baseline or training run finishes
- **THEN** its output lists every split it read

### Requirement: External labels reach consensus by fixed rules

`minijev sessions consensus` SHALL decide each external label from recorded opinions by the rules in `consensus.py`:
round-1 agreement, review of disputes and of a seeded 10% audit, adjudication, tool facts, revision rounds and
deliberation. An item without consensus SHALL be marked contested and SHALL be left out of every split. A rerun on the
same opinions SHALL write the same label file.

#### Scenario: Reviewers split
- **WHEN** the two reviewers of an item give different labels
- **THEN** the item goes to two adjudicators, and if they also split it is contested

### Requirement: Test is read once, by the fixed comparison

Baselines and training SHALL read only train and val. `minijev sessions compare` SHALL be the only reader of test,
and SHALL compare an adapter with the previous-call baseline and the calibrated base model by paired bootstrap deltas
of test log loss, with results per project.

#### Scenario: Training asks for test
- **WHEN** a session training task requests the test split
- **THEN** it fails with an assertion
