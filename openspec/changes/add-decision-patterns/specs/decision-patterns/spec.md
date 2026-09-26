## ADDED Requirements

### Requirement: Kind of work comes from a fixed table

The `work_kind` question SHALL label each call by a fixed table from the call (tool, Bash kind, git subcommand, and
the verb of an MCP tool name) to one of: inspect, change, run, remote, research, browse, publish, ask, orchestrate.
The table (`minijev/sessions/patterns.py`) SHALL be committed before any fold result is computed, and a change to it
SHALL give a new version.

#### Scenario: A Bash test run
- **WHEN** the next call is a Bash command of kind "run"
- **THEN** its `work_kind` label is "run"

### Requirement: Tool choice is asked given the kind of work

For each kind of work that more than one tool can do, a `tool_<kind>` question SHALL be a Choice whose options are
the tools that can do that kind (`TOOLS_BY_KIND`). Its rows SHALL be the calls of that kind, and its label SHALL be
the tool id of the call. A tool id SHALL name every MCP tool "mcp" and every browser tool "browser", so that no label
names a domain.

#### Scenario: Options follow the kind
- **WHEN** a call's kind of work is "research"
- **THEN** it is a row of `tool_research`, whose options are WebSearch, WebFetch and mcp

#### Scenario: A domain-specific MCP tool
- **WHEN** the call is `mcp__travel__add_guide_entry`
- **THEN** its tool id is "mcp"

### Requirement: Waste is labelled from the log

The `waste` question SHALL label a call as waste if it failed and the next call in the same turn uses the same tool;
if it reads a file already read in the turn with no edit between; or if the user rejected it.

#### Scenario: A retry after a failure
- **WHEN** a Bash call fails and the next call in the turn is also Bash
- **THEN** the failed call is labelled waste

### Requirement: Token use is measured per kind of work and per chain

`minijev sessions tokens` SHALL report, per kind of work and per chain pattern (the sequence of tool groups in a
turn), the fresh input, cache read, cache write and output tokens of the assistant messages, with cache reads apart.

#### Scenario: A breakdown
- **WHEN** the user runs `minijev sessions tokens`
- **THEN** it prints and saves the four token counts per kind of work and for the most common chain patterns

### Requirement: Results are reported per fold

Evaluation on a leave-one-project-out version SHALL report each fold on its own, with its train label counts, and
SHALL NOT pool folds. A class with fewer than 20 train rows in a fold SHALL be reported apart.

#### Scenario: A class missing from a fold's train split
- **WHEN** "mcp" has fewer than 20 train rows in the travel-planner fold
- **THEN** the report lists "mcp" as unsupported for that fold

### Requirement: Hand-off is evaluated as a risk-coverage curve

For each question, the evaluation SHALL report accuracy on the kept decisions against the share handed off, over the
confidence thresholds, and SHALL choose the threshold on val.

#### Scenario: A threshold
- **WHEN** the val-chosen threshold keeps 70% of decisions
- **THEN** the report gives test accuracy on the kept 70% and the share handed off
