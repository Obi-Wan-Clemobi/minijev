# Session data — method and data card

*How `minijev.sessions` turns Claude Code session logs into datasets, what each step guarantees and what it does not,
and what we measured on our own logs. The how-to is in [SESSIONS.md](SESSIONS.md). Evidence labels follow
RESEARCH.md: **Stated**, **Observed**, **Inferred**, **Measured**. This page holds no text from real rows.*

Terms: a **session** is one Claude Code conversation, stored as one `.jsonl` log. A **call** is one tool use in a
session. A **turn** is one message that the user typed, with the calls that follow it. A **decision point** is the
moment before one call, with the state that existed then. **train**, **val** and **test** are the three disjoint
parts of a version. A **version** (v1, v2, …) is a frozen set of rows with a manifest of hashes. A **decision
pattern** is a mapping from a request and the work so far to the next kind of step, that holds across projects. The
**domain** is what a project is about (board games, travel). A **fold** is one leave-one-project-out split: one
held-out project is test, and the other projects give train and val.

## 1. Summary of our data (version v6)

| | Value | Evidence |
|---|---|---|
| Source | Our own Claude Code logs, 2026-05-30 to 2026-09-25, 9 project folders | Measured |
| Log files used | 233 main-thread files (subagent transcripts left out); 232 with at least one turn | Measured |
| Turns / calls | 504 turns; 4156 calls: 4032 ok, 118 error, 5 rejected, 1 unknown status | Measured |
| Sessions per split | train 135 · val 54 · test 43 | Measured |
| Questions | `next_tool`, `bash_kind`, `will_fail` (labels from the log); `needs_approval` (consensus labels of Claude models) | |
| Where it is | `~/.minijev-private/sessions/v6/`, never in git | |
| Code | git commit 266780f plus uncommitted `src/minijev/sessions/` with sha256 f3573f9b… (v6 manifest) | Measured |

**What this data is not.** It is not a sample of "good agent behaviour". Three of the four questions teach a model
to predict what Claude did in our sessions, right or wrong (section 9, threat T1).

## 2. From log to session

The log format is not a documented interface (Inferred: we found no published schema). The parser uses these
fields: `type`, `message.content`, `message.id`, `isMeta`, `isSidechain`, `cwd`, `timestamp`, and `tool_use` /
`tool_result` blocks. `extract` counts what it could not use, so a format change shows as a jump in "unknown status"
or "sessions without turns".

| Entry | Kept? | Reason |
|---|---|---|
| A user message with real text | Yes: it starts a turn | |
| Command wrappers, system reminders, hook output, task notifications, interruption notes (`isMeta` or a known prefix) | No | The user did not type them |
| Subagent transcripts (`subagents/` folder, `isSidechain`) | No | Another agent's thread; its decisions have another context |
| Assistant `tool_use` blocks | Yes: one call each, deduplicated by id | A guard: in our logs no id repeats (Measured, 4156 ids) |
| `tool_result` blocks | Only the status: ok, error, or rejected | Results are the largest source of secrets (Inferred) |
| Assistant text | No | Text before a call can name the call, which would leak the label (Inferred) |

A call's **summary** is the Bash command without leading `cd X &&`, `NAME=value`, `export`, `set`, `exec`, `time` and
`(`; or the file path; or the input as JSON. It keeps 200 characters.

## 3. States

A state is the text the model sees. It holds the project name, the user's turn (at most 600 characters), and the last
4 calls of this turn before the decision, each with its status. For `will_fail` and `needs_approval` it also holds the
next call. v1 used 1500 characters and 8 calls; we cut the budget because training took 12 s per example (Measured,
v1, eager attention) against 6.7 s (Measured, v2, sdpa attention), and the mean state fell from 404 to 238 tokens (Measured, 400 sampled `next_tool` states).

Rules, and how each is checked:

| Rule | Why | Check |
|---|---|---|
| A state holds only what existed before the decision | No label leak | `test_state_holds_no_own_call` |
| A call from the same assistant message as the decision shows `pending`, not its status | Parallel calls: their results did not exist yet | `test_parallel_call_status_is_pending` |
| Only calls of the same turn are listed | Earlier turns are other tasks | By construction |

## 4. Labels

| Question | Type | Label | Source |
|---|---|---|---|
| `next_tool` | Choice, 8 options | The tool group of the call: bash, edit, read, web, browser, ask, mcp, other | Log |
| `bash_kind` | Choice, 5 options | A rule on the next Bash command's first word: inspect, run, git, remote, other | Log (a rule, not a judgement) |
| `will_fail` | Noul | The call's result has `is_error` | Log |
| `needs_approval` | Noul | "Should the agent ask before this call?" | Consensus of Claude labellers, reviewers and adjudicators (section 4.2) |

**What `is_error` covers** (Measured, all 123 error results including rejections): command exited non-zero 62;
other tool errors 35; harness permission denials 12; user rejections 5 (these are status `rejected`, not in
`will_fail`); tool input errors 4; hook blocks 3; timeouts 2. So `will_fail` = "the call did not succeed", not only
"the command was wrong".

### 4.1 `needs_approval`: how points were chosen and labelled

1. **Random sample.** 600 decision points drawn uniformly from all 4124 points (seed 0). This sample is the only one
   that estimates prevalence.
2. **Risk-pattern sample.** After the first 400 random labels (7 positives), we wrote a regex of risky command words
   (`labels/RISK_PATTERN-v1.txt`) and labelled all 185 matching points that were not yet labelled. The regex was
   written after we saw labels, so it can fit them (threat T6).
3. **Overlap.** 3 points were in both samples (a timing error in our pipeline). Each is kept once, as a random-sample
   point. 782 points in total.
4. **First labels (superseded).** One Claude Sonnet 5 agent per part, with rubric v1. A blind Claude Opus 5.5 check
   on all 39 positives and 100 negatives agreed on 97.1% (kappa 0.93; the set over-samples positives, so the kappa of
   the full population is lower, Inferred). Opus said yes on 4 more items, in two cases that rubric v1 did not settle:
   risky calls that the user asked for, and deletes of local app data. v5 replaces these labels with section 4.2.

### 4.2 `needs_approval`: consensus labels

The rules are code (`minijev/sessions/consensus.py`, run by `minijev sessions consensus`) and are tested in
`tests/test_sessions.py`. No person changed a label.

1. **Rubric v2.** It anchors the question on the user's own rule for agents: "for actions that are hard to reverse
   or outward-facing, confirm first unless durably authorized or explicitly told to proceed". Step 1 asks whether the
   call is risky. Step 2 asks whether the request explicitly asked for this action.
2. **Round 1.** Two labeller sets label all 782 points, blind to each other, each set in its own random order: 4
   Claude Sonnet 5 agents (set A) and 4 Claude Opus 5.5 agents (set B). An item is **agreed** if both give the same
   label and neither is unsure.
3. **Review.** Every disputed item, and a seeded random 10% of the agreed items, gets 2 reviewers (Sonnet and Opus).
   The 10% is an audit for errors that both sets share. Each reviewer sees the labellers' opinions but not the other
   reviewer. If the 2 agree and neither is unsure, their label is final. If reviewers overturn more than 5% of the
   audited items, every agreed item is reviewed.
4. **Adjudication.** Items that the reviewers split on go to 2 adjudicators (Sonnet and Opus), each blind to the
   other. Agreement without unsure is final; otherwise the item is **contested**.
5. **Tool facts.** Some states cannot show whether data has a copy. The travel-planner guide tools rewrite one YAML
   file under `data/guides/`. We first checked git as it is today (all 8 guide files tracked) and sent every
   guide-tool item that anyone labelled "yes" to adjudication with the fact "not risky". That check was wrong for
   older calls: for each call we then compared its timestamp with the file's first commit (`git log
   --diff-filter=A`). The file was committed before 90 of the 262 guide-tool calls and not yet committed before 172
   (Measured). The facts now carry the item ids they apply to (`facts.json`), and the sweep of step 8 re-decided every
   labelled guide-tool item.
6. **Rubric v2.1, revision round.** The adjudicators named two gaps: `rm -r` of a folder that git tracks, and standing
   permission ("handle the whole process"). v2.1 settles both from the rubric's own words ("data that has no copy";
   "durably authorized"). 2 new labellers re-decided, blind, the contested items and every decided item whose next call
   is `rm -r` (15 items).
7. **Deliberation.** For the items still contested, the 2 labellers saw each other's opinion. Items that stay split,
   or where a labeller stays unsure, remain **contested**, and they are left out of every split.
8. **v2.1 sweep.** Step 6 applied v2.1 only to 15 items. To apply it to every item it can change, 2 new blind labellers
   re-decided every decided item where any opinion had marked the call risky (the two clarifications change "risky" and
   "explicit"), plus every labelled guide-tool item with its corrected fact: 101 items. Splits went to a second
   deliberation. No labeller in steps 6 to 8 could consult another agent.

| Stage | Items in | Result (Measured) |
|---|---|---|
| Round 1 | 782 | 720 agreed, 62 disputed (46 of them only because a labeller was unsure). Set A vs set B: 97.95% agreement, kappa 0.83, positive agreement 2a/(2a+b+c) = 0.84 (41 both yes, 5 only A, 11 only B) |
| Audit of agreed items | 72 | No reviewer voted against round 1 on any audited item, sure or unsure |
| Review | 134 | 89 decided; 45 to adjudication (16 of them by the first tool fact) |
| Adjudication | 45 | 36 adjudicated, 9 contested |
| Revision (v2.1) | 15 | 11 revised, 4 contested |
| Deliberation | 4 | 1 deliberated, 3 contested |
| v2.1 sweep | 101 | 90 revised, 11 split; 15 labels changed from v5 (14 from no to yes, most of them guide edits before the file's first commit) |
| Second deliberation | 11 | 5 deliberated, 6 contested |
| **Final (v6)** | 782 | 773 labelled (48 yes), 9 contested |

21 final labels differ from the round-1 majority or break a round-1 tie (Measured). The 9 contested items are of two
kinds. In some, a labeller stays unsure because the state lacks what the decision needs: a reply that names a menu
option the state does not show; a script cut off at the 200-character limit; a browser click with no visible target;
a short reply to a request the state does not show. In the others, the two final labellers read the same request
differently (for example, whether "fix it for everyone affected" covers running a repair on the live database). Both
kinds are limits of the state design (section 3) or of the rubric, not of one labeller (Inferred).

A rerun of `minijev sessions consensus needs_approval` wrote a byte-identical label file, whose sha256 matches the v6
manifest (Measured).

| | Random sample | Risk-pattern sample |
|---|---|---|
| Labelled (not contested) | 596 | 177 |
| Yes | 21 (3.5%) | 27 |

14 of the 21 random-sample positives are guide edits made before the guide file's first commit (Measured). Their label
depends on a fact that the state does not show (threat T14).

A consensus of models is still the models' opinion under the rubric, not ground truth. Two model families (Sonnet
and Opus) reduce, but do not remove, shared blind spots: both are Claude, and Claude labels Claude's own actions
(threat T7). No human has labelled any point; the user chose model consensus instead.

## 5. Splits and versions

Inside each project, sessions go in start-time order into train until they hold 70% of the project's calls, then
val until 85%, then test. All rows from one session are in one split.

- **Why per project:** a global time split put whole projects, and their tools, in one split. In the first extract,
  222 of 236 edit calls were in test (Measured).
- **Why by call count:** a cut by session count gave val 151 and test 388 rows of 4156, because a few long sessions
  hold most calls (Measured).
- **What it does not control:** near-duplicate states across sessions (the same request typed twice) are not removed
  (threat T9).

A version is written once. `manifest.json` holds the sha256 of each question file, the session ids per split, the
extract diagnostics, and the provenance: git commit, a dirty flag, a hash of the module's code, the state budget, the
split cuts, the number of private strings per source (never the values), and the sha256 of the label files and
rubrics. `load_split` refuses a file whose hash changed, and it logs every read in `ACCESS`; baselines and training
write that log into their results.

| Version | Status | Why |
|---|---|---|
| v1 | Deleted | State text before the private-strings pass (it held personal details); budget 1500/8 |
| v2 | Deleted | `needs_approval` rows lack sample kind and unsure flag; no provenance |
| v3 | Deleted | The privacy audit found router details in 2 of 200 rows (section 6) |
| v4 | Superseded | `needs_approval` has the rubric v1 labels. Its other three question files are byte-identical to v5 (Measured: same sha256) |
| v5 | Superseded | Consensus before the v2.1 sweep and the corrected tool fact |
| v6 | Current | Final consensus labels. `next_tool`, `bash_kind` and `will_fail` are byte-identical to v4 and v5 (Measured) |

**Claims measured on deleted versions**, and what still supports them: the v1 baselines
(`results/baselines-v1-*.json`); the v3 privacy audit (`labels/privacy-audit-result.jsonl` and `-summary.json`); the
rubric-v1 agreement check (`labels/agreement-opus.jsonl`, `labels/needs_approval-rubric-v1.jsonl`). The training
times (12 s per example on v1 with eager attention, 6.7 s on v2 with sdpa) come from short timing runs whose output
was not kept: Measured, but not reproducible from a file.

**Test is untouched.** No baseline and no training run has read a test split of any version. For v4 runs, the
`splits_accessed` log in each result shows it (Measured). The v1 baseline runs came before that log; their code read
only train and val (Observed, in the code). `train_lora.py` asserts it for session tasks.

## 6. Privacy

Two independent scrub layers run before any row is written.

| Layer | What | Count in v4 |
|---|---|---|
| Patterns | Emails, key and token formats, bearer headers, `KEY=value`, cloud billing IDs, home paths, scratch paths, long hex and random strings | 13 patterns |
| Literals: user | Username, git name and email | 5 strings, 164 replacements |
| Literals: env | Values in `.env` files of every folder a session worked in (plain words, numbers and local URLs excluded) | 5 strings, 7 replacements |
| Literals: file | `private-strings.txt`, written after a review and an audit (below) | 53 strings, 399 replacements |

**What `check` proves, and what it does not.** `check` counts pattern and literal hits in every text of the
extract and of every row, and `freeze` refuses to write unless all counts are zero. For patterns this only shows
that the scrub ran: it tests the scrub's own patterns. For literals it shows that no known private string passed. It
cannot find a private string that nobody listed.

**The review that wrote `private-strings.txt`.** A Claude Sonnet 5 agent read all 504 turns and searched the call
summaries. It proposed 44 strings in 6 classes: people's names, Wi-Fi names, router and keychain details, LAN IPs in
router context, account and budget IDs, and fragments of the user's own name. We checked that each matches the data as
a whole word, and we added 1 street address that a travel guide is anchored on.

**Residual rate** (Measured). A second Claude Sonnet 5 agent, which had not seen the first review, read 200 random
v3 `next_tool` states in full. It found private data in 2 rows (1.0%; 95% Wilson interval 0.3% to 3.6%): router model
names and router admin paths in file names, from one home-network session. It found no names, addresses, account IDs
or credentials. We added those 4 strings and 4 stems of the router model to `private-strings.txt` and froze v4; none
of the 4 strings is in v4 (Measured). v4 has not had a fresh audit. The v3 rate is our best estimate of what a single
review leaves behind, and a fresh sample of v4 would give its own rate.

Adapters trained on this data can memorize parts of it. They go to `~/.minijev-private/adapters/`, never to `poc/`.

## 7. Baselines (val)

The `next_tool`, `bash_kind` and `will_fail` files of v4, v5 and v6 are byte-identical, so these v4 numbers are the
v6 numbers (Measured, Qwen2.5-0.5B-Instruct, bootstrap 95% intervals, 2000 resamples). Every run read only train and val
(Measured: `splits_accessed`).

| Question (val n) | Baseline | Log loss [95% CI] | Accuracy | ECE |
|---|---|---|---|---|
| `next_tool` (355) | prior | 1.860 [1.702, 2.013] | 0.465 | 0.256 |
| | previous call | **1.212 [1.058, 1.373]** | **0.623** | 0.148 |
| | zero-shot | 3.375 [3.216, 3.544] | 0.099 | 0.378 |
| | zero-shot + bias/temperature | 1.844 [1.660, 2.018] | 0.462 | 0.256 |
| `bash_kind` (165) | prior | 1.180 [1.052, 1.316] | 0.552 | 0.032 |
| | previous call | **1.001 [0.887, 1.129]** | **0.661** | 0.131 |
| | zero-shot | 4.034 [3.632, 4.434] | 0.267 | 0.714 |
| | zero-shot + bias/temperature | 1.186 [1.054, 1.330] | 0.552 | 0.054 |
| `will_fail` (351, 18 yes) | prior | 0.211 [0.129, 0.293] | 0.949 | 0.024 |
| | previous call | 0.222 [0.156, 0.297] | 0.954 | 0.039 |
| | zero-shot | 0.413 [0.382, 0.445] | 0.906 | 0.187 |
| | zero-shot + bias/temperature | **0.210 [0.131, 0.291]** | 0.949 | 0.021 |

What the table shows:
- The base model, zero-shot, is worse than the class prior on all three questions. With a fitted bias and temperature
  it only reaches the prior: it holds no usable signal for these questions before training.
- The previous call in the turn is the bar for `next_tool` and `bash_kind`. A fine-tune must beat it.
- Nothing beats the prior on `will_fail`: the intervals of all four baselines overlap.

### 7.1 The adapter comparison (fixed before the first result)

We wrote this rule, and the code that applies it (`minijev sessions compare`), while the first adapter trained and
before any adapter result existed:

1. The base model and the adapter use the same attention implementation as the baselines (eager).
2. Each model gets its own bias and temperature, fitted on val (as in E22). Val also chooses the epoch.
3. Test is read once, by `compare`, and its `splits_accessed` log shows it.
4. The main measure is test log loss. The adapter (with its bias and temperature) is compared with the previous-call
   baseline and with the calibrated base model by a paired bootstrap over the same test rows. A gain counts only if
   the 95% interval of the delta is below zero.
5. Results are also given per project (threat T4) and are labelled "predicts Claude" (threat T1).

**`needs_approval` has no val baseline.** Its v6 val split has 1 positive in 57 rows, so no metric on it means
anything. On the random sample (all splits, v6 labels), "yes if the risk regex matches" finds 6 of 21 positives with
precision 6/35 (Measured); it misses the guide edits of threat T14. This is the bar for a future model. The regex was written after the first labels
(threat T6).

## 7.2 Where the tokens go

**Interactive and SDK sessions.** The log records each session's entry point. 34 sessions come from the interactive
entry points (`cli`, `claude-desktop`): a person typed the turns. 198 sessions come from SDK entry points (`sdk-py` 126,
`sdk-cli` 72): the travel-planner app ran Claude Code as a program, with generated prompts (Measured). The SDK
sessions hold 247 of the 504 turns but only 486 of the 4156 calls and 2% of the cache reads. `extract` counts sessions
per entry point, and `--interactive-only` (on `freeze`, `stats` and `tokens`) leaves the SDK sessions out. Questions
about how the user works (area, preferences) use interactive sessions only.

`minijev sessions tokens --interactive-only` (`sessions/tokens.py`) sums the usage that every assistant message
records, per kind of work (`sessions/patterns.py`) and per chain pattern. A message's tokens are split evenly over its
calls; a message without calls counts as "answer". A **chain pattern** is the sequence of the kinds of work of a
turn's calls, with repeats merged. Measured on the 34 interactive sessions (257 turns, 3764 assistant messages):

| Kind of work | Calls | Cache read | Cache write | Output |
|---|---|---|---|---|
| inspect | 1723 | 368.4 M | 8.23 M | 1.03 M |
| run | 1081 | 292.4 M | 3.12 M | 1.11 M |
| answer (no call) | – | 84.9 M | 0.69 M | 0.23 M |
| change | 316 | 71.3 M | 0.79 M | 0.44 M |
| browse | 193 | 41.2 M | 0.32 M | 0.05 M |
| remote | 139 | 27.5 M | 0.18 M | 0.06 M |
| orchestrate | 103 | 18.8 M | 0.19 M | 0.06 M |
| research | 53 | 7.7 M | 0.07 M | 0.03 M |
| publish | 30 | 6.4 M | 0.15 M | 0.01 M |
| ask | 32 | 5.9 M | 0.05 M | 0.04 M |
| **Total** | 3670 | **924.4 M** | **13.8 M** | **3.08 M** |

Fresh (uncached) input is 7.8 thousand tokens in total.

- **Round trips drive the volume.** Every assistant message reads the whole context again from the cache: the median
  message reads 196 thousand cached tokens. A turn has a median of 6 messages, and the longest 10% of turns have 38 or
  more (maximum 193). The tokens of a turn grow with its number of messages more than with its answer length.
- **Inspection is the largest share:** 40% of cache reads and 60% of cache writes. 1587 of the 1723 inspect calls use
  Bash (`cat`, `grep`, `sed` and so on) rather than Read, Grep or Glob.
- **Calls are rarely batched:** 186 of the 3373 messages with calls make 2 or more calls. The most expensive turns
  are long loops of inspect and run, one call per message.

Inferred: the decisions worth handing to minijev are the ones that cut messages: batching independent inspect calls
into one message, and stopping an inspect-run loop early. The price of each token type differs, and this page does not
convert the counts into cost.

## 8. How to check the claims on this page

| Claim | Command or file |
|---|---|
| Counts per split and label | `minijev sessions stats` |
| Zero pattern and literal hits | `minijev sessions check` |
| A version is unchanged | `load_split` (fails on a changed file); hashes in `manifest.json` |
| Provenance | `manifest.json` → `provenance` |
| No state leaks its label | `pytest tests/test_sessions.py` |
| Which splits a result read | `splits_accessed` in each results file and `train_log.json` |
| The consensus labels follow from the recorded opinions | Rerun `minijev sessions consensus needs_approval`; it must write the same `needs_approval.jsonl` (the audit sample is seeded). Every input file's sha256 is in the manifest's `provenance.label_files` |

## 9. Threats to validity

| # | Threat | Effect | What we do |
|---|---|---|---|
| T1 | `next_tool` and `bash_kind` imitate Claude, not a correct policy | A good score means "predicts Claude", not "makes good choices" | Say so in every result |
| T2 | One user, 4 months | Whether results transfer to other people or other kinds of work is unknown | The module exists so that others can run it on their data |
| T3 | Claude Code and model versions changed during the logs | Behaviour, and even the log format, drift over time | Time split per project; diagnostics in `extract` |
| T4 | Projects are unequal: travel-planner and board-game-event-planner hold 63% of calls (Measured) | Pooled numbers mostly describe two projects | Report per project when it matters |
| T5 | Test is sparse in places: `next_tool` test has 5 edit, 2 read and 0 web rows; `will_fail` test has 11 positives; `needs_approval` val has 1 positive and test 8 | Per-class numbers and small-class claims are not supported | Report overall numbers with bootstrap intervals only; do not train `needs_approval` until val has enough positives |
| T6 | The risk-pattern regex was written after seeing labels | The labelled set over-represents what the regex finds | Report prevalence from the random sample only; the regex is a baseline to beat |
| T7 | Claude labels Claude's own actions | Shared blind spots | Two model families, blind rounds, audits and adjudication (section 4.2). No human labels, by the user's choice |
| T8 | `is_error` mixes command failures with harness denials and hook blocks | `will_fail` is "did not succeed" | Stated in section 4 |
| T9 | Near-duplicate states across sessions are not removed | A small train-to-test overlap can inflate scores | Not measured yet |
| T10 | Residual private data after the scrub | Privacy risk for the data and any adapter | Measured residual rate: 1.0% of rows in v3, fixed in v4; v4 not re-audited (section 6); all data outside git |
| T11 | The state can lack what a decision needs (Claude's text is left out; call summaries are cut at 200 characters) | Some labels cannot be decided from the state | 3 contested items left out (section 4.2) |
| T12 | Rules 4 (second adjudicator), 6, 7 and 8 and rubric v2.1 were added after we saw round-1 and review results | The process was tuned on the data it labels, as in T6 | Every rule is code with a test; every stage's files are hashed in the manifest; the counts per stage are reported |
| T13 | Agents are not strictly independent: one deliberator consulted an advisor model before answering | "Two model families" overstates independence | Later rounds forbid consulting other agents; still both families are Claude (T7) |
| T14 | Some labels depend on a fact the state does not show: whether git held a copy (the guide-file commit time) | 14 of 21 random-sample positives cannot be learned from the state; they add label noise for a readout | Open. Either show the fact in the state (a "tracked in git" marker in the call summary), or leave these items out of training and report with and without them |
