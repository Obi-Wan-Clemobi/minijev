# Session data: train minijev on your own Claude Code sessions

`minijev.sessions` turns your Claude Code session logs into minijev datasets. Each row is a state (the text before a
decision) and a label (what happened, or what a labeller said). You can then measure baselines and fine-tune on
your own work. The method, its checks, our measurements and the threats to validity are in
[SESSIONS_METHOD.md](SESSIONS_METHOD.md).

A **session** is one Claude Code conversation. Claude Code stores it as one `.jsonl` log file in
`~/.claude/projects/<folder>/`. A **call** is one tool use in a session (for example, a Bash command or a file edit).
A **turn** is one message that you typed, with the calls that follow it.

## Privacy first

The logs hold everything you typed and every command that Claude ran. They can hold names, addresses, network names,
tokens and file contents. The module protects this data in four ways:

1. **Everything stays under one private folder.** The folder is `~/.minijev-private`, or `MINIJEV_PRIVATE`. The
   module refuses a folder inside a git work tree unless you pass `--allow-git`.
2. **A state never holds a tool result or Claude's text.** Tool results are the largest source of secrets.
3. **Scrub with two independent layers.**
   - Patterns: key formats, emails, home paths, cloud billing IDs, and long random strings.
   - Literals: your username, your git name and email, every value in a `.env` file of a folder that a session
     worked in, and every line of `~/.minijev-private/private-strings.txt`.
4. **`check` must find zero hits before `freeze` writes anything.**

Patterns do not find names, network names or street addresses. Before you train, review a sample and add what you
find to `private-strings.txt`:

```bash
minijev sessions sample next_tool -n 20
```

An adapter trained on this data can memorize parts of it. Keep adapters under the private folder too, and do not
publish them.

## Quick start

Run these commands from `poc/` (or anywhere that has the `minijev` package installed):

```bash
uv run minijev sessions sync       # copy ~/.claude/projects/**/*.jsonl to ~/.minijev-private/sessions-raw/
uv run minijev sessions extract    # parse and scrub; prints what it could not use
uv run minijev sessions sample next_tool -n 20   # review; add private strings; extract again
uv run minijev sessions check      # zero hits, or exit 1
uv run minijev sessions stats      # rows per question, split and label
uv run minijev sessions freeze     # writes ~/.minijev-private/sessions/v1/ (never rewritten)
uv run minijev sessions baselines v1             # prior, previous-call and zero-shot readout, on val
```

`baselines` reads the base model once per row. On a 6-thread Intel CPU with Qwen2.5-0.5B, a readout took 1.06 s
for a state of 186 tokens on average (Measured). Pass `--no-readout` to count the baselines without a model.

`extract` prints diagnostics. The log format has no public documentation, and it can change between Claude Code
versions. A large `calls with status unknown` count or `sessions without turns` count means that the format changed.

## Keep your logs

Claude Code deletes logs after `cleanupPeriodDays` (default: 30 days). To keep more:

1. Set `"cleanupPeriodDays": 365` in `~/.claude/settings.json`.
2. Run `minijev sessions sync` often. `sync` never deletes, so the archive keeps logs that Claude Code removes. To
   run it after every session, add a `SessionEnd` hook to `~/.claude/settings.json`:

```json
"SessionEnd": [{"hooks": [{"type": "command", "command": "rsync -a --include='*/' --include='*.jsonl' --exclude='*' ~/.claude/projects/ ~/.minijev-private/sessions-raw/", "timeout": 60}]}]
```

## The questions

| Question | Type | State | Label | Label source |
|---|---|---|---|---|
| `next_tool` | Choice: bash, edit, read, web, browser, ask, mcp, other | The turn and its calls so far | The tool group of the next call | log |
| `bash_kind` | Choice: inspect, run, git, remote, other | The same | The kind of the next Bash command | log |
| `will_fail` | Noul | The same, and the next call | The call returned an error | log |
| `needs_approval` | Noul | The same, and the next call | A labeller's answer | external |

A **log label** is what happened in the session. An **external label** comes from a person or a model, in
`~/.minijev-private/sessions/labels/<question>.jsonl`, one `{"id": ..., "y": 0 or 1, "by": ...}` per line. Only
labelled points become rows. To get points to label:

```bash
uv run minijev sessions points needs_approval -n 200 > to_label.jsonl
```

To label with several labellers and reach consensus, put each labeller's file in
`labels/<question>/round1/` (each item labelled by at least 2, blind to each other), then run
`minijev sessions consensus <question>`. It writes the next queue (review, then adjudication) until every item is
decided, and then the consensus file. The rules are in `minijev/sessions/consensus.py`; our run is in
[SESSIONS_METHOD.md](SESSIONS_METHOD.md) §4.2.

A label from a model teaches that model's opinion, not ground truth. Set `"by"` to the labeller, and do not mix
model labels into a test split that you report as ground truth.

The questions form a ladder: first predict what the agent does (`next_tool`, `bash_kind`), then the outcome
(`will_fail`), then the decision before the agent acts (`needs_approval`). You can chain these questions into a flow
on the State machine page of the web app.

### Rules for a state

- A state holds only what existed before the decision.
- A call from the same assistant message as the decision is a parallel call. Its status shows as `pending`, because
  its result did not exist yet.
- The user turn keeps at most 600 characters, each call summary 200 characters, and the list the last 4 calls of
  the turn.

### Add your own question

Write a Python file that registers a question, and pass it with `--questions`:

```python
# my_questions.py
from minijev.sessions import Question, register, state

def rows(s):
    for k, c in enumerate(s["calls"]):
        if c["turn"] is not None and c["tool"] == "Bash":
            turn = s["turns"][c["turn"]]
            yield f"{s['session']}:{k}", state(s, turn, s["calls"][:k], c, msg=c["msg"]), int(c["summary"].startswith("pytest"))

register(Question("runs_tests", "noul", ["no", "yes"], {"instructions": "Is this call a test run?"}, rows))
```

```bash
uv run minijev sessions --questions my_questions.py stats
```

A question's `rows` function receives one parsed session:
`{session, project, start, cwd, turns: [{i, text, ts}], calls: [{msg, tool, group, summary, kind, status, turn, ts}]}`.
It yields `(row id, state text, label)`. For a Noul, label 0 is "no" and label 1 is "yes".

## Splits

Splits are by session start time inside each project. In each project, the oldest sessions are train until they
hold 70% of the project's calls. The next sessions are val until 85%, and the rest are test. All rows from one
session are in one split, because rows in one session are correlated.

A frozen version (`v1`, `v2`, …) is never rewritten. `load_split` fails if a file changed after the freeze. New
sessions go into a new version.

```python
from minijev.sessions import load_split
train = load_split("next_tool", "train", "v1")   # [{"id", "session", "project", "text", "y", "split"}, ...]
```

## Baselines

Every fine-tune result must beat all four baselines on the same split:

| Baseline | What it is |
|---|---|
| prior | The train label frequencies |
| previous call | P(label given the previous call in the turn), counted on train. Calls in a turn are strongly correlated, so this bar is the one that counts. |
| zero-shot | The minijev readout of the base model with the question's prompt |
| zero-shot + bias/temperature | The readout with a bias and temperature fitted on 600 train rows. It changes no weight. |

The results go to `~/.minijev-private/sessions/results/`.

## Limits

- One person's sessions are a small dataset. Our logs held 505 turns, 4156 calls and 118 errors (Measured,
  2026-05-30 to 2026-09-25). See [SESSIONS_METHOD.md](SESSIONS_METHOD.md) §9 for the threats to validity.
- The mix of tools depends on the project. Report results per project when one project dominates.
- The Bash kind rule is a list of command names in `minijev/sessions/logs.py`. Commands that are not in the list
  count as "other".
