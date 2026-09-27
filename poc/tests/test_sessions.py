"""minijev.sessions on synthetic logs: parsing, no future results in a state, scrub, labels, freeze."""

import json
import re
from pathlib import Path

import pytest

from minijev.sessions import Paths, Question, check, extract, freeze, load_split, register, rows, state, stats
from minijev.sessions import dataset, logs, questions, scrub


def _use(msg, uid, name, command):
    return {"type": "assistant", "timestamp": "2026-01-01T00:00:01Z", "message": {"id": msg, "content": [
        {"type": "tool_use", "id": uid, "name": name, "input": {"command": command}}]}}


def _result(uid, error):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": uid, "is_error": error, "content": "x"}]}}


def _entries(cwd="/work/app", day="01"):
    return [
        {"type": "user", "cwd": cwd, "message": {"content": "<command-name>/clear</command-name>"}},
        {"type": "user", "cwd": cwd, "timestamp": f"2026-01-{day}T00:00:00Z", "message": {"content": "fix the test"}},
        _use("m1", "a", "Bash", "cd /x && ls"),   # two parallel calls: one message, two entries
        _use("m1", "b", "Bash", "pytest"),
        _result("a", False),
        _result("b", True),
        _use("m2", "c", "Edit", "unused"),
    ]


def _write(path: Path, entries) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return path


def test_meta_entries_are_not_turns(tmp_path):
    s = logs.parse(_write(tmp_path / "s.jsonl", _entries()))
    assert [t["text"] for t in s["turns"]] == ["fix the test"]
    assert [c["status"] for c in s["calls"]] == ["ok", "error", "unknown"]
    assert s["calls"][0]["summary"] == "ls" and s["calls"][0]["kind"] == "inspect" and s["calls"][1]["kind"] == "run"
    assert s["project"] == "app"


def test_parallel_call_status_is_pending(tmp_path):
    s = logs.parse(_write(tmp_path / "s.jsonl", _entries()))
    turn, calls = s["turns"][0], s["calls"]
    second = state(s, turn, calls[:1], calls[1], msg=calls[1]["msg"])
    assert "[Bash] ls -> pending" in second and "error" not in second
    third = state(s, turn, calls[:2], msg=calls[2]["msg"])
    assert "[Bash] ls -> ok" in third and "[Bash] pytest -> error" in third


def test_state_holds_no_own_call(tmp_path):
    s = logs.parse(_write(tmp_path / "s.jsonl", _entries()))
    for q in questions.QUESTIONS.values():
        for rid, text, _ in q.rows(s):
            if rid.split(":")[1].startswith("t"):
                continue   # a per-turn question (area): its state has no calls
            k = int(rid.split(":")[1])
            listed = [l for l in text.split("\nNext call:")[0].split("\n") if l.startswith("- [")]
            assert len(listed) == sum(c["turn"] == s["calls"][k]["turn"] for c in s["calls"][:k])


def test_project_name():
    home = Path.home()
    assert logs.project_name(str(home / "Code" / "app")) == "app"
    assert logs.project_name(str(home / "Code" / "app" / "web")) == "app/web"
    assert logs.project_name(str(home)) == "home" and logs.project_name("/srv/tool") == "tool"


def test_scrub_removes_secrets():
    text = scrub.scrub(f"key sk-{'a' * 30} API_KEY=abcdefgh123 me@example.com {scrub.HOME}/x ABC123-DEF456-789ABC "
                       "/tmp/claude-501/-a-b/c/scratchpad/q.sql")
    assert scrub.hits(text) == [] and "~/x" in text and "<scratch>/q.sql" in text


def test_literals_match_whole_words_only():
    pattern = scrub.literal_pattern(frozenset({"dev", "Wanda"}))
    assert scrub.scrub("dev ran device check for Wanda", pattern) == "<private> ran device check for <private>"


def test_env_values_are_literals(tmp_path):
    (tmp_path / ".env").write_text("TOKEN=abcdefgh12345\nSHORT=abc\n# X=commentedout123\nNODE_ENV=development\n"
                                   "API=http://localhost:3000/api\nPORT=80800000\n")
    assert scrub.env_values({str(tmp_path)}) == {"abcdefgh12345"}


def test_git_guard(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit):
        Paths(tmp_path / "private")
    assert Paths(tmp_path / "private", allow_git=True).root == tmp_path / "private"


def _root(tmp_path) -> Paths:
    paths = Paths(tmp_path / "root")
    for i, day in enumerate(["01", "02", "03", "04"]):
        _write(paths.raw / "proj" / f"s{i}.jsonl", _entries(day=day))
    (paths.raw / "proj" / "subagents").mkdir()
    _write(paths.raw / "proj" / "subagents" / "x.jsonl", _entries())   # left out
    paths.private_strings.write_text("# private\nfix\n")
    return paths


def test_extract_check_freeze(tmp_path):
    paths = _root(tmp_path)
    diag = extract(paths)
    assert diag["sessions written"] == 4 and diag["files"] == 4
    assert all(t["text"] == "<private> the test" for s in dataset.sessions(paths) for t in s["turns"])
    assert check(paths, quiet=True)
    split = dataset.split_of(dataset.sessions(paths))
    assert [split[f"s{i}"] for i in range(4)] == ["train", "train", "train", "val"]   # 9 of 12 calls seen < 85%
    assert sum(stats(paths)["next_tool"]["train"].values()) == 9
    freeze(paths, "v1")
    assert len(load_split("will_fail", "train", "v1", paths)) == 6
    import shutil
    shutil.rmtree(paths.out / "v1")
    assert dataset.next_version(paths) == "v2"   # a deleted version's name is never reused
    freeze(paths)
    with pytest.raises(SystemExit):
        freeze(paths, "v2")
    (paths.out / "v2" / "will_fail.jsonl").write_text("{}")
    with pytest.raises(ValueError):
        load_split("will_fail", "train", "v2", paths)


def test_external_labels_join(tmp_path):
    paths = _root(tmp_path)
    extract(paths)
    q = questions.QUESTIONS["needs_approval"]
    assert rows(paths, q) == []
    todo = dataset.points(paths, q, 100)
    assert len(todo) == 12 and all("\nNext call:\n" in p["text"] for p in todo)
    paths.labels.mkdir()
    (paths.labels / "needs_approval.jsonl").write_text(json.dumps({"id": "s0:1", "y": 1, "by": "person"}))
    [r] = rows(paths, q)
    assert (r["id"], r["y"], r["by"]) == ("s0:1", 1, "person") and "[Bash] pytest" in r["text"]
    (paths.labels / "needs_approval.jsonl").write_text(json.dumps({"id": "s0:1", "y": 1, "status": "contested"}))
    assert rows(paths, q) == []   # a contested label is left out
    assert len(dataset.points(paths, q, 100)) == 11


def test_custom_question(tmp_path):
    def runs_tests(s):
        for k, c in enumerate(s["calls"]):
            if c["tool"] == "Bash":
                yield f"{s['session']}:{k}", state(s, s["turns"][c["turn"]], s["calls"][:k], c, c["msg"]), \
                    int(bool(re.match("pytest", c["summary"])))
    register(Question("runs_tests", "noul", ["no", "yes"], {"instructions": "Is this call a test run?"}, runs_tests))
    try:
        paths = _root(tmp_path)
        extract(paths)
        assert [r["y"] for r in rows(paths, questions.QUESTIONS["runs_tests"])][:2] == [0, 1]
    finally:
        del questions.QUESTIONS["runs_tests"]


def test_consensus_rules(tmp_path):
    from minijev.sessions import consensus

    def put(stage, name, rows):
        (tmp_path / stage).mkdir(exist_ok=True)
        (tmp_path / stage / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = [f"s:{k}" for k in range(20)]
    put("round1", "A0", [{"id": i, "y": 0, "by": "a"} for i in ids])
    put("round1", "B0", [{"id": i, "y": int(i == "s:0"), "by": "b", "unsure": i == "s:1"} for i in ids])
    r = consensus.decide(tmp_path, {})
    audit = set(r["queues"]["review"]) - {"s:0", "s:1"}
    assert r["counts"]["disputed_round1"] == 2 and len(audit) == 2 and not r["queues"]["adjudicate"]
    assert all(r["final"][i]["status"] == "agreed" for i in r["final"])
    # reviewers: agree on s:0 (yes), split on s:1, confirm the audited items
    rev = lambda name, y1: [{"id": "s:0", "y": 1, "by": name}, {"id": "s:1", "y": y1, "by": name}] + \
        [{"id": i, "y": 0, "by": name} for i in audit]
    put("review", "R0", rev("r0", 0)); put("review", "R1", rev("r1", 1))
    r = consensus.decide(tmp_path, {})
    assert r["final"]["s:0"]["y"] == 1 and r["final"]["s:0"]["status"] == "reviewed"
    assert r["queues"]["adjudicate"] == ["s:1"] and not r["counts"]["second_round"]
    put("adjudicate", "J", [{"id": "s:1", "y": 1, "by": "j"}])
    put("adjudicate", "K", [{"id": "s:1", "y": 0, "by": "k"}])   # adjudicators split: contested, majority of all
    r = consensus.decide(tmp_path, {})
    assert r["final"]["s:1"]["status"] == "contested" and not any(r["queues"].values())
    assert r["final"]["s:1"]["y"] == 0   # opinions on s:1: 0, 0 (round 1), 0, 1 (review), 1, 0 (adjudication)
    assert r["counts"]["final_labels"] == {0: 18, 1: 1} and r["counts"]["changed_from_round1_majority"] == 1   # s:0 was a tie


def test_consensus_fact_rule(tmp_path):
    from minijev.sessions import consensus
    (tmp_path / "round1").mkdir()
    for name, ys in (("A0", {"s:0": 1, "s:1": 0}), ("B0", {"s:0": 1, "s:1": 0})):
        (tmp_path / "round1" / f"{name}.jsonl").write_text("".join(json.dumps({"id": i, "y": y, "by": name}) + "\n"
                                                                   for i, y in ys.items()))
    (tmp_path / "facts.json").write_text(json.dumps([{"pattern": r"^\[guide\]", "fact": "git-tracked"}]))
    texts = {"s:0": "x\nNext call:\n[guide] remove", "s:1": "x\nNext call:\n[guide] remove"}
    r = consensus.decide(tmp_path, {}, texts)
    assert r["queues"]["adjudicate"] == ["s:0"]           # a yes on a fact item goes to the adjudicator
    assert "s:1" not in r["queues"]["adjudicate"]         # all-no items already fit the fact
    (tmp_path / "adjudicate").mkdir()
    for name in ("J", "K"):
        (tmp_path / "adjudicate" / f"{name}.jsonl").write_text(json.dumps({"id": "s:0", "y": 0, "by": name}) + "\n")
    r = consensus.decide(tmp_path, {}, texts)
    assert r["final"]["s:0"] == r["final"]["s:0"] | {"y": 0, "status": "adjudicated"}


def test_training_cannot_read_test():
    import train_lora
    with pytest.raises(AssertionError):
        train_lora.sessions_split("sessions-next_tool", "test")


def _lopo_root(tmp_path, monkeypatch):
    """Three project groups: app (4 sessions, with a sub-folder app/web), tool (4 sessions), tiny (1 session)."""
    monkeypatch.setattr(dataset, "MIN_FOLD_CALLS", 10)
    paths = Paths(tmp_path / "root")
    code = Path.home() / "Code"   # project names are relative to the home folder (logs.project_name)
    for i, (cwd, day) in enumerate([("app", "01"), ("app", "02"), ("app/web", "03"), ("app", "04"), ("tool", "01"),
                                    ("tool", "02"), ("tool", "03"), ("tool", "04"), ("tiny", "05")]):
        _write(paths.raw / "p" / f"s{i}.jsonl", _entries(cwd=str(code / cwd), day=day))
    extract(paths)
    return paths


def test_leave_one_project_out(tmp_path, monkeypatch):
    paths = _lopo_root(tmp_path, monkeypatch)
    all_sessions = dataset.sessions(paths)
    assert dataset.folds(all_sessions) == ["app", "tool"]            # tiny has 3 calls < 10: never a fold
    split = dataset.split_of(all_sessions, "leave-one-project-out", "app")
    assert {split[f"s{i}"] for i in range(4)} == {"test"}            # app/web belongs to app
    assert [split[f"s{i}"] for i in range(4, 8)] == ["train", "train", "train", "val"]   # newest 15% of tool is val
    assert split["s8"] == "train"                                    # a small project is always train
    with pytest.raises(ValueError):
        dataset.split_of(all_sessions, "leave-one-project-out", "tiny")


def test_lopo_freeze_and_load(tmp_path, monkeypatch):
    paths = _lopo_root(tmp_path, monkeypatch)
    freeze(paths, "v1", strategy="leave-one-project-out", show_project=False)
    m = dataset.manifest("v1", paths)
    assert m["split"] == {"strategy": "leave-one-project-out", "folds": ["app", "tool"], "show_project": False,
                          "interactive_only": False}
    test = load_split("next_tool", "test", "v1", paths, fold="tool")
    assert {r["project"] for r in test} == {"tool"} and all("Project:" not in r["text"] for r in test)
    assert "tool" not in {r["project"] for r in load_split("next_tool", "train", "v1", paths, fold="tool")}
    with pytest.raises(ValueError):
        load_split("next_tool", "test", "v1", paths)                 # a fold is required
    assert dataset.ACCESS[-1]["fold"] == "tool"


def test_work_kind_table():
    from minijev.sessions.patterns import tool_id, work_kind

    def call(tool, summary="", kind=None):
        return {"tool": tool, "summary": summary, "kind": kind}
    assert work_kind(call("Read", "/x.py")) == "inspect" and work_kind(call("Edit", "/x.py")) == "change"
    assert work_kind(call("Bash", "git status", "git")) == "inspect"
    assert work_kind(call("Bash", "git commit -m x", "git")) == "change"
    assert work_kind(call("Bash", "git push origin main", "git")) == "publish"
    assert work_kind(call("Bash", "pytest", "run")) == "run" and work_kind(call("Bash", "ssh host ls", "remote")) == "remote"
    assert work_kind(call("mcp__app__remove_entry")) == "change" and work_kind(call("mcp__app__lookup_place")) == "inspect"
    assert work_kind(call("mcp__mail__send_message")) == "publish"
    assert work_kind(call("mcp__claude-in-chrome__computer")) == "browse"
    assert work_kind(call("mcp__claude-in-chrome__get_page_text")) == "inspect"
    assert work_kind(call("AskUserQuestion")) == "ask" and work_kind(call("ToolSearch")) == "orchestrate"
    assert tool_id("mcp__travel__add_entry") == "mcp" and tool_id("NotebookEdit") == "Edit"


def test_waste_rules(tmp_path):
    from minijev.sessions import QUESTIONS
    use = lambda msg, uid, name, inp: {"type": "assistant", "message": {"id": msg, "content": [
        {"type": "tool_use", "id": uid, "name": name, "input": inp}]}}
    entries = [
        {"type": "user", "cwd": "/w/app", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "go"}},
        use("m1", "a", "Bash", {"command": "pytest"}), _result("a", True),       # fails, then Bash again: waste
        use("m2", "b", "Bash", {"command": "pytest -x"}), _result("b", False),
        use("m3", "c", "Read", {"file_path": "/f"}), _result("c", False),
        use("m4", "d", "Read", {"file_path": "/f"}), _result("d", False),       # read again, no edit between: waste
        use("m5", "e", "Edit", {"file_path": "/f"}), _result("e", False),
        use("m6", "f", "Read", {"file_path": "/f"}), _result("f", False),       # read after an edit: not waste
    ]
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    assert [y for _, _, y in QUESTIONS["waste"].rows(s)] == [1, 0, 0, 1, 0, 0]


def test_token_breakdown(tmp_path):
    from minijev.sessions.tokens import breakdown, chain
    use = lambda msg, uid, name, inp, cr: {"type": "assistant", "message": {"id": msg, "usage": {
        "input_tokens": 1, "cache_read_input_tokens": cr, "cache_creation_input_tokens": 0, "output_tokens": 10},
        "content": [{"type": "tool_use", "id": uid, "name": name, "input": inp}]}}
    entries = [
        {"type": "user", "cwd": "/w/app", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "go"}},
        use("m1", "a", "Read", {"file_path": "/f"}, 100),
        use("m1", "b", "Read", {"file_path": "/g"}, 100),    # the same message streamed again: counted once
        use("m2", "c", "Bash", {"command": "pytest"}, 300),
    ]
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    out = breakdown([s])
    assert out["by_kind"]["inspect"]["cache_read"] == 100 and out["by_kind"]["run"]["cache_read"] == 300
    assert out["total"]["output"] == 20 and list(out["by_chain"]) == ["inspect > run"]
    assert chain(["inspect", "inspect", "run", "inspect"]) == "inspect > run > inspect"


def test_risk_coverage():
    from minijev.sessions.evaluate import choose_threshold, confidence, risk_coverage
    assert confidence([0.5, 0.5]) == 0 and confidence([1.0, 0.0]) == 1
    probs = [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.55, 0.45]]
    curve = risk_coverage(probs, [0, 0, 1, 0], steps=10)
    assert curve[0] == {"threshold": 0.0, "kept": 4, "coverage": 1.0, "accuracy": 0.75}
    chosen = choose_threshold(curve, 0.9, min_kept=2)
    assert chosen["coverage"] == 0.5 and chosen["accuracy"] == 1.0   # keep the two confident ones, hand off the rest
    assert choose_threshold(curve, 0.9, min_kept=3) is None          # 2 kept rows are too few to trust


def test_sdk_sessions_are_marked_and_can_be_left_out(tmp_path):
    entries = _entries()
    entries[1]["entrypoint"] = "sdk-py"
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    assert s["entrypoint"] == "sdk-py" and not s["interactive"]
    assert dataset.select([s], interactive_only=True) == [] and dataset.select([s], False) == [s]


def test_kappa_multiclass(tmp_path):
    from minijev.sessions import consensus
    (tmp_path / "round1").mkdir()
    a = [0, 1, 2, 2, 1, 0]; b = [0, 1, 2, 1, 1, 0]
    for name, ys in (("A", a), ("B", b)):
        (tmp_path / "round1" / f"{name}.jsonl").write_text("".join(json.dumps({"id": f"s:{i}", "y": y}) + "\n"
                                                                   for i, y in enumerate(ys)))
    k = consensus.pairwise_kappa(tmp_path)["A-B"]
    pe = (2 * 2 + 2 * 3 + 2 * 1) / 36   # class rates: A 2,2,2; B 2,3,1
    assert abs(k["kappa"] - (5 / 6 - pe) / (1 - pe)) < 1e-9


def test_area_rows_interactive_only_with_context(tmp_path):
    from minijev.sessions import QUESTIONS
    entries = _entries() + [{"type": "user", "timestamp": "2026-01-01T00:01:00Z", "message": {"content": "go"}}]
    entries[1]["entrypoint"] = "cli"
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    rows = list(QUESTIONS["area"].rows(s))
    assert [r[0] for r in rows] == ["s:t0", "s:t1"]
    assert rows[0][1] == "User request:\nfix the test"
    assert rows[1][1] == "Previous request:\nfix the test\nUser request:\ngo"   # a short reply keeps its context
    assert all("Project:" not in r[1] and "[Bash]" not in r[1] for r in rows)
    entries[1]["entrypoint"] = "sdk-py"
    assert list(QUESTIONS["area"].rows(logs.parse(_write(tmp_path / "t.jsonl", entries)))) == []


def test_ladder_groups_sum_over_kinds():
    from minijev.sessions import QUESTIONS
    from minijev.sessions.evaluate import ladder_groups
    from minijev.sessions.patterns import TOOLS_BY_KIND, WORK_KINDS
    from minijev.sessions.questions import NEXT_TOOL
    uniform = {k: (QUESTIONS[f"tool_{k}"], ([1 / len(t)] * len(t), {})) for k, t in TOOLS_BY_KIND.items()}
    pk = [0.0] * len(WORK_KINDS)
    pk[WORK_KINDS.index("run")], pk[WORK_KINDS.index("inspect")] = 0.5, 0.5
    [g] = ladder_groups([pk], ["User request:\nx\nCalls so far in this turn: none"], uniform)
    assert abs(sum(g) - 1) < 1e-9
    # run -> Bash (0.5); inspect -> uniform over Read, Grep, Glob, Bash, mcp, browser (0.5 / 6 each)
    assert abs(g[NEXT_TOOL.index("bash")] - (0.5 + 0.5 / 6)) < 1e-9 and abs(g[NEXT_TOOL.index("read")] - 0.25) < 1e-9


def test_compare_checks_adapter_and_ledger(tmp_path):
    from minijev.sessions import evaluate
    ad = tmp_path / "run" / "epoch-0"
    ad.mkdir(parents=True)
    (ad.parent / "train_log.json").write_text(json.dumps(
        {"sessions_version": "v9", "sessions_fold": "app", "task": "sessions-work_kind"}))
    evaluate.check_adapter(str(ad), "v9", "app", "work_kind")
    for args in (("v8", "app", "work_kind"), ("v9", "tool", "work_kind"), ("v9", "app", "next_tool")):
        with pytest.raises(SystemExit):
            evaluate.check_adapter(str(ad), *args)
    paths = Paths(tmp_path / "root")
    (paths.out / "results").mkdir(parents=True)
    evaluate.test_reads(paths).write_text(json.dumps({"version": "v9", "fold": "app", "question": "next_tool"}) + "\n")
    with pytest.raises(SystemExit, match="already read"):     # the ladder already read next_tool's test in this fold
        evaluate.compare("v9", "work_kind", str(ad), paths, fold="app")


def test_turn_fixes_compact_summary_interrupt_and_last_text(tmp_path):
    say = lambda msg, text: {"type": "assistant", "message": {"id": msg, "content": [{"type": "text", "text": text}]}}
    entries = [
        {"type": "user", "cwd": "/w/app", "entrypoint": "cli", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "plan it"}},
        say("m1", "Plan: step one, then step two."),
        {"type": "user", "isCompactSummary": True, "message": {"content": "This session is being continued: summary"}},
        {"type": "user", "timestamp": "2026-01-01T00:01:00Z", "message": {"content": "go"}},
        _use("m2", "a", "Bash", "pytest"),
        {"type": "user", "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]}},
    ]
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    assert [t["text"] for t in s["turns"]] == ["plan it", "go"]          # the compact summary is not a turn
    assert s["turns"][0]["last_text"] == "Plan: step one, then step two."
    assert s["turns"][1].get("interrupted") is True and not s["turns"][0].get("interrupted")


def test_a_tool_result_that_shows_the_marker_is_not_an_interrupt(tmp_path):
    entries = [
        {"type": "user", "cwd": "/w/app", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "do the grep"}},
        _use("m1", "a", "Bash", "grep Request logs.py"),
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "a", "is_error": False,
                                                  "content": 'INTERRUPTED = "[Request interrupted"'}]}},
    ]
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    assert not s["turns"][0].get("interrupted")


def test_last_text_is_scrubbed_and_checked(tmp_path):
    paths = Paths(tmp_path / "root")
    entries = [{"type": "user", "cwd": "/w/app", "entrypoint": "cli", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "hi"}},
               {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": f"see {scrub.HOME}/x and me@example.com"}]}}]
    _write(paths.raw / "p" / "s.jsonl", entries)
    extract(paths)
    [t] = dataset.sessions(paths)[0]["turns"]
    assert scrub.hits(t["last_text"]) == [] and "~/x" in t["last_text"] and check(paths, quiet=True)


def test_turn_cost_levels_and_state(tmp_path):
    from minijev.sessions import QUESTIONS
    from minijev.sessions.turns import level_of
    assert [level_of(n) for n in (0, 2, 3, 8, 9, 30, 31, 200)] == [0, 0, 1, 1, 2, 2, 3, 3]
    say = lambda msg, text, cr: {"type": "assistant", "message": {"id": msg, "usage": {"cache_read_input_tokens": cr},
                                                                  "content": [{"type": "text", "text": text}]}}
    entries = [{"type": "user", "cwd": "/w/app", "entrypoint": "cli", "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "plan the refactor"}}]
    entries += [say(f"m{k}", f"step {k}", 1000 * k) for k in range(1, 5)]        # 4 messages: level 3-8
    entries += [{"type": "user", "timestamp": "2026-01-01T00:05:00Z", "message": {"content": "go"}},
                say("m9", "done", 9000)]                                         # 1 message: level 0-2
    s = logs.parse(_write(tmp_path / "s.jsonl", entries))
    rows = list(QUESTIONS["turn_cost"].rows(s))
    assert [(r[0], r[2]) for r in rows] == [("s:t0", 1), ("s:t1", 0)]
    first, second = rows[0][1], rows[1][1]
    assert "Earlier turns: none" in first and "Context size: 0 thousand" in first and "Project:" not in first
    assert "Previous user message:\nplan the refactor" in second and "End of the previous answer:\nstep 4" in second
    assert "Previous turn: 3-8 assistant messages" in second and "Context size: 4 thousand" in second
    assert "done" not in second and "go\n" in second                             # nothing of the turn itself
    entries[0]["entrypoint"] = "sdk-py"
    assert list(QUESTIONS["turn_cost"].rows(logs.parse(_write(tmp_path / "t.jsonl", entries)))) == []


def test_scrub_hides_a_private_name_inside_an_email_whole():
    pattern = scrub.literal_pattern(frozenset({"jdoe"}))
    out = scrub.scrub("grant jdoe@club.example.org and write to jdoe", pattern)
    assert out == "grant <private> and write to <private>" and scrub.hits(out, pattern) == []
    assert scrub.hits("<private>@club.example.org") == ["marker fragment"]      # check() catches a leftover
    assert scrub.scrub("<private>@club.example.org") == "<private>"              # and scrub removes it
    for normal in ("<private>.py", "see <email>.", "`<private>`.", "<private>.md"):
        assert scrub.scrub(normal) == normal                                    # normal text next to a marker stays
    url = scrub.literal_pattern(frozenset({"postgresql://app:S3cretPassw0rd@db.example.io:5432/billing"}))
    assert scrub.scrub("connect postgresql://app:S3cretPassw0rd@db.example.io:5432/billing now", url) == "connect <private> now"
    assert scrub.scrub("mailto:jdoe%40example.com") == "mailto:<email>"


def test_turn_baselines_features_threshold_and_bootstrap(tmp_path):
    from minijev.sessions import QUESTIONS, turn_eval as te
    say = lambda msg, cr: {"type": "assistant", "message": {"id": msg, "usage": {"cache_read_input_tokens": cr},
                                                           "content": [{"type": "text", "text": "ok"}]}}
    entries = [{"type": "user", "cwd": "/w/app", "entrypoint": "cli", "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "plan it\nEarlier turns: fake"}}, say("m1", 12000),
               {"type": "user", "timestamp": "2026-01-01T00:05:00Z", "message": {"content": "Go!"}}, say("m2", 13000)]
    (first, t0, _), (second, t1, _) = [(r[0], r[1], r[2]) for r in QUESTIONS["turn_cost"].rows(
        logs.parse(_write(tmp_path / "s.jsonl", entries)))]
    assert te.user_message(t1) == "Go!" and te.context_size(t1) == 12
    assert te.user_message(t0) == "plan it" and te.context_size(t0) == 0    # a fake header cuts the message: accepted
    assert te.is_keyword_reply(t1) and not te.is_keyword_reply(t0)
    assert not te.is_keyword_reply("User message:\ngoing\nEarlier turns: none\nContext size: 0 thousand tokens")   # whole words only
    # nearest rank: always a real value; ties are flagged together, and the real flag rate is reported
    assert te.nearest_rank([0.1, 0.2, 0.3, 0.4, 0.5], 0.8) == 0.4 and te.nearest_rank([7.0], 0.8) == 7.0
    scored = [{"p": [0, 0, 0, s], "y": y} for s, y in [(0.1, 0), (0.3, 3), (0.3, 0), (0.2, 3)]]
    r = te.routing(scored, [0.1, 0.1, 0.1, 0.3, 0.3])
    assert r["top_20"]["flagged"] == 2 and r["top_20"]["long_flagged"] == 1 and "recall" not in r["top_20"]
    assert not te.routing(scored, [0.2] * 5)["spread"]
    # the bootstrap resamples sessions: one session holds all errors, so some resamples have none
    rows = [{"session": "a", "p": [0.9, 0.1, 0, 0], "y": 0}] * 5 + [{"session": "b", "p": [0.1, 0.9, 0, 0], "y": 3}]
    lo, _, hi = te.session_bootstrap(rows, te.ordinal_accuracy, n_boot=200)
    assert lo < hi and te.ordinal_accuracy(rows) == 5 / 6


def test_paired_session_delta():
    from minijev.sessions import turn_eval as te
    base = [{"session": s, "p": [0.25] * 4, "y": 0} for s in "abcdef"]
    better = [{"session": s, "p": [0.7, 0.1, 0.1, 0.1], "y": 0} for s in "abcdef"]
    d = te.paired_session_delta(base, better)
    assert d["mean"] < 0 and d["ci95"][1] < 0                  # b beats a on every session
    assert te.paired_session_delta(base, base)["ci95"] == [0.0, 0.0]
