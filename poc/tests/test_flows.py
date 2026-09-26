"""Flows: the executor with a fake ask() (fast), and the tool-selection template end to end (model)."""

import json
from pathlib import Path

import pytest

from minijev.flows import Flow, check, outcome, run

TEMPLATES = Path(__file__).parent.parent / "flows" / "templates"


def load(name: str) -> Flow:
    return Flow.model_validate(json.loads((TEMPLATES / f"{name}.json").read_text()))


def fake(answers: dict):
    """ask() that returns a fixed Jev-shaped answer per step id, and records every request."""
    seen = []

    def ask(req):
        seen.append(req)
        (qid,) = req["questions"]
        return {"answers": {qid: answers[qid]}}
    return ask, seen


def choice(pick: str, keys: list[str], conf: float) -> dict:
    return {"type": "choice", "choice": pick, "confidence": conf, "probabilities": {k: float(k == pick) for k in keys}}


def events(flow, answers, query="find all TODO comments in Python files"):
    ask, seen = fake(answers)
    out = list(run(flow, query, ask))
    return out, seen


@pytest.mark.parametrize("name", ["tool-selection", "customer-triage"])
def test_templates_are_valid(name):
    assert check(load(name))["errors"] == []


def test_confident_search_goes_straight_to_the_search_tool():
    out, seen = events(load("tool-selection"), {
        "tool_type": choice("search", ["search", "file", "network"], 0.9),
        "search_tool": choice("ripgrep", ["grep", "ripgrep"], 0.4)})
    assert [e["step"] for e in out if e["event"] == "decision"] == ["tool_type", "search_tool"]
    assert out[-1]["status"] == "completed" and out[-1]["path"] == ["tool_type", "search_tool"]


def test_unsure_search_asks_the_clarifying_question_first():
    out, _ = events(load("tool-selection"), {
        "tool_type": choice("search", ["search", "file", "network"], 0.3),
        "contents_or_names": {"type": "noul", "noul": 0.2},
        "file_tool": choice("find", ["find", "cat", "mv"], 0.8)})
    assert out[-1]["path"] == ["tool_type", "contents_or_names", "file_tool"]
    assert out[1]["answer"] is False and out[0]["transition"] == 1  # the fallback arrow, after the condition failed


def test_later_steps_see_earlier_decisions():
    _, seen = events(load("tool-selection"), {
        "tool_type": choice("search", ["search", "file", "network"], 0.9),
        "search_tool": choice("grep", ["grep", "ripgrep"], 0.9)})
    assert seen[0]["state"] == seen[1]["state"] == "find all TODO comments in Python files"   # one cached state
    assert not seen[0]["questions"]["tool_type"]["instructions"].startswith("Decided so far")
    assert seen[1]["questions"]["search_tool"]["instructions"].startswith(
        "Decided so far: What type of tool is needed for this task? → search (0.90)\n")


def test_score_uses_the_rounded_expected_level():
    s = load("customer-triage").steps["upset"]
    assert outcome(s, {"type": "score", "score": 1.49, "confidence": 0.3, "probabilities": {}})[0] == 1
    assert outcome(s, {"type": "score", "score": 1.5, "confidence": 0.3, "probabilities": {}})[0] == 2


def test_noul_confidence_is_zero_at_even_odds():
    s = load("customer-triage").steps["urgent"]
    assert outcome(s, {"type": "noul", "noul": 0.5}) [:2] == (True, 0.0)
    assert outcome(s, {"type": "noul", "noul": 0.1})[:2] == (False, pytest.approx(0.8))


def test_no_matching_arrow_stops_at_that_step():
    f = load("tool-selection")
    f.steps["tool_type"].transitions = f.steps["tool_type"].transitions[:1]  # only "search" with confidence >= 0.5
    out, _ = events(f, {"tool_type": choice("file", ["search", "file", "network"], 0.9)})
    assert out[-1]["status"] == "no_transition" and out[-1]["step"] == "tool_type"


def test_a_loop_without_exit_stops_after_max_steps():
    f = load("customer-triage")
    f.steps["urgent"].transitions[0].target = "urgent"
    out, _ = events(f, {"urgent": {"type": "noul", "noul": 0.9}})
    assert out[-1]["status"] == "max_steps" and len(out[-1]["decisions"]) == 20


def test_invalid_flow_does_not_run():
    f = load("tool-selection")
    f.steps["search_tool"].transitions[0].target = "nowhere"
    out, seen = events(f, {})
    assert out == [out[-1]] and out[-1]["status"] == "invalid" and not seen
    assert any("nowhere" in e["message"] for e in out[-1]["errors"])


def test_check_warns_about_unreachable_steps_and_open_answers():
    f = load("tool-selection")
    f.steps["tool_type"].transitions = [t for t in f.steps["tool_type"].transitions if t.from_answer != "file"]
    w = [x["message"] for x in check(f)["warnings"]]
    assert any("'file'" in m for m in w)


@pytest.mark.model
def test_tool_selection_end_to_end():
    from minijev import MODEL, Engine
    from minijev.judge import ask
    from minijev.settings import Settings
    engine = Engine(MODEL, attn="eager", threads=6)
    out = list(run(load("tool-selection"), "find all TODO comments in Python files",
                   lambda req: ask(engine, req, settings=Settings())))
    end = out[-1]
    assert end["status"] == "completed" and end["path"][0] == "tool_type" and end["path"][-1] in ("search_tool", "file_tool")


# ---- the /v1/flows endpoints (no model: engine() and ask() are replaced)

@pytest.fixture
def api(tmp_path, monkeypatch):
    import shutil
    from fastapi.testclient import TestClient
    import minijev.api.server as server
    shutil.copytree(TEMPLATES, tmp_path / "templates")
    monkeypatch.setattr(server, "FLOWS", tmp_path)
    monkeypatch.setattr(server, "engine", lambda: None)
    return server, TestClient(server.app)


def test_save_list_load_delete_round_trip(api):
    _, c = api
    templates = c.get("/v1/flows/templates").json()
    flow = next(t for t in templates if t["id"] == "customer-triage")  # has true/false, int levels, missing from_answer
    r = c.post("/v1/flows", json=flow)
    assert r.status_code == 200 and r.json()["errors"] == []
    assert [f["id"] for f in c.get("/v1/flows").json()] == ["customer-triage"]
    back = c.get("/v1/flows/customer-triage").json()
    assert Flow.model_validate(back) == Flow.model_validate(flow)
    assert back["steps"]["upset"]["transitions"][2] == {"from_answer": 2, "from_question": None, "target": "human",
                                                        "condition": None}
    assert c.delete("/v1/flows/customer-triage").status_code == 204
    assert c.get("/v1/flows/customer-triage").status_code == 404


def test_check_reports_a_bad_target_and_ids_are_safe(api):
    _, c = api
    flow = json.loads((TEMPLATES / "tool-selection.json").read_text())
    flow["steps"]["search_tool"]["transitions"][0]["target"] = "nowhere"
    assert any("nowhere" in e["message"] for e in c.post("/v1/flows/check", json=flow).json()["errors"])
    assert c.get("/v1/flows/..%2Fsecret").status_code in (400, 404)


def test_run_streams_one_line_per_step_and_an_error_ends_the_stream(api, monkeypatch):
    server, c = api
    answers = {"tool_type": choice("search", ["search", "file", "network"], 0.9),
               "search_tool": choice("grep", ["grep", "ripgrep"], 0.9)}
    monkeypatch.setattr(server, "ask", lambda e, body, mode, settings: {"answers": {q: answers[q] for q in body["questions"]}})
    flow = json.loads((TEMPLATES / "tool-selection.json").read_text())
    lines = [json.loads(l) for l in c.post("/v1/flows/run", json={"flow": flow, "query": "find TODOs"}).text.splitlines()]
    assert [l["event"] for l in lines] == ["decision", "decision", "end"] and lines[-1]["status"] == "completed"

    def boom(*a, **k):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(server, "ask", boom)
    lines = [json.loads(l) for l in c.post("/v1/flows/run", json={"flow": flow, "query": "x"}).text.splitlines()]
    assert lines[-1]["event"] == "end" and lines[-1]["status"] == "error" and "model crashed" in lines[-1]["message"]


def test_check_warns_when_an_answer_has_only_conditional_arrows():
    f = load("customer-triage")
    f.steps["team"].transitions[0].condition = {"confidence_gte": 0.7}
    f = Flow.model_validate(f.model_dump())
    assert any("only 'if sure'" in w["message"] and w["step"] == "team" for w in check(f)["warnings"])


# ---- enhance-flow-engine: fan-out, options from an earlier answer, ranking, hand-off, cost

def multi(answers: dict):
    """ask() for requests with several questions; it reports 10 input tokens per request."""
    seen = []

    def ask(req):
        seen.append(req)
        return {"answers": {q: answers[q] for q in req["questions"]}, "usage": {"input_tokens": 10}}
    return ask, seen


LADDER = {
    "id": "ladder", "name": "Kind of work, then tool", "start": "kind",
    "steps": {
        "kind": {"id": "kind", "type": "choice", "instructions": "What kind of work is next?",
                 "criteria": {"inspect": None, "research": None},
                 "fanout": [{"id": "parallel", "type": "noul", "instructions": "Can the next calls run together?"}],
                 "transitions": [{"from_answer": None, "target": "ESCALATE", "condition": {"confidence_lt": 0.4}},
                                 {"from_answer": None, "target": "tool"}]},
        "tool": {"id": "tool", "type": "choice", "instructions": "Which tool?",
                 "options_from": {"step": "kind", "map": {"inspect": {"Read": None, "Grep": None, "Bash": None},
                                                          "research": {"WebSearch": None, "WebFetch": None}}},
                 "transitions": [{"from_answer": None, "target": "DONE"}]},
    },
}


def test_fanout_answers_several_questions_in_one_request():
    f = Flow.model_validate(LADDER)
    ask, seen = multi({"kind": choice("inspect", ["inspect", "research"], 0.8), "parallel": {"type": "noul", "noul": 0.9},
                       "tool": choice("Grep", ["Read", "Grep", "Bash"], 0.7)})
    out = list(run(f, "find where the retry limit is set", ask))
    assert set(seen[0]["questions"]) == {"kind", "parallel"}              # one request for the fan-out step
    assert [(e["question_id"], e["answer"]) for e in out if e["event"] == "decision"] == [
        ("kind", "inspect"), ("parallel", True), ("tool", "Grep")]
    assert out[-1]["status"] == "completed" and out[-1]["path"] == ["kind", "tool"]
    assert out[-1]["usage"] == {"input_tokens": 20, "requests": 2}
    assert seen[0]["state"] == seen[1]["state"] == "find where the retry limit is set"


def test_options_follow_the_earlier_answer():
    f = Flow.model_validate(LADDER)
    ask, seen = multi({"kind": choice("research", ["inspect", "research"], 0.8), "parallel": {"type": "noul", "noul": 0.2},
                       "tool": choice("WebFetch", ["WebSearch", "WebFetch"], 0.6)})
    list(run(f, "what changed in the new API version", ask))
    assert seen[1]["questions"]["tool"]["criteria"] == {"WebSearch": None, "WebFetch": None}


def test_check_fails_when_an_answer_has_no_options():
    bad = json.loads(json.dumps(LADDER))
    del bad["steps"]["tool"]["options_from"]["map"]["research"]
    assert any("no options for the answer(s) ['research']" in e["message"] for e in check(Flow.model_validate(bad))["errors"])


def test_choice_decisions_are_ranked():
    f = Flow.model_validate(LADDER)
    probs = {"kind": {"type": "choice", "choice": "inspect", "confidence": 0.5, "probabilities": {"inspect": 0.75, "research": 0.25}},
             "parallel": {"type": "noul", "noul": 0.5}, "tool": {"type": "choice", "choice": "Bash", "confidence": 0.4,
             "probabilities": {"Read": 0.2, "Grep": 0.1, "Bash": 0.7}}}
    ask, _ = multi(probs)
    tool = [e for e in run(f, "x", ask) if e.get("question_id") == "tool"][0]
    assert tool["ranked"] == [["Bash", 0.7], ["Read", 0.2], ["Grep", 0.1]]


def test_low_confidence_hands_off_to_the_large_model():
    f = Flow.model_validate(LADDER)
    ask, seen = multi({"kind": choice("inspect", ["inspect", "research"], 0.3), "parallel": {"type": "noul", "noul": 0.5}})
    end = list(run(f, "x", ask))[-1]
    assert end["status"] == "escalated" and end["step"] == "kind" and end["confidence"] == 0.3
    assert end["condition"] == {"confidence_gte": None, "confidence_lt": 0.4} and len(seen) == 1


def test_a_transition_can_test_a_fanout_answer():
    f = json.loads(json.dumps(LADDER))
    f["steps"]["kind"]["transitions"] = [{"from_question": "parallel", "from_answer": True, "target": "DONE"},
                                         {"from_answer": None, "target": "tool"}]
    ask, _ = multi({"kind": choice("inspect", ["inspect", "research"], 0.9), "parallel": {"type": "noul", "noul": 0.9}})
    assert list(run(Flow.model_validate(f), "x", ask))[-1]["path"] == ["kind"]
    f["steps"]["kind"]["transitions"][0]["from_question"] = "nope"
    assert any("'nope'" in e["message"] for e in check(Flow.model_validate(f))["errors"])


def test_check_fails_when_a_path_skips_the_options_source():
    f = json.loads(json.dumps(LADDER))
    f["start"] = "gate"
    f["steps"]["gate"] = {"id": "gate", "type": "noul", "instructions": "Is it simple?",
                          "transitions": [{"from_answer": True, "target": "tool"}, {"from_answer": False, "target": "kind"}]}
    flow = Flow.model_validate(f)
    assert any("without 'kind'" in e["message"] for e in check(flow)["errors"])
    end = list(run(flow, "x", multi({})[0]))[-1]
    assert end["status"] == "invalid" and end["usage"] == {"input_tokens": 0, "requests": 0}


def test_coverage_warnings_with_mixed_arrows_and_unique_fanout_ids():
    f = json.loads(json.dumps(LADDER))
    f["steps"]["kind"]["transitions"] = [{"from_question": "parallel", "from_answer": True, "target": "DONE"},
                                         {"from_answer": "inspect", "target": "tool"}]
    w = [x["message"] for x in check(Flow.model_validate(f))["warnings"]]
    assert any("no arrow for the answer(s) ['research']" in m for m in w)
    f["steps"]["tool"]["fanout"] = [{"id": "parallel", "type": "noul", "instructions": "again?"}]
    assert any("'parallel' is not unique" in e["message"] for e in check(Flow.model_validate(f))["errors"])


def test_api_runs_fanout_flow_and_streams_new_fields(api, monkeypatch):
    server, c = api
    answers = {"kind": choice("inspect", ["inspect", "research"], 0.3), "parallel": {"type": "noul", "noul": 0.8}}
    monkeypatch.setattr(server, "ask", lambda e, body, mode, settings: {
        "answers": {q: answers[q] for q in body["questions"]}, "usage": {"input_tokens": 7}})
    lines = [json.loads(l) for l in c.post("/v1/flows/run", json={"flow": LADDER, "query": "find it"}).text.splitlines()]
    assert [l.get("question_id") for l in lines[:-1]] == ["kind", "parallel"]
    assert lines[0]["ranked"][0][0] == "inspect"
    assert lines[-1]["status"] == "escalated" and lines[-1]["usage"] == {"input_tokens": 7, "requests": 1}
    saved = c.post("/v1/flows", json=LADDER).json()
    assert saved["errors"] == [] and c.get("/v1/flows/ladder").json()["steps"]["tool"]["options_from"]["step"] == "kind"


def test_check_fails_when_options_come_from_the_step_itself():
    f = json.loads(json.dumps(LADDER))
    f["steps"]["tool"]["options_from"]["step"] = "tool"
    assert any("from this step itself" in e["message"] for e in check(Flow.model_validate(f))["errors"])
