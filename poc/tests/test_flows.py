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
    assert seen[0]["state"]["decisions so far"] == []
    assert seen[1]["state"]["decisions so far"] == [
        {"question": "What type of tool is needed for this task?", "answer": "search", "confidence": 0.9}]


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
    assert back["steps"]["upset"]["transitions"][2] == {"from_answer": 2, "target": "human", "condition": None}
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
