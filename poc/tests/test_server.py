"""The web API returns what the POC returns. Loads Qwen2.5-0.5B-Instruct."""

import pytest
from fastapi.testclient import TestClient

import server
from minijev_poc import MODEL, Engine, Settings, ask

pytestmark = pytest.mark.model

REQ = {
    "state": "Hi, my Stripe integration has failed for 3 days. Losing sales. Help ASAP.",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
            "billing": "Payments, invoices, refunds", "technical": "Integrations, bugs, outages", "sales": "Pricing"}},
        "tone": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
    },
}
DEFAULTS = {f: getattr(Settings(), f) for f in ("temp_noul", "temp_choice", "temp_score", "bias_noul",
                                                  "choice_mode", "score_mode")}


@pytest.fixture(scope="module")
def client():
    server._engine = Engine(MODEL, attn="eager", threads=6)  # fixed model: ignore minijev.env
    return TestClient(server.app)


def test_ask_matches_poc(client):
    r = client.post("/v1/ask", json={**REQ, "settings": DEFAULTS})
    assert r.status_code == 200, r.text
    direct = ask(server._engine, REQ, settings=Settings(), debug=True)
    for qid in REQ["questions"]:
        assert r.json()["debug"]["raw"][qid]["logits"] == pytest.approx(direct["debug"]["raw"][qid]["logits"], abs=1e-4)
    assert r.json()["usage"]["output_tokens"] == 0


def test_settings_change_the_answer_not_the_logits(client):
    hot = client.post("/v1/ask", json={**REQ, "settings": {**DEFAULTS, "temp_noul": 3.0}}).json()
    cold = client.post("/v1/ask", json={**REQ, "settings": DEFAULTS}).json()
    assert hot["answers"]["urgency"]["noul"] < cold["answers"]["urgency"]["noul"]
    assert hot["debug"]["raw"]["urgency"]["logits"] == pytest.approx(cold["debug"]["raw"]["urgency"]["logits"], abs=1e-4)


def test_tree_layout_adds_up(client):
    t = client.post("/v1/tree", json={**REQ, "settings": DEFAULTS}).json()
    assert len(t["branches"]) == 1 + 1 + 4  # Noul, listwise Choice, 4 Score levels
    assert t["total"] == t["prefix"]["length"] + sum(b["length"] for b in t["branches"])
    assert all(b["positions"][0] == t["prefix"]["length"] for b in t["branches"])  # positions restart
    lo, hi = t["prefix"]["state_span"]
    assert "".join(t["prefix"]["tokens"][lo:hi]).rstrip() == REQ["state"]  # BPE can merge the last character with the blank line
    assert client.post("/v1/ask", json={**REQ, "settings": DEFAULTS}).json()["usage"]["input_tokens"] == t["total"]


def test_bad_requests_are_400(client):
    one_option = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}}}
    assert client.post("/v1/ask", json=one_option).status_code == 400
    assert client.post("/v1/ask", json={**REQ, "settings": {"temp": 2}}).status_code == 400
    assert client.post("/v1/ask", json={**REQ, "mode": "fast"}).status_code == 400


def test_compare_readout_equals_logprobs(client):
    r = client.post("/v1/compare", json={**REQ, "methods": ["readout", "logprobs_cached"]}).json()
    assert r["methods"]["logprobs_cached"]["agrees_with_readout"] == 1.0
    assert r["methods"]["readout"]["output_tokens"] == 0


def test_presets_and_results(client):
    ids = [p["id"] for p in client.get("/v1/presets").json()]
    assert ids[:2] == ["support", "shoes"]
    res = client.get("/v1/results").json()
    assert res["0.5B"]["calibration"]["metrics"]["raw"]["ece"] == pytest.approx(0.161, abs=1e-3)
