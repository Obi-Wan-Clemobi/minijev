"""Fast tests: no model is loaded."""

import math

import pytest
import torch

from experiments import bootstrap_ci, parse_probs_lenient
from minijev import Engine, answer, choice_confidence, score_confidence, softmax


def test_choice_confidence_endpoints():
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0)
    assert choice_confidence([1, 0, 0]) == pytest.approx(1)
    assert choice_confidence([0.6, 0.4]) == pytest.approx(0.2)  # (k·p_max − 1)/(k − 1)


def test_score_confidence_is_ordinal():
    # DESIGN.md §5.5: the same 0.6/0.4 split costs more across distant levels.
    assert score_confidence([0.6, 0.4, 0]) == pytest.approx(0.4)
    assert score_confidence([0.6, 0, 0.4]) == pytest.approx(0)
    assert score_confidence([0, 1, 0, 0]) == pytest.approx(1)
    assert score_confidence([0.25] * 4) == 0


def test_score_confidence_equals_choice_confidence_on_three_adjacent_levels():
    # RESEARCH.md §3.3: why the Choice formula seemed to fit 3-level Scores.
    for p in ([0.7, 0.3, 0], [0.1, 0.9, 0], [0, 0.45, 0.55]):
        assert score_confidence(p) == pytest.approx(choice_confidence(p))


def test_answer_shapes():
    assert answer({"type": "noul"}, [0.0, 0.0])["noul"] == pytest.approx(0.5)
    a = answer({"type": "score", "criteria": ["a", "b", "c"]}, [0.0, 0.0, 0.0])
    assert a["score"] == pytest.approx(1.0) and a["legend"] == {"0": "a", "1": "b", "2": "c"}
    assert sum(softmax([1.0, 2.0, 3.0], temperature=2.0)) == pytest.approx(1)


def test_pack_mask_and_positions():
    pos, mask, last = Engine.pack(None, 3, [2, 1])
    visible = (mask[0, 0] == 0).int().tolist()
    assert visible == [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],  # branch 1 sees the prefix and itself
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 1],  # branch 2 never sees branch 1
    ]
    assert pos[0].tolist() == [0, 1, 2, 3, 4, 3]  # positions restart after the prefix
    assert last.tolist() == [4, 5]


KEYS = ["World", "Sports", "Business", "Technology"]


@pytest.mark.parametrize("text, winner", [
    ('{"World": 0.1, "Sports": 0.7, "Business": 0.1, "Technology": 0.1}', "Sports"),
    ('{"topic": "Technology", "probability": 0.7}', "Technology"),
    ('{"topic": "Business: Business, companies and the economy", "probability": 0.7}', "Business"),
    ('{"Sport": 0.999, "Business": 0.001, "Technology": 0.001, "World": 0.001}', "Sports"),
    ('{"business": 0.98, "technology": 0.02}', "Business"),
    ('{"STATE": "World: World news, politics}, {"Probability": 0.8}}', "World"),
])
def test_lenient_parser_on_observed_replies(text, winner):
    p = parse_probs_lenient(text, KEYS)
    assert p is not None and sum(p) == pytest.approx(1)
    assert KEYS[max(range(4), key=p.__getitem__)] == winner


def test_lenient_parser_spreads_unassigned_mass():
    assert parse_probs_lenient('{"topic": "World", "probability": 0.4}', KEYS) == pytest.approx([0.4, 0.2, 0.2, 0.2])
    assert parse_probs_lenient("I think it is about sport.", KEYS) is None


def test_bootstrap_ci_brackets_the_mean():
    xs = [0, 1] * 50
    lo, hi = bootstrap_ci(xs, lambda s: sum(s) / len(s))
    assert lo < 0.5 < hi and hi - lo < 0.3
    assert bootstrap_ci([1] * 10, lambda s: sum(s) / len(s)) == [1, 1]


def test_settings_file_env_and_defaults(tmp_path, monkeypatch):
    from minijev import Settings
    f = tmp_path / "x.env"
    f.write_text("MINIJEV_TEMP_NOUL=2.5   # comment\nMINIJEV_CHOICE_MODE=pointwise\n")
    monkeypatch.setenv("MINIJEV_TEMP_SCORE", "3")
    s = Settings.load(f)
    assert (s.temp_noul, s.temp_score, s.temp_choice, s.choice_mode) == (2.5, 3.0, None, "pointwise")
    assert s.calibrator("choice") == (1.0, 0.0, "none")  # unset and uncalibrated: raw
    monkeypatch.setenv("MINIJEV_TEMP_NOUL", "4")  # the environment wins over the file
    assert Settings.load(f).temp_noul == 4.0
    monkeypatch.setenv("MINIJEV_TEMPNOUL", "4")
    with pytest.raises(ValueError):
        Settings.load(f)


def test_shipped_env_file_uses_fitted_calibration(monkeypatch):
    from minijev import SETTINGS_FILE, Settings
    for k in [k for k in __import__("os").environ if k.startswith("MINIJEV_")]:
        monkeypatch.delenv(k)
    s = Settings.load(SETTINGS_FILE)
    assert s.calibration == "fitted" and s.choice_mode == "selected"
    assert (s.temp_noul, s.bias_noul, s.temp_choice, s.temp_score) == (None, None, None, None)  # no hand-typed values


def test_fitted_calibration_is_used_and_overridable(tmp_path, monkeypatch):
    import json
    import minijev.settings
    from minijev import Settings
    (tmp_path / "Qwen2.5-0.5B-Instruct.json").write_text(json.dumps({
        "provenance": {"fitted_on": "train"},
        "noul": {"temperature": 2.0, "platt": {"a": 0.5, "b": 0.3}, "selected": "platt"},
        "choice": {"temperature": {"listwise": 3.0, "pointwise": 1.5, "averaged": 2.5}, "selected_mode": "pointwise"}}))
    monkeypatch.setattr(minijev.settings, "CALIBRATION_DIR", tmp_path)
    s = Settings(calibration="fitted", choice_mode="selected").with_calibration()
    assert s.calibrator("noul") == (2.0, 0.3, "fitted")  # Platt: T = 1/a, b
    assert s.resolved_choice_mode() == "pointwise"
    assert s.calibrator("choice") == (1.5, 0.0, "fitted")
    assert s.calibrator("choice", "listwise") == (3.0, 0.0, "fitted")
    assert Settings(calibration="fitted", temp_choice=4.0).with_calibration().calibrator("choice") == (4.0, 0.0, "manual")
    assert Settings(calibration="none").with_calibration().calibrator("noul") == (1.0, 0.0, "none")
    assert s.calibrator("score") == (1.0, 0.0, "none")  # Scores are never fitted

def test_temperature_and_bias():
    q = {"type": "noul"}
    assert answer(q, [2.0, 0.0], temperature=2.0)["noul"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert answer(q, [0.0, 0.0], bias=1.0)["noul"] == pytest.approx(1 / (1 + math.exp(-1)))
