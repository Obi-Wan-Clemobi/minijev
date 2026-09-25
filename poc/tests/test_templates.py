"""Pre-registration (PLAN.md Task 1.2): the templates in use are the ones recorded, and test data cannot be read
while tuning."""

import json
from pathlib import Path

import pytest

import data
from minijev_poc import template_fingerprint

PREREG = Path(__file__).resolve().parents[1] / "templates" / "v1-preregistration.json"


def test_templates_match_the_preregistration():
    # A template change must come with a new version file (v2-preregistration.json), chosen on val only.
    assert template_fingerprint() == json.loads(PREREG.read_text())["code_fingerprint_sha256"]


def test_test_split_is_closed_while_tuning():
    with data.tuning():
        data.load_split("boolq", "val")  # allowed
        with pytest.raises(PermissionError):
            data.load_split("boolq", "test")
    data.load_split("boolq", "test")  # allowed again outside tuning
