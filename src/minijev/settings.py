"""Settings: the dials of ask() and Engine, read from minijev.env and MINIJEV_* variables, and the fitted
calibration file. Paths default to the current directory: run the API and the CLI from poc/, or set
MINIJEV_ENV_FILE and MINIJEV_CALIBRATION_DIR."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SETTINGS_FILE = Path(os.environ.get("MINIJEV_ENV_FILE", Path.cwd() / "minijev.env"))
CALIBRATION_DIR = Path(os.environ.get("MINIJEV_CALIBRATION_DIR", Path.cwd() / "calibration"))
_FLOAT_OR_FITTED = ("temp_noul", "temp_choice", "temp_score", "bias_noul")


@dataclass(frozen=True)
class Settings:
    """The dials of ask() and Engine. Each field is MINIJEV_<FIELD> in minijev.env or in the environment.

    Precedence: environment variable > minijev.env > the defaults below. The defaults are the uncalibrated
    behaviour that the experiments measure; experiments always use Settings(), never the file.

    Calibration: with calibration="fitted", the values come from calibration/<model>.json, which E14 fitted on
    the train split and chose on the val split (datasets/splits_v2.json). A temperature or bias that is set
    explicitly overrides the fitted value; None means "use the fitted value" (or 1.0 / 0.0 without calibration).
    """

    model: str = MODEL
    threads: int = 6
    attn: str = "eager"  # or "sdpa"
    calibration: str = "none"  # "fitted": use calibration/<model>.json; "none": raw probabilities
    temp_noul: float | None = None  # temperature T: logits are divided by T. T > 1 = less confident
    temp_choice: float | None = None
    temp_score: float | None = None
    bias_noul: float | None = None  # Platt shift b, added to the Noul log-odds after the temperature
    choice_mode: str = "listwise"  # listwise | pointwise | averaged | selected (the mode chosen on val)
    score_mode: str = "pointwise"  # or "listwise": all levels in one prompt (ablation)
    min_label_mass: float = 0.5  # warn when less next-token probability than this is on the labels
    state_cache: int = 0  # the server keeps the key/values of this many recent states (Task 4.2); 0 = off
    share_question: bool = False  # two-level tree: the question text runs once, shared by its items (Task 4.1)
    fitted: dict = field(default_factory=dict, compare=False, repr=False)  # the loaded calibration file

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        values = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip("\"'")
        values.update({k: v for k, v in os.environ.items() if k.startswith("MINIJEV_")})
        known = {"MINIJEV_" + f.name.upper(): f for f in fields(cls) if f.name != "fitted"}
        unknown = sorted(set(values) - set(known) - {"MINIJEV_INSECURE_SSL"})
        if unknown:
            raise ValueError(f"unknown settings {unknown}; known: {sorted(known)}")
        kw = {}
        for k, f in known.items():
            if k not in values:
                continue
            v = values[k]
            if f.name in _FLOAT_OR_FITTED:
                kw[f.name] = None if v.lower() in ("", "fitted", "none") else float(v)
            elif isinstance(f.default, bool):
                kw[f.name] = v.lower() in ("1", "true", "yes")  # bool("false") is True
            else:
                kw[f.name] = type(f.default)(v)
        return cls(**kw).with_calibration()

    def with_calibration(self) -> "Settings":
        """Load calibration/<model>.json when calibration="fitted"; check every dial."""
        assert self.calibration in ("none", "fitted"), f"MINIJEV_CALIBRATION={self.calibration!r}"
        assert self.choice_mode in ("listwise", "pointwise", "averaged", "selected"), f"MINIJEV_CHOICE_MODE={self.choice_mode!r}"
        assert self.score_mode in ("listwise", "pointwise"), f"MINIJEV_SCORE_MODE={self.score_mode!r}"
        assert all(t is None or t > 0 for t in (self.temp_noul, self.temp_choice, self.temp_score)), "temperatures must be > 0"
        fitted = {}
        if self.calibration == "fitted":
            p = CALIBRATION_DIR / f"{self.model.split('/')[-1]}.json"
            if p.exists():
                fitted = json.loads(p.read_text())
        return replace(self, fitted=fitted)

    def resolved_choice_mode(self) -> str:
        if self.choice_mode == "selected":
            return (self.fitted.get("choice") or {}).get("selected_mode", "listwise")
        return self.choice_mode

    def calibrator(self, qtype: str, mode: str | None = None) -> tuple[float, float, str]:
        """(temperature, bias, source) for one question. source: "manual", "fitted" or "none"."""
        f = self.fitted
        if qtype == "noul":
            if self.temp_noul is not None or self.bias_noul is not None:
                return (self.temp_noul or 1.0, self.bias_noul or 0.0, "manual")
            if f.get("noul"):
                n = f["noul"]
                if n["selected"] == "platt":
                    return 1 / n["platt"]["a"], n["platt"]["b"], "fitted"
                return n["temperature"], 0.0, "fitted"
            return 1.0, 0.0, "none"
        if qtype == "choice":
            if self.temp_choice is not None:
                return self.temp_choice, 0.0, "manual"
            t = ((f.get("choice") or {}).get("temperature") or {}).get(mode or self.resolved_choice_mode())
            return (t, 0.0, "fitted") if t else (1.0, 0.0, "none")
        if self.temp_score is not None:
            return self.temp_score, 0.0, "manual"
        return 1.0, 0.0, "none"  # no labelled ordinal data yet: Scores are never fitted

    def temperature(self, qtype: str, mode: str | None = None) -> float:
        return self.calibrator(qtype, mode)[0]
