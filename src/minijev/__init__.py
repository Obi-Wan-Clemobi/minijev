"""minijev: typed, calibrated decisions read out of one forward pass of a small open model.

    from minijev import Engine, Settings, ask
    engine = Engine()                       # the model in minijev.env (default Qwen2.5-0.5B-Instruct)
    ask(engine, {"state": "…", "questions": {"q": {"type": "noul", "instructions": "…?"}}})
"""

from .engine import MODES, Branch, Engine
from .judge import ask, branches_for, raw_scores, validate, with_modes
from .primitives import answer, choice_confidence, class_logits, rounded, score_confidence, softmax
from .prompt import (CONTENT_FREE_STATE, LETTERS, NO, SYSTEM, YES, choice_block, noul_block, proposal_block, render,
                     render_inline, score_listwise_block, template_fingerprint)
from .settings import CALIBRATION_DIR, MODEL, SETTINGS_FILE, Settings

__all__ = ["MODES", "Branch", "Engine", "ask", "branches_for", "raw_scores", "validate", "with_modes", "answer",
           "choice_confidence", "class_logits", "rounded", "score_confidence", "softmax", "CONTENT_FREE_STATE",
           "LETTERS", "NO", "SYSTEM", "YES", "choice_block", "noul_block", "proposal_block", "render", "render_inline",
           "score_listwise_block", "template_fingerprint", "CALIBRATION_DIR", "MODEL", "SETTINGS_FILE", "Settings"]
__version__ = "0.1.0"
