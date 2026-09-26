"""Training data for minijev from your own Claude Code sessions. See docs/SESSIONS.md.

    minijev sessions sync            copy ~/.claude/projects/*.jsonl to the private archive (never deletes)
    minijev sessions extract         parse and scrub the archive into events.jsonl
    minijev sessions sample Q        print random states of question Q: review them for private data
    minijev sessions check           zero pattern hits and zero private-string hits, or exit 1
    minijev sessions stats           rows per question, split and label
    minijev sessions freeze          write a frozen version (v1, v2, …) of every question
    minijev sessions points Q        unlabelled decision points of an external question, for a labeller
    minijev sessions consensus Q     combine labellers and reviewers of an external question (consensus.py)
    minijev sessions baselines V     prior, previous-call and zero-shot readout baselines on val of version V
"""

from .dataset import Paths, check, extract, freeze, load_split, manifest, rows, sessions, split_of, stats
from .questions import QUESTIONS, Question, load, register, state

__all__ = ["Paths", "QUESTIONS", "Question", "check", "extract", "freeze", "load", "load_split", "manifest", "register",
           "rows", "sessions", "split_of", "state", "stats"]
