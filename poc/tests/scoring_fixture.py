"""Write web/tests/scoring-fixture.json: Python answer() outputs for the TS parity test (web/lib/scoring.ts).

Run from poc/: PYTHONPATH=. uv run python tests/scoring_fixture.py
"""

import json
import random
from pathlib import Path

from minijev_poc import answer

rng = random.Random(0)
cases = []
for _ in range(60):
    kind = rng.choice(["noul", "choice", "score"])
    k = 2 if kind == "noul" else rng.randint(2, 7)
    q = {"type": kind, "instructions": "?"}
    if kind == "choice":
        q["criteria"] = {f"o{i}": None for i in range(k)}
    if kind == "score":
        q["criteria"] = [f"level {i}" for i in range(k)]
    logits = [rng.uniform(-8, 4) for _ in range(k)]
    t = rng.choice([1.0, 0.5, 1.93, 2.72, 3.5])
    b = rng.choice([0.0, 0.208, -1.0]) if kind == "noul" else 0.0
    cases.append({"q": q, "logits": logits, "t": t, "b": b, "expected": answer(q, logits, t, b)})

out = Path(__file__).resolve().parents[2] / "web" / "tests" / "scoring-fixture.json"
out.write_text(json.dumps(cases, indent=1))
print(f"wrote {len(cases)} cases to {out}")
