"""Task 4.2: time a request with the state cache off, and on (a hit), interleaved. Median of 7."""
import json, statistics, time
from minijev import Engine, raw_scores
from minijev.engine import StateCache
from minijev.fixtures import GDPR_QUESTIONS, SUPPORT_TICKET, gdpr_state
from minijev.judge import with_modes
from minijev.settings import Settings

e = Engine("Qwen/Qwen2.5-0.5B-Instruct", attn="eager", threads=6)
s = Settings(score_mode="pointwise", choice_mode="listwise")
fix = lambda r: {**r, "questions": {k: with_modes(q, s) for k, q in r["questions"].items()}}
cases = {"support ticket (3 questions)": fix(SUPPORT_TICKET)}
for n in (512, 2000):
    cases[f"GDPR {n} tokens (13 questions)"] = fix({"state": gdpr_state(e, n), "questions": GDPR_QUESTIONS})
out = []
for name, req in cases.items():
    for mode in ("kv", "packed"):
        times = {"off": [], "hit": []}
        e.state_cache = StateCache(4)
        raw_scores(e, req, mode, True)  # warm up, and store the state
        for _ in range(7):
            for c in ("off", "hit"):
                e.state_cache.size = 0 if c == "off" else 4
                t = time.perf_counter(); _, u = raw_scores(e, req, mode, True); times[c].append(time.perf_counter() - t)
        off, hit = statistics.median(times["off"]), statistics.median(times["hit"])
        out.append({"case": name, "mode": mode, "input_tokens": u["input_tokens"], "prefix_tokens": len(e.prefix_ids(req["state"])),
                    "off_s": round(off, 3), "hit_s": round(hit, 3), "speedup": round(off / hit, 2)})
        print(out[-1], flush=True)
print(json.dumps(out))
# The results are in poc/results/state_cache.json.
