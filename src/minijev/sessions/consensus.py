"""Consensus labels for an external question, from several model labellers and reviewers.

Folder layout, under ROOT/sessions/labels/<question>/:

    round1/*.jsonl       labellers: {"id", "y", "unsure", "why", ...}; each item labelled by 2 or more of them, blind
    review/*.jsonl       reviewers: the same fields, for the items in queue-review.jsonl; each item gets 2, blind to
                         each other but not to the labellers
    adjudicate/*.jsonl   2 adjudicators, for the items in queue-adjudicate.jsonl; each sees every opinion and the facts,
                         but not the other adjudicator

Rules (decide() applies them; nothing else changes a label):
1. An item is **agreed** if all its labellers give the same y and none is unsure. Otherwise it is **disputed**.
2. Every disputed item, and an audit sample of AUDIT of the agreed items (seeded), goes to review.
3. A reviewed item whose 2 reviewers agree (and neither is unsure) gets their y. If they split, it goes to adjudication.
4. An adjudicated item whose 2 adjudicators agree (and neither is unsure) gets their y. Otherwise it is
   **contested**: it gets the majority y of all its opinions (a tie gives 1, "ask first"), and unsure true.
5. If reviewers overturn more than MAX_OVERTURN of the audited agreed items, the agreed items share errors: every
   agreed item goes to review (a second audit round), and rules 3 and 4 apply to them.
6. facts.json lists facts that a state cannot show, each checked outside the data: {"pattern": regex on the next
   call, "fact": text, "ids": optional list of the only item ids it applies to}. An item whose next call matches a fact, and that any labeller or reviewer answered y = 1,
   goes to adjudication with the fact, and rule 4 decides it.
7. revision/*.jsonl: a later round under a revised rubric, for items that the revision can change (the contested items,
   and decided items that the clarifications touch). Each item gets 2 new opinions, blind to all earlier ones. If they
   agree (and neither is unsure) the item is **revised** to their y. Otherwise it is contested, as in rule 4.
8. deliberation/*.jsonl: for items still contested after rule 7, 2 labellers each see the other's rule-7 opinion and
   may converge. Same test as rule 7 (status **deliberated**). What stays split is contested: the state does not hold
   what is needed to decide it.

The consensus file ROOT/sessions/labels/<question>.jsonl gets one line per decided item: {"id", "y", "status", "by",
"sample", "unsure"}. status is agreed, reviewed, adjudicated or contested. A contested item has unsure true.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

AUDIT, MAX_OVERTURN, SEED = 0.10, 0.05, 2026
LATER_STAGES = (("revision", "revised"), ("deliberation", "deliberated"),     # rules 7 and 8: first v2.1 round
                ("revision2", "revised"), ("deliberation2", "deliberated"))   # the same, for the v2.1 sweep


def read(folder: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for f in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                out[r["id"]].append(r | {"file": f.stem})
    return out


def majority(opinions: list[dict]) -> int | None:
    """The most common y, or None for a tie (a tie has no majority, so any final label counts as a change)."""
    top = Counter(o["y"] for o in opinions).most_common(2)
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None


def agreed(opinions: list[dict]) -> bool:
    return len({o["y"] for o in opinions}) == 1 and not any(o.get("unsure") for o in opinions)


def adjudicated(earlier: list[dict], adj: list[dict]) -> dict:
    """Rule 4."""
    everyone = earlier + adj
    if agreed(adj):
        return {"y": adj[0]["y"], "status": "adjudicated", "by": [o["by"] for o in everyone]}
    m = majority(everyone)
    return {"y": 1 if m is None else m, "status": "contested", "by": [o["by"] for o in everyone]}


def facts_for(folder: Path, text: str, rid: str | None = None) -> list[str]:
    path = folder / "facts.json"
    if not path.exists() or "\nNext call:\n" not in text:
        return []
    call = text.split("\nNext call:\n")[1]
    return [f["fact"] for f in json.loads(path.read_text())
            if re.search(f["pattern"], call) and ("ids" not in f or rid in f["ids"])]


def decide(folder: Path, sample_of: dict[str, str], texts: dict[str, str] | None = None) -> dict:
    """Apply the rules to what exists so far. Returns {final, queues, counts}; queues hold the ids still to do.
    texts (id -> state) is needed for rule 6."""
    r1, rev, adj = read(folder / "round1"), read(folder / "review"), read(folder / "adjudicate")
    texts = texts or {}
    ids = sorted(r1)
    thin = [i for i in ids if len(r1[i]) < 2]
    if thin:
        raise SystemExit(f"{len(thin)} items have fewer than 2 labellers, for example {thin[0]}")
    ok = [i for i in ids if agreed(r1[i])]
    disputed = [i for i in ids if not agreed(r1[i])]
    audit = sorted(random.Random(SEED).sample(ok, max(1, round(AUDIT * len(ok))))) if ok else []
    audited = [i for i in audit if len(rev[i]) >= 2]
    overturned = [i for i in audited if agreed(rev[i]) and rev[i][0]["y"] != r1[i][0]["y"]]
    second_round = bool(audited) and len(audited) == len(audit) and len(overturned) > MAX_OVERTURN * len(audited)
    to_review = set(disputed) | set(audit) | (set(ok) if second_round else set())

    by_fact = {i for i in ids if facts_for(folder, texts.get(i, ""), i) and any(o["y"] == 1 for o in r1[i] + rev[i])}
    final, need_review, need_adj = {}, [], []
    for i in ids:
        if i in by_fact:
            if len(adj[i]) < 2:
                need_adj.append(i)
            else:
                final[i] = adjudicated(r1[i] + rev[i], adj[i])
        elif i not in to_review:
            final[i] = {"y": r1[i][0]["y"], "status": "agreed", "by": [o["by"] for o in r1[i]]}
        elif len(rev[i]) < 2:
            need_review.append(i)
        elif agreed(rev[i]):
            final[i] = {"y": rev[i][0]["y"], "status": "reviewed", "by": [o["by"] for o in r1[i] + rev[i]]}
        elif len(adj[i]) < 2:
            need_adj.append(i)
        else:
            final[i] = adjudicated(r1[i] + rev[i], adj[i])
    for stage, status in LATER_STAGES:   # rules 7 and 8, in order
        for i, ops in read(folder / stage).items():
            if i in final and len(ops) >= 2:
                earlier = final[i]["by"]
                final[i] = adjudicated([], ops) | {"by": earlier + [o["by"] for o in ops]}
                final[i]["status"] = status if agreed(ops) else "contested"
    for i, f in final.items():
        f |= {"id": i, "sample": sample_of.get(i, "unknown"), "unsure": f["status"] == "contested"}
    changed = sum(final[i]["y"] != majority(r1[i]) for i in final)
    counts = {"items": len(ids), "agreed_round1": len(ok), "disputed_round1": len(disputed), "audit": len(audit),
              "audit_overturned": len(overturned), "second_round": second_round,
              "final": dict(Counter(f["status"] for f in final.values())), "final_positive": sum(f["y"] for f in final.values()), "by_fact": len(by_fact),
              "changed_from_round1_majority": changed, "need_review": len(need_review), "need_adjudication": len(need_adj)}
    return {"final": final, "queues": {"review": need_review, "adjudicate": need_adj}, "counts": counts}


def pairwise_kappa(folder: Path) -> dict:
    """Cohen's kappa between the labeller files of round 1, on the items both labelled, per pair of labeller names
    (the part of the file name before its last character: A0..A3 -> A)."""
    r1 = read(folder / "round1")
    by_set = defaultdict(dict)
    for i, ops in r1.items():
        for o in ops:
            by_set[o["file"][:-1]][i] = o["y"]
    names, out = sorted(by_set), {}
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            common = sorted(set(by_set[a]) & set(by_set[b]))
            pairs = [(by_set[a][i], by_set[b][i]) for i in common]
            n = len(pairs)
            if not n:
                continue
            po = sum(x == y for x, y in pairs) / n
            pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            out[f"{a}-{b}"] = {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
    return out


def write(folder: Path, result: dict, texts: dict[str, str], out: Path) -> None:
    """The review and adjudication queues (with every opinion so far), and the consensus file of decided items."""
    r1, rev = read(folder / "round1"), read(folder / "review")
    keep = ("y", "risky", "explicit", "why", "unsure", "by")
    for name, ids in result["queues"].items():
        with (folder / f"queue-{name}.jsonl").open("w") as f:
            for i in ids:
                ops = r1[i] + (rev[i] if name == "adjudicate" else [])
                f.write(json.dumps({"id": i, "text": texts[i], "facts": facts_for(folder, texts[i], i),
                                    "opinions": [{k: o.get(k) for k in keep} for o in ops]}, ensure_ascii=False) + "\n")
    with out.open("w") as f:
        for i in sorted(result["final"]):
            r = result["final"][i]
            f.write(json.dumps({"id": i, "y": r["y"], "status": r["status"], "by": "+".join(sorted(set(r["by"]))),
                                "sample": r["sample"], "unsure": r["unsure"]}, ensure_ascii=False) + "\n")
