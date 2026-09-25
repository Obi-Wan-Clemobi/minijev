"""The evaluation and calibration data: sources, fixed splits, and checked loading.

    uv run python data.py build     download the sources, build datasets/splits_v2.json (once; refuses to overwrite)
    uv run python data.py check     verify every claim that docs/DATA.md makes, from the files; exits 1 on failure

Rules that make the data defensible:
- Every item keeps its index in the source split and a SHA-256 of its row; loading re-checks both.
- The splits are drawn once, from a fixed seed, stratified by label. They never change after creation.
- BoolQ: all questions about the same passage go to the same split (grouped), so no test passage is seen in training.
- AG News: exact duplicate texts are removed before sampling.
- train: fit calibration. val: choose between methods. test: report only. load_split() records every access, and
  results files store that record, so anyone can see which split a number came from.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA, SPLITS = HERE / "data", HERE / "datasets" / "splits_v2.json"
SEED = 2026
SIZES = {  # per split
    "boolq": {"train": 500, "val": 200, "test": 300},
    "ag_news": {"train": 600, "val": 200, "test": 400},  # divisible by 4: equal topics in every split
}
SOURCES = {
    "boolq": {"hf_id": "google/boolq", "config": "default", "split": "validation", "label": "answer",
              "labels": {"0": "false", "1": "true"}, "file": "boolq_validation.json"},
    "ag_news": {"hf_id": "fancyzhx/ag_news", "config": "default", "split": "test", "label": "label",
                "labels": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}, "file": "agnews_test_full.json"},
}
ACCESS: list[dict] = []  # every load_split() call in this process: (dataset, split, n)
_TUNING = [False]


class tuning:
    """Context for choosing anything (a template, a mode, a threshold): reading the test split inside it raises.

        with data.tuning():
            rows = data.load_split("boolq", "val")    # allowed
            rows = data.load_split("boolq", "test")   # raises PermissionError
    """

    def __enter__(self):
        _TUNING[0] = True

    def __exit__(self, *exc):
        _TUNING[0] = False


def row_sha(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str) -> list[dict]:
    """The full source split, cached in data/. Pages of 100 rows from the Hugging Face datasets-server API."""
    from experiments import fetch  # certificate-checked download

    src = SOURCES[name]
    path = DATA / src["file"]
    if not path.exists():
        rows, offset = [], 0
        while True:
            url = (f"https://datasets-server.huggingface.co/rows?dataset={src['hf_id']}&config={src['config']}"
                   f"&split={src['split']}&offset={offset}&length=100")
            page_path = DATA / f"{name}_page_{offset}.json"
            for attempt in range(6):  # the API rate-limits bursts (HTTP 429): wait and retry
                try:
                    fresh = not page_path.exists()
                    page = json.loads(fetch(url, page_path).read_text())
                    if fresh:
                        time.sleep(0.5)
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429 or attempt == 5:
                        raise
                    time.sleep(2 ** attempt * 5)
            rows += [r["row"] for r in page["rows"]]
            offset += 100
            if offset >= page["num_rows_total"]:
                break
        path.write_text(json.dumps(rows, ensure_ascii=False))
    return json.loads(path.read_text())


def label_of(name: str, row: dict) -> int:
    return int(row[SOURCES[name]["label"]])


def build() -> None:
    if SPLITS.exists():
        sys.exit(f"{SPLITS} exists. The splits are frozen; delete the file only to start a new, separate version.")
    rng = random.Random(SEED)
    out = {"version": "v2", "created": time.strftime("%Y-%m-%d"), "seed": SEED, "datasets": {}}
    for name, sizes in SIZES.items():
        rows = download(name)
        src = SOURCES[name]
        if name == "boolq":
            # Group by passage: every question about one passage stays in one split.
            groups = defaultdict(list)
            for i, r in enumerate(rows):
                groups[r["passage"]].append(i)
            units = list(groups.values())
            rng.shuffle(units)
            # Each passage group goes to the split that has room for it and whose "yes" rate stays closest to the
            # full-set rate. Groups that fit nowhere are left out. Groups are never split.
            full = sum(label_of(name, r) for r in rows) / len(rows)
            splits = {split: [] for split in sizes}
            yes = {split: 0 for split in sizes}
            for g in units:
                g_yes = sum(label_of(name, rows[i]) for i in g)
                room = [sp for sp in sizes if len(splits[sp]) + len(g) <= sizes[sp]]
                if not room:
                    if all(len(splits[sp]) == sizes[sp] for sp in sizes):
                        break
                    continue
                best = min(room, key=lambda sp: (abs((yes[sp] + g_yes) / (len(splits[sp]) + len(g)) - full),
                                                 len(splits[sp]) / sizes[sp]))
                splits[best] += g
                yes[best] += g_yes
            rule = "grouped by passage: all questions about one passage are in one split; label rate kept within 5 points of the full split"
        else:
            seen, by_label = set(), defaultdict(list)
            for i, r in enumerate(rows):
                key = " ".join(r["text"].split()).lower()
                if key in seen:
                    continue  # exact duplicate text: drop
                seen.add(key)
                by_label[label_of(name, r)].append(i)
            for idx in by_label.values():
                rng.shuffle(idx)
            splits, pos = {}, {k: 0 for k in by_label}
            for split, n in sizes.items():
                per = n // len(by_label)
                splits[split] = []
                for k in sorted(by_label):
                    splits[split] += by_label[k][pos[k]:pos[k] + per]
                    pos[k] += per
                rng.shuffle(splits[split])
            rule = "exact-duplicate texts removed; equal items per topic in every split"
        out["datasets"][name] = {
            "source": {**{k: src[k] for k in ("hf_id", "config", "split")}, "rows": len(rows),
                       "file": f"data/{src['file']}", "file_sha256": file_sha(DATA / src["file"]),
                       "fetched": time.strftime("%Y-%m-%d"), "api": "datasets-server.huggingface.co/rows"},
            "labels": src["labels"], "sampling": rule,
            "splits": {split: [{"i": i, "sha256": row_sha(rows[i]), "y": label_of(name, rows[i])} for i in idx]
                       for split, idx in splits.items()},
        }
    SPLITS.parent.mkdir(exist_ok=True)
    SPLITS.write_text(json.dumps(out, indent=1))
    print(f"wrote {SPLITS}")
    check()


def load_split(name: str, split: str) -> list[dict]:
    """The rows of one split, each with its label as 'y'. Fails loudly if any row differs from the frozen split."""
    if split == "test" and _TUNING[0]:
        raise PermissionError(f"{name}/test was read inside data.tuning(): the test split is for reporting only")
    spec = json.loads(SPLITS.read_text())["datasets"][name]
    rows = download(name)
    out = []
    for item in spec["splits"][split]:
        row = rows[item["i"]]
        if row_sha(row) != item["sha256"]:
            raise ValueError(f"{name}/{split}: source row {item['i']} changed since the split was frozen "
                             f"({spec['source']['fetched']}). The results would not be comparable.")
        out.append({**row, "y": item["y"], "source_index": item["i"]})
    ACCESS.append({"dataset": name, "split": split, "n": len(out)})
    return out


def splits_fingerprint() -> str:
    return file_sha(SPLITS)[:16]


def summary() -> dict:
    """Facts about the data, computed from the files. docs/DATA.md and the Data page show these."""
    spec = json.loads(SPLITS.read_text())
    out = {"version": spec["version"], "created": spec["created"], "seed": spec["seed"],
           "splits_sha256": file_sha(SPLITS), "datasets": {}}
    for name, d in spec["datasets"].items():
        rows = download(name)
        per = {}
        for split, items in d["splits"].items():
            labels = Counter(d["labels"][str(it["y"])] for it in items)
            if name == "boolq":
                texts = [rows[it["i"]]["passage"] for it in items]
                q = [len(rows[it["i"]]["question"].split()) for it in items]
            else:
                texts = [rows[it["i"]]["text"] for it in items]
                q = []
            words = sorted(len(t.split()) for t in texts)
            per[split] = {"n": len(items), "labels": dict(sorted(labels.items())),
                          "words_median": words[len(words) // 2], "words_max": words[-1],
                          **({"question_words_median": sorted(q)[len(q) // 2]} if q else {})}
        out["datasets"][name] = {"source": d["source"], "labels": d["labels"], "sampling": d["sampling"], "splits": per}
    return out


def check(results: list | None = None, quiet: bool = False) -> bool:
    """Verify the claims of docs/DATA.md against the files. Prints each check (unless quiet) and appends
    {"dataset", "claim", "passed"} to results when given; returns True when all pass."""
    spec = json.loads(SPLITS.read_text())
    ok = True
    current = {"name": ""}

    def claim(text: str, passed: bool) -> None:
        nonlocal ok
        ok &= passed
        if results is not None:
            results.append({"dataset": current["name"], "claim": text, "passed": bool(passed)})
        if not quiet:
            print(f"  {'PASS' if passed else 'FAIL'}  {text}")

    for name, d in spec["datasets"].items():
        current["name"] = name
        if not quiet:
            print(name)
        path = HERE / d["source"]["file"]
        claim(f"source file matches its frozen SHA-256 ({path.name})", file_sha(path) == d["source"]["file_sha256"])
        rows = json.loads(path.read_text())
        claim(f"source has {d['source']['rows']} rows", len(rows) == d["source"]["rows"])
        idx = {s: [it["i"] for it in items] for s, items in d["splits"].items()}
        claim("every split item's row matches its SHA-256", all(row_sha(rows[it["i"]]) == it["sha256"]
                                                              for items in d["splits"].values() for it in items))
        claim("every stored label matches the source row", all(label_of(name, rows[it["i"]]) == it["y"]
                                                               for items in d["splits"].values() for it in items))
        claim(f"split sizes are {SIZES[name]}", {s: len(v) for s, v in idx.items()} == SIZES[name])
        sets = {s: set(v) for s, v in idx.items()}
        claim("train, val and test share no source row", not (sets["train"] & sets["val"] or sets["train"] & sets["test"]
                                                              or sets["val"] & sets["test"]))
        key = (lambda r: r["passage"]) if name == "boolq" else (lambda r: " ".join(r["text"].split()).lower())
        texts = {s: {key(rows[i]) for i in v} for s, v in idx.items()}
        claim("no " + ("passage" if name == "boolq" else "text") + " appears in two splits",
              not (texts["train"] & texts["test"] or texts["val"] & texts["test"] or texts["train"] & texts["val"]))
        if name == "boolq":
            full = sum(label_of(name, r) for r in rows) / len(rows)
            rates = {s: sum(it["y"] for it in items) / len(items) for s, items in d["splits"].items()}
            claim(f"'yes' rate of every split is within 5 points of the full split ({full:.3f}): "
                  + ", ".join(f"{s} {r:.3f}" for s, r in rates.items()), all(abs(r - full) <= 0.05 for r in rates.values()))
        else:
            counts = {s: Counter(it["y"] for it in items) for s, items in d["splits"].items()}
            claim("every split has the same number of items per topic", all(len(set(c.values())) == 1 for c in counts.values()))
    if not quiet:
        print("ALL CLAIMS HOLD" if ok else "SOME CLAIMS FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    elif cmd == "check":
        sys.exit(0 if check() else 1)
    elif cmd == "summary":
        print(json.dumps(summary(), indent=1))
    else:
        sys.exit(__doc__)
