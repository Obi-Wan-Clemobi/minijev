"""LoRA fine-tune on the frozen train splits, with option-shuffle augmentation (PLAN Task 6.2, hard-label variant).

    uv run --group train python train_lora.py parity            training forward == cached readout (train items)
    uv run --group train python train_lora.py time --steps 20   seconds per example, before a long run
    uv run --group train python train_lora.py train             the run; one checkpoint per epoch in adapters/
    uv run --group train python train_lora.py train --task sst5  E22: a per-task adapter for SST-5 Scores only

One adapter for both question types, as Jev uses one set of weights (RESEARCH.md row 7):
- Noul: BoolQ train (500), the production Noul prompt, target Yes or No.
- Choice: AG News train (600), the production listwise prompt. Each item appears in ORDERS random option orders per
  epoch, and the target is the letter where the true option now sits. The model sees every topic at every letter.

--task sst5 (E22): a per-task adapter. SST-5 train (300), the production listwise Score prompt, target the letter of
the true level. Levels are an ordered scale and always appear in order, so there is no shuffle. Same HYPER.

Loss: log loss on the readout. The class logit is the logsumexp over the label variants at the last prompt token,
exactly as minijev.primitives.class_logits builds it; the loss is cross-entropy over those class logits. Log loss is
a strictly proper scoring rule, so training against labels also trains calibration (DESIGN.md §7).

Data use: train is trained on. val chooses the epoch (inside data.tuning(), so a test read raises). test is read only
by experiments.py lora. The hyperparameters below were fixed before the first run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

import data
from minijev.engine import Engine
from minijev.fixtures import AG_OPTIONS, AG_QUESTION, SST5_LEVELS, SST5_QUESTION
from minijev.prompt import choice_block, noul_block, score_listwise_block
from minijev.settings import MODEL

HERE = Path(__file__).parent
ADAPTERS = HERE / "adapters"
HYPER = {  # fixed before the first run; not tuned
    "rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lr": 2e-4, "warmup_fraction": 0.05, "schedule": "linear decay", "accumulate": 8, "epochs": 2,
    "orders_per_epoch": 2, "seed": 2026, "weight_decay": 0.0,
}


def boolq_question(r: dict) -> dict:
    return {"type": "noul", "instructions": r["question"][0].upper() + r["question"][1:] + "?"}


def choice_question(order: list[int]) -> dict:
    """The AG News question with letter position p showing option order[p], as experiments.readout_cache does."""
    keys = list(AG_OPTIONS)
    return {"type": "choice", "instructions": AG_QUESTION, "criteria": {keys[j]: AG_OPTIONS[keys[j]] for j in order}}


def example(engine: Engine, kind: str, r: dict, order: list[int] | None = None) -> dict:
    """Token ids (state and question tokenized separately, as at inference), label classes and the target class."""
    if kind == "sst5":
        q = {"type": "score", "instructions": SST5_QUESTION, "criteria": SST5_LEVELS}
        return {"ids": engine.prefix_ids(r["text"]) + engine.suffix_ids(score_listwise_block(q)),
                "classes": engine.letters[:len(SST5_LEVELS)], "target": r["y"], "kind": kind}
    if kind == "boolq":
        return {"ids": engine.prefix_ids(r["passage"]) + engine.suffix_ids(noul_block(boolq_question(r))),
                "classes": [engine.yes, engine.no], "target": 0 if r["y"] else 1, "kind": kind}
    return {"ids": engine.prefix_ids(r["text"]) + engine.suffix_ids(choice_block(choice_question(order))),
            "classes": engine.letters[:len(order)], "target": order.index(r["y"]), "kind": kind}


def class_logits(engine: Engine, ex: dict) -> torch.Tensor:
    """The readout's class logits for one example, with gradients when the model is in training."""
    logits = engine.model(torch.tensor([ex["ids"]]), logits_to_keep=1).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    return torch.stack([torch.logsumexp(lp[ids], 0) for ids in ex["classes"]])


def loss_of(engine: Engine, ex: dict) -> torch.Tensor:
    return -torch.log_softmax(class_logits(engine, ex), -1)[ex["target"]]


def train_stream(engine: Engine, epoch: int, orders: int, task: str = "mixed") -> list[dict]:
    """With task "sst5": every SST-5 train item once, shuffled. Otherwise: One epoch: every BoolQ train item once, every AG News train item in `orders` fresh random orders, mixed.
    The order RNG is seeded by epoch and item, independent of the per-item orders the test readouts use."""
    rng = random.Random(f"{HYPER['seed']}-epoch-{epoch}")
    if task == "sst5":
        out = [example(engine, "sst5", r) for r in data.load_split("sst5", "train")]
        rng.shuffle(out)
        return out
    out = [example(engine, "boolq", r) for r in data.load_split("boolq", "train")]
    for r in data.load_split("ag_news", "train"):
        for _ in range(orders):
            out.append(example(engine, "ag_news", r, rng.sample(range(len(AG_OPTIONS)), len(AG_OPTIONS))))
    rng.shuffle(out)
    return out


def target_balance(stream: list[dict]) -> dict:
    """Share of each target letter among the Choice examples. About 0.25 each, or the training teaches a position."""
    t = [ex["target"] for ex in stream if ex["kind"] == "ag_news"]
    return {"ABCD"[k]: t.count(k) / len(t) for k in range(4)}


@torch.no_grad()
def val_nll(engine: Engine, task: str = "mixed") -> dict:
    """Mean log loss on val (BoolQ Noul; AG News listwise in the per-item order the E14 readouts use), or on SST-5
    val (listwise Score) for task "sst5"."""
    engine.model.eval()
    if task == "sst5":
        with data.tuning():
            exs = [example(engine, "sst5", r) for r in data.load_split("sst5", "val")]
        out = {"sst5": sum(loss_of(engine, ex).item() for ex in exs) / len(exs)}
        return out | {"mean": out["sst5"]}
    with data.tuning():
        bq = [example(engine, "boolq", r) for r in data.load_split("boolq", "val")]
        ag = [example(engine, "ag_news", r, random.Random(r["source_index"]).sample(range(4), 4))
              for r in data.load_split("ag_news", "val")]
    out = {k: sum(loss_of(engine, ex).item() for ex in exs) / len(exs) for k, exs in (("boolq", bq), ("ag_news", ag))}
    out["mean"] = (out["boolq"] + out["ag_news"]) / 2
    return out


def lora_engine(model: str, attn: str, threads: int) -> Engine:
    from peft import LoraConfig, get_peft_model
    engine = Engine(model, attn=attn, threads=threads)
    cfg = LoraConfig(r=HYPER["rank"], lora_alpha=HYPER["alpha"], lora_dropout=HYPER["dropout"],
                     target_modules=HYPER["target_modules"], task_type="CAUSAL_LM")
    engine.model = get_peft_model(engine.model, cfg)
    return engine


def parity(args) -> None:
    """The training forward (no adapter) gives the same class logits as the cached E14 readouts."""
    engine = Engine(args.model, attn=args.attn, threads=args.threads)
    run = args.model.split("/")[-1]
    cache = {d: json.loads((HERE / "results" / "readouts" / run / f"{d}_train.json").read_text())["items"]
             for d in ("boolq", "ag_news")}
    worst = 0.0
    with torch.no_grad():
        for r, it in list(zip(data.load_split("boolq", "train"), cache["boolq"]))[:5]:
            z = class_logits(engine, example(engine, "boolq", r))
            worst = max(worst, abs((z[0] - z[1]).item() - it["z"]))
        for r, it in list(zip(data.load_split("ag_news", "train"), cache["ag_news"]))[:5]:
            z = class_logits(engine, example(engine, "ag_news", r, it["order"])).tolist()
            canon = [z[it["order"].index(c)] for c in range(4)]  # back to the fixed label order, as the cache stores it
            worst = max(worst, max(abs(a - b) for a, b in zip(canon, it["listwise"])))
    print(f"largest difference to the cached readout: {worst:.2e}")
    assert worst < 1e-3, "the training forward does not match the readout"


def run(args, max_steps: int | None = None) -> None:
    torch.manual_seed(HYPER["seed"])
    engine = lora_engine(args.model, args.attn, args.threads)
    out = ADAPTERS / (args.model.split("/")[-1] + ("" if args.task == "mixed" else f"-{args.task}"))
    streams = [train_stream(engine, e, HYPER["orders_per_epoch"], args.task) for e in range(HYPER["epochs"])]
    if args.task == "sst5":  # the splits are stratified: 60 items per level
        balance = {"ABCDE"[k]: sum(ex["target"] == k for ex in streams[0]) / len(streams[0]) for k in range(5)}
    else:
        balance = target_balance(streams[0])
        assert all(abs(v - 0.25) < 0.05 for v in balance.values()), "target letters are not balanced"
    print("target letters (epoch 0):", {k: round(v, 3) for k, v in balance.items()})

    params = [p for p in engine.model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=HYPER["lr"], weight_decay=HYPER["weight_decay"])
    total = sum(len(s) for s in streams) // HYPER["accumulate"]
    warm = max(1, int(HYPER["warmup_fraction"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

    log = {"hyper": HYPER, "task": args.task, "model": args.model, "attn": args.attn, "target_balance_epoch0": balance,
           "examples_per_epoch": len(streams[0]), "optimizer_steps": total, "epochs": []}
    start_epoch = 0
    state = out / "state.pt"
    if state.exists() and max_steps is None:  # resume after the last finished epoch
        saved = torch.load(state, weights_only=True)
        engine.model.load_state_dict(saved["model"], strict=False)
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        log, start_epoch = saved["log"], saved["epoch"] + 1
        print(f"resumed after epoch {saved['epoch']}")

    t0, step = time.perf_counter(), 0
    for epoch in range(start_epoch, HYPER["epochs"]):
        engine.model.train()
        losses, running = [], 0.0
        for i, ex in enumerate(streams[epoch]):
            loss = loss_of(engine, ex)
            (loss / HYPER["accumulate"]).backward()
            running += loss.item()
            if (i + 1) % HYPER["accumulate"] == 0:
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                losses.append(running / HYPER["accumulate"])
                running = 0.0
                if step % 10 == 0 or max_steps:
                    per = (time.perf_counter() - t0) / ((i + 1) + (epoch - start_epoch) * len(streams[0]))
                    print(f"epoch {epoch} step {step}/{total}  loss {statistics_tail(losses):.3f}  "
                          f"{per:.2f} s/example", flush=True)
                if max_steps and step >= max_steps:
                    print(f"{per:.2f} s per example; a full run is {per * sum(len(s) for s in streams) / 3600:.1f} h "
                          f"of training plus the val passes")
                    return
        val = val_nll(engine, args.task)
        print(f"epoch {epoch} val NLL: " + "  ".join(f"{k} {v:.3f}" for k, v in val.items()), flush=True)
        engine.model.save_pretrained(out / f"epoch-{epoch}")
        log["epochs"].append({"epoch": epoch, "train_loss_per_step": losses, "val_nll": val,
                              "seconds": time.perf_counter() - t0})
        torch.save({"model": {k: v for k, v in engine.model.state_dict().items() if "lora_" in k},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "log": log, "epoch": epoch}, state)
        (out / "train_log.json").write_text(json.dumps(log, indent=1))

    best = min(log["epochs"], key=lambda e: e["val_nll"]["mean"])
    log["selected_epoch"] = best["epoch"]
    log["selection_rule"] = ("lowest val NLL (SST-5 listwise Score)" if args.task == "sst5" else
                             "lowest mean val NLL (BoolQ, AG News listwise)") + ", inside data.tuning()"
    log["splits_accessed"] = list(data.ACCESS)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"selected epoch {best['epoch']} (val NLL {best['val_nll']['mean']:.3f}); adapter in {out}/epoch-{best['epoch']}")


def statistics_tail(xs: list[float], n: int = 10) -> float:
    tail = xs[-n:]
    return sum(tail) / len(tail) if tail else math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["parity", "time", "train"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--task", default="mixed", choices=["mixed", "sst5"],
                    help="mixed: BoolQ + AG News (E20). sst5: a per-task adapter for SST-5 Scores (E22)")
    args = ap.parse_args()
    if args.command == "parity":
        parity(args)
    else:
        run(args, args.steps if args.command == "time" else None)


if __name__ == "__main__":
    main()
