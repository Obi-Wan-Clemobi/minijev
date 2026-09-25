# minijev

minijev is a small, local, open re-creation of the *idea* behind TypeSafe's Jev: text in, typed and calibrated
probabilistic decisions out. It runs a small open model (Qwen2.5, 0.5B or 1.5B parameters) on a laptop CPU. It is a
learning project: we built it to understand the mechanism and to measure it. It is not a copy of Jev.

## Objectives

1. **Understand the mechanism.** A normal LLM generates its answer token by token, and code parses the text. minijev
   does a **readout**: one forward pass, then it takes the probabilities of the allowed answer labels (`Yes`/`No`,
   `A`/`B`/`C`). No tokens are generated.
2. **Measure it honestly.** Every claim has an evidence label: **Measured** (our runs), **Observed**, **Stated** (a
   source says it), or **Inferred** (our reasoning). Defended numbers come from frozen held-out test splits.
3. **Find the weaknesses, then fix them one at a time.** [docs/WEAKNESSES.md](docs/WEAKNESSES.md) is the register.
4. **Make it visible.** A local web app shows every step, with help text in plain words.

## How it works

A request has a **state** (the input text) and one or more **questions**. There are three question types:

| Type | Example | Answer |
|---|---|---|
| Noul | "Is this message urgent?" | P(yes) |
| Choice | "Which team: billing, technical or sales?" | one option, with a probability for each option |
| Score | "How upset: calm … angry?" | the expected level, with a probability for each level |

The model does a **prefill** of the state once. Each question is a **branch** after the state, and it sees only the
state and itself. The **packed** mode runs the state and all branches in one forward pass, with an attention mask
between the branches. A fitted temperature then makes the probabilities honest (**calibration**). The terms are
defined in the glossary in [CLAUDE.md](CLAUDE.md).

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Node.js 20 or newer. The model runs on the CPU; no GPU is necessary.

1. Run `./setup.sh --model`. It installs the Python and web dependencies and downloads Qwen2.5-0.5B-Instruct
   (about 1 GB).
2. Run `tilt up`. It starts the API on port 8000 and the web app on port 3000. Install Tilt with
   `brew install tilt-dev/tap/tilt` if you do not have it.
3. Open http://localhost:3000.

Without Tilt, use two terminals:
1. Run `cd poc && uv run minijev serve --port 8000`.
2. Run `cd web && npm run dev`.

From the command line: `cd poc && uv run minijev ask "Hi, my card was charged twice" "Is this urgent?"`.
The model and the calibration dials are set in `poc/minijev.env`.

## The web app

| Page | What you can do |
|---|---|
| Playground | Write a state and questions, run them, and turn the calibration dials. The dials change the answers at once, with no new model run. |
| Compare | Compare a readout with the same model generating its answer: speed, format errors and quality. |
| State machine | Chain questions into a flow: each answer picks the next step. Build it on a canvas, run it on a request, and watch each step decide. |
| Under the hood | See the prefix tree, the attention mask and the tokens of each branch. |
| Findings | Charts of the measurements, from `poc/results/*.json`, including the LoRA fine-tune. |
| Data | The data card, with its checks run again on each load. |
| Weaknesses | The weakness register, with filters. |

[web/README.md](web/README.md) describes each page in more detail.

## What we have built so far

Each item lists its main measured result. The experiment numbers (E1–E20) match
[docs/RESEARCH.md](docs/RESEARCH.md) §7.3 and [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

1. **The readout engine and the playground** (E1–E13, exploratory). Three evaluation modes (naive, kv, packed) give
   the same answers. On this CPU, a generated token costs about 51× a prefilled token, and a readout generates none.
2. **A measurement foundation** (PLAN.md Phase 1). Frozen train/val/test splits of BoolQ, AG News and SST-5, with a
   SHA-256 per row. `data.py check` verifies every claim of the data card.
3. **Held-out calibration** (E14). Temperatures are fitted on train, chosen on val, and reported on test. On BoolQ
   test at 0.5B, the calibration error (ECE) falls from 0.160 to 0.100. Accuracy is 0.693 at 0.5B and 0.773 at 1.5B.
4. **An installable package** (`src/minijev`): the engine, the prompts, the readout, calibration, the HTTP API and a
   CLI.
5. **Foundation work** (PLAN.md Phase 4):
   - **Two-level tree:** 21.6% fewer tokens for Score questions.
   - **State cache:** 1.2–2.9× faster when a state repeats.
   - **Template sensitivity (E15):** 81 prompt wordings compared.
   - **Opposite questions (E17):** an opt-in step that makes them sum to 1.
   - **Criteria library (E18):** for vague questions.
   - **Long states (E19):** tested up to 8k tokens.
   - **Contrastive levels (E16):** tested and rejected.
6. **A LoRA fine-tune** (E20). LoRA trains a small add-on (about 1 M weights) on the train splits, with the options
   shuffled. On test at 0.5B:
   - BoolQ accuracy goes from 0.693 to 0.770, the level of the base 1.5B model.
   - Answer flips when only the option order changes go from 22% to 9%.
   - The training took 2.4 h on the CPU.
7. **The State machine page.** Flows of steps, with confidence conditions on the arrows, 2 templates, save and reload,
   and export and import. One step takes 0.6–1.4 s at 0.5B.

[PLAN.md](PLAN.md) has the task tracker. It lists what is done and what is blocked: a Claude baseline needs an API key,
and larger models need a GPU.

## Run the experiments

All commands run from `poc/`. Results go to `poc/results/`.

| Command | What it does |
|---|---|
| `uv run python experiments.py demo` | One request, answered in all three modes |
| `uv run python experiments.py heldout` | E14: fit on train, choose on val, report on test; writes `calibration/<model>.json` |
| `uv run python experiments.py <name> --model Qwen/Qwen2.5-1.5B-Instruct` | Any experiment on the larger model |
| `uv run python data.py check` | Verify every claim of the data card |
| `uv run --group train python train_lora.py train` | E20: train the LoRA adapter (about 2.4 h at 0.5B on a CPU) |
| `uv run --group train python experiments.py lora --adapter adapters/Qwen2.5-0.5B-Instruct/epoch-1` | E20: base model vs adapter on test |

`uv run python experiments.py --help` lists every experiment.

## Checks

1. Run `cd poc && uv run pytest -m "not model"`. These are the fast tests; they need no model.
2. Run `cd poc && uv run --group train pytest`. This runs all tests, and it loads the 0.5B model.
3. Run `cd web && npm test && npm run lint && npx tsc --noEmit`.

## Repository map

| Path | Contents |
|---|---|
| [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | What we built and measured, step by step. **Start here.** |
| [docs/DESIGN.md](docs/DESIGN.md) | The design |
| [docs/RESEARCH.md](docs/RESEARCH.md) | What is known about Jev, with sources and evidence levels, and our results |
| [docs/DATA.md](docs/DATA.md) | The data card: sources, licences, frozen splits, and what each split is for |
| [docs/WEAKNESSES.md](docs/WEAKNESSES.md) | Every known weakness, with evidence and a candidate fix |
| [PLAN.md](PLAN.md) | The implementation plan and task tracker |
| [src/minijev/](src/minijev/) | The package, the HTTP API (`api/`) and the flow executor (`flows.py`) |
| [poc/](poc/) | Experiments, data, splits, results, calibration files, the LoRA adapter and flow templates |
| [web/](web/) | The web app (Next.js) |
| [CLAUDE.md](CLAUDE.md) | Writing rules and the glossary |

## Limits

- **Small models.** 0.5B is weak on some tasks; 1.5B is better but about 2× slower. Latency is seconds, not Jev's
  100 ms (DESIGN.md §8).
- **Narrow data.** The held-out results cover English Wikipedia questions, 2004–2005 news and movie-review sentences.
  A temperature or an adapter fitted here may not hold for other text.
- **Licences.** AG News is for non-commercial research, and SST-5 has no stated licence. The LoRA adapter was trained
  on AG News, so it is for research only.
