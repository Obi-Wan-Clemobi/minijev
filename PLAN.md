# minijev — Implementation Plan

*Implementation roadmap for 19 documented weaknesses. Based on research initiative output (10 researchers + 20 Opus peer reviews). See `docs/SOLUTIONS.md` for full analysis.*

## Status: Ready for Implementation

**Priority:** Fix measurement foundation first (Phase 1), then deploy quick wins (Phase 2), then build foundation work.

**Context:** All solutions are evidence-based from committed experiments (E1-E13). No solution requires resources we lack except where explicitly noted (GPU, API keys).

---

## Phase 1: Measurement Foundation (Week 1-2)
**Blocker:** Current experiments have contaminated templates, unpinned datasets, and wide confidence intervals. Nothing can be declared "fixed" without closing W15-W18 first.

### Task 1.1: Pin Dataset Versions (W17)
**Owner:** TBD | **Effort:** 1 day | **Priority:** Critical

**Goal:** Eliminate silent dataset drift that would invalidate all comparisons.

**Implementation:**
1. Read current dataset loading in `poc/experiments.py` (lines that fetch BoolQ/AG News from HF)
2. Add manifest file `poc/datasets/manifest.json` with schema:
   ```json
   {
     "boolq": {
       "revision": "main",
       "rows": 400,
       "sha256": "per-row checksums array",
       "fetched": "2025-09-24"
     }
   }
   ```
3. Implement `--update-manifest` flag that:
   - Downloads datasets
   - Computes per-row SHA256 (hash of serialized row dict)
   - Writes manifest.json
4. Modify dataset loaders to:
   - Verify checksums on load
   - Fail loudly with diff if mismatch detected
   - Show manifest metadata in experiment output

**Acceptance:**
- `python experiments.py --update-manifest` generates manifest
- Subsequent runs verify checksums
- Intentional dataset corruption causes clear error message
- No existing experiment results change

**Files to modify:**
- `poc/experiments.py` (dataset loading functions)
- Create `poc/datasets/manifest.json`

---

### Task 1.2: Pre-Register Template (W15)
**Owner:** TBD | **Effort:** 1 day | **Priority:** Critical

**Goal:** Document current template choices before any future changes to establish held-out validation.

**Implementation:**
1. Extract all prompt templates from `poc/minijev_poc.py`:
   - System line for Noul/Choice/Score
   - Question instruction format
   - Option presentation format
   - Label tokens used
2. Create `poc/templates/v1-preregistration.json`:
   ```json
   {
     "version": "v1",
     "locked_at": "2025-09-24",
     "sha256": "hash of this file's stable content",
     "primitives": {
       "noul": {
         "system": "exact template string",
         "format": "instruction formatting"
       },
       "choice": { ... },
       "score": { ... }
     },
     "note": "These templates were chosen with E1-E13 test cases visible. Future changes must use disjoint held-out splits."
   }
   ```
3. Create `poc/datasets/splits.json`:
   ```json
   {
     "boolq": {
       "train_indices": [0, 1, 2, ...],
       "val_indices": [...],
       "test_indices": [...],
       "note": "Indices fixed before any template tuning. Test set never used for template selection."
     }
   }
   ```
4. Modify experiments to:
   - Load splits.json
   - Assert train/val/test are disjoint
   - Report which split each result uses
   - Fail if test indices accessed during template tuning

**Acceptance:**
- Template preregistration file exists with SHA256
- Split file exists with disjoint train/val/test
- All future template changes documented in new version files
- Experiments report split membership

**Files to create:**
- `poc/templates/v1-preregistration.json`
- `poc/datasets/splits.json`

---

### Task 1.3: Improve Timing Measurement (W16)
**Owner:** TBD | **Effort:** 1 day | **Priority:** High

**Goal:** Reduce ±10-20% timing noise to make latency comparisons valid.

**Implementation:**
1. Find latency experiments in `poc/experiments.py` (E3 and others)
2. Add warm-up runs:
   ```python
   # Warm up: run once to load model, compile kernels
   for _ in range(3):
       _ = judge.ask(state, questions, settings, mode)
   
   # Measure: 5 runs with rotation to avoid order effects
   times = []
   for _ in range(5):
       start = time.perf_counter()
       _ = judge.ask(state, questions, settings, mode)
       times.append(time.perf_counter() - start)
   ```
3. Change reporting from mean to median + IQR:
   ```python
   median = np.median(times)
   q1, q3 = np.percentile(times, [25, 75])
   print(f"Latency: {median:.3f}s [Q1={q1:.3f}, Q3={q3:.3f}]")
   ```
4. Add rotation to mitigate order effects:
   - Run experiments in [A, B, A, B, A] order
   - Report per-condition median

**Acceptance:**
- All latency experiments use warm-up + 5 runs
- Results report median and IQR, not mean
- Rotation order documented in output
- E3 re-run shows narrower confidence intervals

**Files to modify:**
- `poc/experiments.py` (latency measurement code)

---

### Task 1.4: Expand Sample Sizes (W15)
**Owner:** TBD | **Effort:** 2 days | **Priority:** Medium

**Goal:** Reduce confidence interval width from ±0.06-0.08 to ±0.03-0.04 on ECE.

**Implementation:**
1. Expand BoolQ evaluation from 400 to 500 samples
2. Expand AG News from 120 to 300 samples
3. Use stratified sampling to preserve label distribution
4. Update E11 (quality) and E13 (order bias) to use expanded sets
5. Re-run experiments, verify intervals narrower
6. Update WALKTHROUGH.md with new results

**Acceptance:**
- BoolQ n=500, AG News n=300
- Bootstrap CIs reported on all metrics
- Intervals ≤±0.04 on ECE
- No change to point estimates (within old CIs)

**Files to modify:**
- `poc/experiments.py` (sample size constants)
- `docs/WALKTHROUGH.md` (update E11, E13 results)

---

## Phase 2: Quick Wins (Week 3)
**Goal:** Deploy zero-effort or low-effort fixes that are already validated.

### Task 2.1: Choose the Choice Mode on Held-Out Data (W1)
**Status:** Done (E14)

The default Choice mode is `MINIJEV_CHOICE_MODE=selected`: for each model, E14 (`experiments.py heldout`) compares
listwise, averaged and pointwise on the **val** split (AG News, 200 articles) by negative log-likelihood, and stores the
winner in `poc/calibration/<model>.json`. E13 showed that averaged and pointwise remove most order flips; the val split,
not a rule of thumb, decides between them. Any mode can still be set per question or with `MINIJEV_CHOICE_MODE`.

---

### Task 2.2: Fitted Calibration Temperatures (W2/W3)
**Status:** Done (E14) · details: docs/TASK_2_2_CALIBRATION_TEMPS.md

Every temperature is fitted on the **train** split of `poc/datasets/splits_v2.json`, chosen on **val**, and reported on
**test** (docs/DATA.md). Noul: temperature and Platt on BoolQ. Choice: one temperature per mode on AG News. Score: not
fitted (no labelled ordinal data). The values live in `poc/calibration/<model>.json` with their provenance; nothing is
estimated or typed by hand. `MINIJEV_CALIBRATION=fitted` applies them; a number in `minijev.env` overrides one primitive.

---

### Task 2.3: Calibration Controls in the Web UI (W2/W3)
**Status:** Done

The Calibration panel reads `/v1/calibration`: it shows where the values came from, a fitted/manual/raw badge per dial,
Reset to the fitted value, and the choice between "fitted on the train split" and raw. The Data page shows the data,
the live checks and the held-out results. Still open: a reliability diagram per primitive from the E14 test split.

---

## Phase 3: Package Structure (Week 4)
**Blocker:** POC is flat files. All advanced work (LoRA, caching, multi-stage) assumes package structure.

### Task 3.1: Create src/minijev Package Layout (W19)
**Owner:** TBD | **Effort:** 1 week | **Priority:** Foundation

**Goal:** Convert POC to installable package with proper structure.

**Implementation:**
1. Create directory structure:
   ```
   src/minijev/
     __init__.py
     api/          # FastAPI server
       __init__.py
       server.py   # from poc/server.py
       schema.py   # pydantic models
     judges/       # core logic
       __init__.py
       judge.py    # from poc/minijev_poc.py Judge class
     prompt/       # template system
       __init__.py
       templates.py
     engine/       # model inference
       __init__.py
       engine.py   # from poc/minijev_poc.py Engine class
     primitives/   # Noul/Choice/Score
       __init__.py
       noul.py
       choice.py
       score.py
     calibrate/    # temperature scaling
       __init__.py
       temperature.py
     cli.py        # Typer CLI
   ```

2. Create `pyproject.toml`:
   ```toml
   [project]
   name = "minijev"
   version = "0.1.0"
   requires-python = ">=3.11"
   dependencies = [
     "torch>=2.4.0",
     "transformers>=4.46.0",
     "fastapi>=0.115.0",
     "uvicorn>=0.32.0",
   ]
   
   [project.scripts]
   minijev = "minijev.cli:app"
   
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   ```

3. Migrate code:
   - Split `poc/minijev_poc.py` (412 lines) into modules
   - Split `poc/server.py` (313 lines) into api/
   - Keep `poc/experiments.py` as top-level script (or move to src/minijev/experiments.py)

4. Update all imports throughout codebase

5. Add CLI entry point:
   ```python
   # src/minijev/cli.py
   import typer
   app = typer.Typer()
   
   @app.command()
   def ask(state: str, question: str, type: str = "noul"):
       """Ask minijev a question."""
       # Implementation
   
   @app.command()
   def serve(port: int = 8000):
       """Start the API server."""
       # Implementation
   ```

6. Update Docker, docker-compose, Tiltfile

7. Run full test suite and experiment regression

**Acceptance:**
- `pip install -e .` works
- `minijev ask "text" "question?"` works
- `minijev serve` starts API
- All tests pass
- E1-E13 reproduce identical results
- Docker build succeeds
- Tilt up works

**Files to create:**
- `pyproject.toml`
- `src/minijev/` (entire structure)

**Files to modify:**
- All imports throughout codebase
- `Dockerfile`
- `docker-compose.yml`
- `Tiltfile`
- `README.md` (update quickstart)

---

## Phase 4: Foundation Work (Weeks 5-11)

### Task 4.1: Two-Level Prefix Tree (W14)
**Owner:** TBD | **Effort:** 1 week | **Priority:** High

**Goal:** Reduce token count 20-30% for pointwise Score by sharing question text.

**Implementation:**
1. Current tree structure (state → levels):
   ```
   state: "Bug report text"
   branches:
     - "Bug report text\n\nProposed answer: workaround exists (yes/no)"
     - "Bug report text\n\nProposed answer: impacts >100 users (yes/no)"
     - "Bug report text\n\nProposed answer: outage (yes/no)"
   ```

2. New three-tier tree (state → question → levels):
   ```
   state: "Bug report text"
   question_branch: "Bug report text\n\nHow severe is this bug?"
   level_branches:
     - "How severe... → workaround exists (yes/no)"
     - "How severe... → impacts >100 users (yes/no)"  
     - "How severe... → outage (yes/no)"
   ```

3. Refactor `proposal_block()` in `src/minijev/prompt/templates.py`:
   - Separate question header from level judgment
   - Return (header, level_blocks) tuple

4. Modify `pack()` in `src/minijev/engine/engine.py`:
   - Add question-level nodes to tree
   - Extend attention mask to 3 levels
   - Update position IDs for 3-tier structure

5. Extend `kv` mode to crop at two levels:
   - Crop 1: state only
   - Crop 2: state + question
   - Crop 3: state + question + level

6. Validate numerical equivalence:
   - Run same Score question in old vs new tree
   - Assert logits match within 1e-3

7. Measure token savings on GDPR Score questions

**Acceptance:**
- Three-tier tree implemented
- Token count reduced 20-30% for pointwise Score
- Logits match old implementation (within 1e-3)
- All Score tests pass
- Attention mask visualization updated in web UI

**Files to modify:**
- `src/minijev/prompt/templates.py`
- `src/minijev/engine/engine.py`
- `web/app/hood/page.tsx` (attention viz)

---

### Task 4.2: Hash-Based State Cache (W13)
**Owner:** TBD | **Effort:** 1 week | **Priority:** Medium

**Goal:** 1.2-1.6x speedup on repeated states via KV cache reuse.

**Implementation:**
1. Add LRU cache keyed by (model_name, state_token_ids):
   ```python
   from functools import lru_cache
   
   class StateCache:
       def __init__(self, max_size: int = 100):
           self.cache: dict[tuple[str, tuple[int, ...]], DynamicCache] = {}
           self.lru: list[tuple] = []
           self.max_size = max_size
       
       def get(self, model: str, tokens: list[int]) -> DynamicCache | None:
           key = (model, tuple(tokens))
           if key in self.cache:
               self.lru.remove(key)
               self.lru.append(key)
               return self.cache[key]
           return None
       
       def put(self, model: str, tokens: list[int], cache: DynamicCache):
           key = (model, tuple(tokens))
           if len(self.cache) >= self.max_size:
               evict = self.lru.pop(0)
               del self.cache[evict]
           self.cache[key] = cache
           self.lru.append(key)
   ```

2. Integrate into `Engine.ask()`:
   ```python
   # Check cache
   state_tokens = self.tokenize(state)
   cached = self.state_cache.get(self.model_name, state_tokens)
   
   if cached:
       # Reuse cached state KV
       kv_cache = cached.copy()
   else:
       # Compute state KV
       kv_cache = self.forward(state_tokens)
       self.state_cache.put(self.model_name, state_tokens, kv_cache)
   ```

3. Make opt-in via `poc/minijev.env`:
   ```bash
   MINIJEV_STATE_CACHE_ENABLED=true
   MINIJEV_STATE_CACHE_SIZE=100
   ```

4. Add cache stats to `/v1/health`:
   ```json
   {
     "cache": {
       "enabled": true,
       "size": 100,
       "entries": 47,
       "hits": 1234,
       "misses": 567,
       "hit_rate": 0.685
     }
   }
   ```

5. Validate numerical equivalence:
   - Run same request twice
   - Assert outputs identical
   - Assert second run faster

6. Measure hit rate on synthetic workload with varying overlap

**Acceptance:**
- Cache implemented with LRU eviction
- Opt-in via environment variable
- Hit rate >60% on agent-loop workload
- 1.2-1.6x speedup on cache hits
- Numerical equivalence validated
- Stats visible in /v1/health

**Files to modify:**
- `src/minijev/engine/engine.py`
- `src/minijev/api/server.py` (health endpoint)
- `poc/minijev.env`

---

### Task 4.3: Template Sensitivity Measurement (W7)
**Owner:** TBD | **Effort:** 1 week | **Priority:** Medium

**Goal:** Quantify template robustness with 50-100 variants.

**Implementation:**
1. Create template generator:
   ```python
   def generate_variants(base_template: str, n: int = 50) -> list[str]:
       variants = []
       # Lexical substitutions
       variants.extend(lexical_subs(base_template, ["Answer", "Respond", "Reply"]))
       # Format changes
       variants.extend(format_changes(base_template, [":", "—", "→"]))
       # Semantic paraphrases
       variants.extend(paraphrase(base_template, n_samples=30))
       return variants[:n]
   ```

2. Implement E25 experiment:
   ```python
   def experiment_template_sensitivity():
       base = "Is this urgent? Answer: yes or no"
       variants = generate_variants(base, n=50)
       
       results = []
       for template in variants:
           # Run on held-out BoolQ split
           accuracy, ece, flips = evaluate(template, test_set)
           results.append({
               "template": template,
               "accuracy": accuracy,
               "ece": ece,
               "flips": flips
           })
       
       # Compute stability metrics
       variance = np.var([r["accuracy"] for r in results])
       flip_rate = np.mean([r["flips"] for r in results])
       
       return {
           "variance": variance,
           "flip_rate": flip_rate,
           "results": results
       }
   ```

3. Add to `poc/experiments.py`

4. Generate heatmap visualization for web UI

**Acceptance:**
- E25 generates 50+ template variants
- Measures variance and flip rate
- Identifies stable vs brittle components
- Results documented in RESEARCH.md
- Heatmap added to web UI

**Files to modify:**
- `poc/experiments.py` (add E25)
- `docs/RESEARCH.md` (document results)
- `web/app/findings/page.tsx` (add heatmap)

---

### Task 4.4: Contrastive Level Descriptions (W5)
**Owner:** TBD | **Effort:** 3 days | **Priority:** Medium

**Goal:** Reduce adjacent-level confusion in pointwise Score by 20-35%.

**Implementation:**
1. Modify Score prompt template:
   ```python
   # Old:
   "Is this moderate severity? Answer: yes or no"
   
   # New with contrastive clause:
   "Is this moderate severity (not minor, not severe)? Answer: yes or no"
   ```

2. Implement in `src/minijev/primitives/score.py`:
   ```python
   def contrastive_description(level: int, total: int, levels: list[str]) -> str:
       contrasts = []
       if level > 0:
           contrasts.append(f"not {levels[level-1]}")
       if level < total - 1:
           contrasts.append(f"not {levels[level+1]}")
       
       if contrasts:
           return f" ({', '.join(contrasts)})"
       return ""
   ```

3. Make opt-in via settings:
   ```python
   settings = Settings(
       score_contrastive: bool = False  # default off, user must enable
   )
   ```

4. Implement E24 ablation study:
   - Same Score questions with/without contrastive
   - Measure Δaccuracy and ΔECE
   - Measure level confusion matrix

5. Add toggle to web UI

**Acceptance:**
- Contrastive descriptions implemented
- Opt-in via settings
- E24 shows 20-35% reduction in adjacent-level confusion
- No regression in overall accuracy
- Toggle in web UI

**Files to modify:**
- `src/minijev/primitives/score.py`
- `src/minijev/api/schema.py` (add setting)
- `poc/experiments.py` (add E24)
- `web/components/RequestEditor.tsx` (add toggle)

---

### Task 4.5: Complementary Normalization (W6)
**Owner:** TBD | **Effort:** 3 days | **Priority:** Low

**Goal:** Enforce P(X) + P(not X) = 1.0 for detected opposite Noul pairs.

**Implementation:**
1. Implement detection:
   ```python
   def detect_complementary(q1: Noul, q2: Noul) -> bool:
       # Semantic similarity or explicit marking
       embedding_sim = cosine_similarity(embed(q1.instructions), embed(q2.instructions))
       return embedding_sim > 0.95  # tune threshold
   ```

2. Implement normalization:
   ```python
   def normalize_complementary(p: float, q: float) -> tuple[float, float]:
       """Geometric mean normalization."""
       if p + q == 0:
           return 0.5, 0.5
       λ = math.sqrt((p * q) / ((1 - p) * (1 - q)))
       p_norm = λ / (1 + λ)
       q_norm = 1 - p_norm
       return p_norm, q_norm
   ```

3. Add opt-in flag:
   ```python
   settings = Settings(
       enforce_consistency: bool = False
   )
   ```

4. Add post-processing layer in `src/minijev/judges/judge.py`

5. Implement E23:
   - Generate N opposite Noul pairs
   - Measure sum deviation from 1.0 before/after normalization

**Acceptance:**
- Detection with >98% precision
- Mean |P(X)+P(¬X)-1| < 0.05 after normalization
- Opt-in via flag
- E23 documents improvement

**Files to modify:**
- `src/minijev/judges/judge.py`
- `src/minijev/api/schema.py`
- `poc/experiments.py` (add E23)

---

### Task 4.6: Contrastive Criteria for Vague Questions (W8)
**Owner:** TBD | **Effort:** 3 days | **Priority:** Medium

**Goal:** Fix literal reading by making criteria explicit and contrastive.

**Implementation:**
1. Extend Noul criteria format:
   ```python
   # Old:
   criteria = "Strong in Python means professional experience"
   
   # New contrastive:
   criteria = {
       "true_means": "5+ years daily professional Python use, contributes to major projects",
       "false_means": "Occasional scripts, tutorials, or <2 years experience"
   }
   ```

2. Update prompt template:
   ```
   Does this person qualify as strong in Python?
   
   True means: 5+ years daily professional Python use, contributes to major projects
   False means: Occasional scripts, tutorials, or <2 years experience
   
   Answer: yes or no
   ```

3. Build criteria template library:
   - Expertise levels (strong/proficient/beginner)
   - Urgency (urgent/routine)
   - Sentiment intensity (upset/calm)
   - 20-30 templates total

4. Implement E20 ablation:
   - W8 failure cases with/without contrastive criteria
   - Measure Δaccuracy

5. Add criteria editor to web UI

**Acceptance:**
- Contrastive criteria format supported
- Template library covers 20+ common vague judgments
- E20 shows improvement on W8 cases
- Criteria editor in web UI

**Files to modify:**
- `src/minijev/primitives/noul.py`
- `src/minijev/prompt/templates.py`
- Create `src/minijev/prompt/criteria_library.py`
- `poc/experiments.py` (add E20)
- `web/components/QuestionEditor.tsx` (criteria editor)

---

## Phase 5: Baseline Comparison (Weeks 9-10)

### Task 5.1: ClaudeJudge Baseline (W18)
**Owner:** TBD | **Effort:** 1 week | **Priority:** Medium | **Requires:** ANTHROPIC_API_KEY

**Goal:** Compare minijev token probs vs Claude verbalized confidence.

**Implementation:**
1. Install TypeSafe's system-one-adapter:
   ```bash
   pip install system-one-adapter
   ```

2. Implement ClaudeJudge wrapper:
   ```python
   from anthropic import Anthropic
   
   class ClaudeJudge:
       def __init__(self, api_key: str):
           self.client = Anthropic(api_key=api_key)
       
       def ask_noul(self, state: str, question: str) -> float:
           prompt = f"{state}\n\n{question}\n\nAnswer yes or no, then state your confidence as a percentage."
           response = self.client.messages.create(
               model="claude-sonnet-4",
               messages=[{"role": "user", "content": prompt}]
           )
           # Parse verbalized confidence from response
           return parse_confidence(response.content[0].text)
   ```

3. Implement E34:
   - Run BoolQ held-out set on both minijev and ClaudeJudge
   - Measure: accuracy, ECE, Brier, cost ($), latency
   - Generate comparison table

4. Add multi-model comparison to web UI

**Acceptance:**
- ClaudeJudge wrapper implemented
- E34 comparison on n=200 BoolQ items
- Cost-quality tradeoffs documented
- Comparison table in web UI

**Files to create:**
- `src/minijev/baselines/claude_judge.py`
- `poc/experiments.py` (add E34)

**Files to modify:**
- `web/app/compare/page.tsx` (add Claude baseline)

---

## Phase 6: Long-term Work (Month 4+)
**Note:** Defer until infrastructure exists or resources available.

### Task 6.1: Model Scaling to 3B-7B (W4)
**Requires:** GPU, 16GB+ VRAM  
**Effort:** 3-4 weeks

### Task 6.2: LoRA Training with Option-Shuffle Augmentation (W1/W4)
**Status:** Hard-label step done (E20, RESEARCH.md R17). `poc/train_lora.py` trains on the frozen train splits on the
CPU (2.4 h at 0.5B); `experiments.py lora --adapter ...` reports on test.
Held-out task (E21, R18): the E20 adapter did not help on SST-5 Scores. Per-task adapter (E22, R19,
`train_lora.py train --task sst5`, 14 min): SST-5 listwise accuracy 0.340 → 0.443, but a bias control fitted on val
with no training gives 0.440. E20 beats the same control on AG News (0.873 against 0.805).
**Still requires:** soft labels (API budget for Claude labels), training data under docs/DATA.md §9, and one adapter
trained on many tasks, tested on held-out tasks.

### Task 6.3: FlexAttention Block-Sparse Attention (W12)
**Requires:** PyTorch 2.5+, GPU  
**Effort:** 3 weeks

### Task 6.4: Two-Stage Choice for Large-k (W10)
**Requires:** k>25 labeled dataset  
**Effort:** 2-3 weeks

### Task 6.5: Long Context Needle-in-Haystack (W11)
**Requires:** Memory limits resolved (dense mask OOMs at 32k)  
**Effort:** 1 week

### Task 6.6: RadixCache with Trie-Based Prefix Matching (W13)
**Requires:** Hash cache results showing insufficient hit rate  
**Effort:** 2-3 weeks

---

## Getting Started

### For Agent Assignment:
1. Pick a task from Phase 1 (measurement foundation)
2. Read task acceptance criteria
3. Check "Files to modify" section
4. Implement solution
5. Run validation commands
6. Update status below

### Validation Commands:
```bash
# Tests
cd poc && uv run pytest

# Experiments (after changes)
python experiments.py demo
python experiments.py quality
python experiments.py order-bias

# Web UI
cd web && npm test && npm run lint

# Full integration
./setup.sh && tilt up
```

---

## Task Status Tracker

| Task | Owner | Status | Started | Completed | Notes |
|------|-------|--------|---------|-----------|-------|
| 1.1 Pin Datasets | - | Done | 2026-09-24 | 2026-09-24 | datasets/splits_v2.json: per-row SHA-256, frozen; `data.py check` |
| 1.2 Pre-Register Template | - | Done | 2026-09-24 | 2026-09-24 | Code fingerprint in templates/v1-preregistration.json |
| 1.3 Improve Timing | - | Done | 2026-09-24 | 2026-09-24 | E3 runs every mode in rotated order |
| 1.4 Expand Samples | - | Done | 2026-09-24 | 2026-09-24 | Held-out test: 300 BoolQ, 400 AG News (docs/DATA.md) |
| 2.1 Mode chosen on val | - | Done | 2026-09-24 | 2026-09-24 | E14 |
| 2.2 Fitted Temps | - | Done | 2026-09-24 | 2026-09-24 | E14; calibration/<model>.json |
| 2.3 Update Web UI | - | Done | 2026-09-24 | 2026-09-24 | Calibration panel, Data page |
| 3.1 Package Structure | - | Done | 2026-09-25 | 2026-09-25 | src/minijev; torch pinned to 2.2.2 (Intel Mac), not >=2.4 |
| 4.1 Two-Level Tree | - | Done | 2026-09-25 | 2026-09-25 | Score branches −21.6% tokens (GDPR); logits within 2e-5; W14 |
| 4.2 State Cache | - | Done | 2026-09-25 | 2026-09-25 | LRU of state key/values; 1.2–2.9× per hit (W13); hit rate depends on the client |
| 4.3 Template Sensitivity | - | Done | 2026-09-25 | 2026-09-25 | E15: 81 templates on BoolQ val; heatmap on Findings (W7) |
| 4.4 Contrastive Levels | - | Done (rejected) | 2026-09-25 | 2026-09-25 | E16 on SST-5: no fewer adjacent errors; worse at 1.5B; stays off (W5) |
| 4.5 Complementary Norm | - | Done | 2026-09-25 | 2026-09-25 | E17; explicit `opposite_of` (no automatic detection); opt-in: at 0.5B it lowers accuracy (W6) |
| 4.6 Contrastive Criteria | - | Done | 2026-09-25 | 2026-09-25 | Library of 20; picker in the editor; E18 mixed at 0.5B (W8) |
| 5.1 ClaudeJudge Baseline | - | Blocked | - | - | **Requires `ANTHROPIC_API_KEY`**; not set on this machine |
| 6.1 Model Scaling 3B–7B | - | Blocked | - | - | Needs a GPU with 16 GB+ |
| 6.2 LoRA + shuffle augmentation | - | Partly done | 2026-09-25 | 2026-09-25 | E20, hard labels, 0.5B: BoolQ 0.693 → 0.770, listwise flips 22% → 9%. E21: no gain on SST-5 (held out). E22: SST-5 adapter 0.340 → 0.443, equal to a 6-number bias control. Soft labels still blocked (API budget) |
| 6.3 FlexAttention | - | Blocked | - | - | Needs PyTorch 2.5+ and a GPU; torch is pinned to 2.2.2 (Intel Mac) |
| 6.4 Two-Stage Choice | - | Blocked | - | - | Needs a labelled dataset with more than 25 options |
| 6.5 Needle-in-Haystack | - | Done (to 8k) | 2026-09-25 | 2026-09-25 | E19 in kv mode on CPU; 32k still open (W11) |
| 6.6 RadixCache | - | Not needed yet | - | - | The hash cache (4.2) has no hit-rate problem to solve yet |

---

## Dependencies Between Tasks

```
Phase 1 (Measurement) → Phase 2 (Quick Wins)
                      → Phase 3 (Package)
                      → Phase 4 (Foundation)
                      → Phase 5 (Baselines)

Phase 3 (Package) blocks:
  - Task 4.1-4.6 (need package structure)
  - Phase 6 (all long-term work)

Task 1.2 (Pre-Register) blocks:
  - Task 4.3 (Template Sensitivity)
  - Any future template tuning
```

---

## Documentation Updates

After each phase completion, update:
- `docs/WEAKNESSES.md` - Mark weaknesses as "Fixed" or "In Progress"
- `docs/RESEARCH.md` - Add new experiment results (E16-E34)
- `docs/WALKTHROUGH.md` - Update with new features
- `README.md` - Update quickstart if packaging changed

---

## Questions or Issues?

- See `docs/SOLUTIONS.md` for full research analysis
- See `docs/WEAKNESSES.md` for current weakness status
- See `docs/DESIGN.md` for architecture details
- Ask in project chat or open GitHub issue
