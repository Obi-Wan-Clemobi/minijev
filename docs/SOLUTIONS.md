# minijev — Solutions to Known Weaknesses

*Research initiative output: 10 researchers × 20 peer reviews (Opus, high effort) → synthesis. Evidence levels follow RESEARCH.md: **Measured** (minijev runs), **Observed** (elsewhere), **Stated** (TypeSafe), **Inferred** (reasoning).*

## Executive Summary

Analysis of 10 research reports and 20 peer reviews covering minijev's 19 documented weaknesses reveals a clear implementation path. The research is grounded in 15+ committed experiments (E1-E13) with measured results.

**Critical findings:**

1. **W1 (option-order bias) is already solved for the target model size**: E13 measured pointwise Choice at 1.5B achieving 0.908 accuracy / 0.073 ECE / 0% flips vs listwise 0.840 / 0.143 / 8%. Making pointwise the default at 1.5B is zero new code—just change MINIJEV_CHOICE_MODE.

2. **Calibration (W2/W3) is half-done**: Temperature scaling works for Noul (ECE 0.161→0.053), and the machinery exists for Choice (fit_temperature_multiclass already runs on AG News). Missing: committed temperatures in minijev.env, Score coverage, and per-mode calibration.

3. **W4 (model capacity) is not fixable by prompting at 0.5B**: BoolQ accuracy 0.630 overlaps with the 0.62 base rate. All distillation/LoRA proposals depend on resources the repo lacks (API key, GPU, labeled data). The pragmatic fix is "use 1.5B."

4. **Measurement discipline (W15-W18) gates everything else**: Current template was chosen with test cases visible (W7/W15), datasets are unpinned (W17), and timing has ±10-20% noise (W16). No weakness can be declared "fixed" without first closing W15-W18.

5. **Infrastructure (W19) is the true blocker**: The POC is two flat files (minijev_poc.py 412 lines, server.py 313 lines, experiments.py 1473 lines). Every advanced proposal (LoRA, Redis, multi-stage serving) assumes a package structure that does not exist.

**Recommended sequence:** Fix measurement first (W15-W18, ~2 weeks), then package structure (W19, ~1 week), then calibration expansion (W2/W3, ~1 week), then the validated quick wins (W1 pointwise default, W14 two-level tree). Defer or drop training-based solutions until infrastructure exists.

---

## Priority Matrix

Solutions ranked by impact × feasibility, with evidence level:

| Tier | Weakness | Solution | Impact | Feasibility | Evidence |
|------|----------|----------|--------|-------------|----------|
| **Quick Win** | W1 | Make pointwise Choice default at 1.5B | 9 | 10 | measured_minijev |
| **Quick Win** | W17 | Per-row SHA256 dataset manifest with --update-manifest flag | 8 | 10 | observed_elsewhere |
| **Quick Win** | W15 | Pre-register template and fix held-out split before tuning | 9 | 9 | measured_minijev |
| **Quick Win** | W2/W3 | Commit fitted Choice/Score temperatures to minijev.env | 8 | 9 | measured_minijev |
| **Quick Win** | W16 | Add warm-up runs and increase timing repetitions to 5 | 6 | 10 | observed_elsewhere |
| **Foundation** | W19 | Package structure: src/minijev with pyproject.toml [project.scripts] | 7 | 8 | observed_elsewhere |
| **Foundation** | W14 | Two-level prefix tree (state→question→levels) for Score | 6 | 8 | theoretical |
| **Foundation** | W5 | Contrastive level descriptions (not minor, not severe) | 7 | 9 | observed_elsewhere |
| **Foundation** | W6 | Complementary normalization (geometric mean of odds) | 5 | 9 | theoretical |
| **Foundation** | W13 | Hash-based exact-match state cache | 6 | 7 | theoretical |
| **Foundation** | W18 | ClaudeJudge baseline via system-one-adapter (verbalized confidence) | 7 | 6 | observed_elsewhere |
| **Foundation** | W7 | Template sensitivity measurement (50-100 variants) | 7 | 7 | observed_elsewhere |
| **Foundation** | W11 | Needle-in-haystack position-controlled eval (2k-8k tokens) | 6 | 6 | theoretical |
| **Foundation** | W8 | Contrastive criteria (True means X, False means Y) | 6 | 8 | theoretical |
| **Long-term** | W9 | Batch Calibration (label-free content-free prior) | 5 | 6 | observed_elsewhere |
| **Long-term** | W12 | FlexAttention BlockMask for tree-structured sparsity | 4 | 4 | observed_elsewhere |
| **Long-term** | W10 | Two-stage Choice (pointwise stage 1, skip stage 2 for now) | 5 | 5 | theoretical |
| **Long-term** | W4 | Model scaling to 3B-7B (requires GPU/bf16) | 8 | 3 | observed_elsewhere |
| **Long-term** | W1/W4 | LoRA with option-shuffle augmentation + proper scoring rules | 8 | 2 | observed_elsewhere |
| **Long-term** | W2/W3 | Hierarchical calibration (T_base + per-domain alpha) | 7 | 4 | theoretical |

---

## Implementation Roadmap

### Quick wins (1-2 weeks)

- **W1**: Change MINIJEV_CHOICE_MODE=pointwise for 1.5B, averaged for 0.5B (0 days, doc update only)
- **W17-A**: Implement per-row SHA256 manifest with explicit --update-manifest flag (1 day)
- **W15-B**: Document current template in preregistration file, fix held-out split (1 day)
- **W2/W3**: Commit fitted temperatures to minijev.env (MINIJEV_TEMP_CHOICE, per-mode) (1 day)
- **W16-A**: Add warm-up runs to latency experiments, report median+IQR over 5 runs (1 day)
- **W15-A**: Pin BoolQ/AG News rows with revision hash, verify checksums on load (2 days)

### Foundation work (3-10 weeks)

- **W19**: Package restructure to src/minijev layout with [project.scripts] (1 week)
- **W7**: Template sensitivity measurement across 50+ variants, measure variance (1 week)
- **W14**: Two-level prefix tree for pointwise Score (state→question→levels) (1 week)
- **W5**: Contrastive level descriptions in proposal_block (3 days)
- **W2/W3**: Expand calibration to Choice at all readout modes, Score with ordinal metrics (1 week)
- **W13**: Hash-based state cache with LRU eviction, key on (model, state_tokens) (1 week)
- **W18-A**: ClaudeJudge baseline using system-one-adapter, verbalized confidence (1 week)
- **W6**: Complementary normalization layer (geometric mean of odds, opt-in flag) (3 days)
- **W8**: Contrastive criteria templates for vague questions (True means X, False means Y) (3 days)
- **W11**: Needle-in-haystack eval at 2k/4k/8k (after resolving dtype/mask memory limits) (1 week)

### Long-term work (3-6 weeks each)

- **W4**: Model scaling to 3B-7B (requires GPU, bf16/int8, new validation suite) (3-4 weeks)
- **W12**: FlexAttention block-sparse attention for packed mode (requires torch 2.5+, GPU) (3 weeks)
- **W10**: Two-stage Choice with top-k selection (requires large-k labeled dataset) (2-3 weeks)
- **W9**: Batch Calibration or Domain-Context Calibration for content-free prior (2 weeks)
- **W1/W4**: LoRA training with option-shuffle augmentation (requires GPU, synthetic data pipeline) (4-6 weeks)
- **W2/W3**: Hierarchical calibration with per-domain adjustment (requires slice-specific data) (3 weeks)
- **W13**: RadixCache with trie-based prefix matching (after hash cache shows value) (2-3 weeks)
- **W11**: Contrastive training for relevance (Focused Transformer approach, requires GPU/data) (4-5 weeks)

---

## UX Integration Plan

### Calibration (W2/W3/W9)

**Data to surface:**
- Current temperature settings per primitive (Noul: 2.72@0.5B, 1.93@1.5B; Choice: TBD; Score: TBD)
- Per-mode calibration status (listwise/pointwise/averaged may have different temps)
- ECE and Brier score for last request (computed from bootstrap_ci on held-out set)
- Calibration fit quality: n samples used, holdout ECE with 95% CI

**Controls needed:**
- Per-primitive temperature sliders (extend existing Calibration.tsx dials)
- Per-mode temperature override (when choice_mode != default)
- Reset to fitted defaults button
- Calibration profile selector (conservative/balanced/aggressive = T*0.8, T*1.0, T*1.2)

**Visualizations:**
- Reliability diagram per primitive (binned confidence vs empirical accuracy)
- ECE decomposition (calibration loss + refinement loss, per Murphy decomposition)
- Per-question calibration quality indicator (flag if confidence in poorly-calibrated bin)

**Experiments:**
- **E11 Quality** (existing): Expand to n=500 BoolQ, n=300 AG News, add Score on Yelp-5
- **E16 Calibration Transfer**: Fit on domain A, test on domain B, measure ECE degradation
- **E17 Mode-Specific Calibration**: Temperature per (primitive, readout_mode, model) tuple

### Position Bias (W1)

**Data to surface:**
- Current choice_mode (listwise/pointwise/averaged)
- Per-mode quality metrics: flip rate, accuracy, ECE (from E13)
- Position bias table: per-letter preference when in listwise mode
- Order-invariance score: mean max |Δp| across shuffles

**Controls needed:**
- choice_mode selector (already exists, promote to top-level preset)
- Show all rotations toggle for averaged mode (display individual rotation results)
- Debiasing method selector (none/PriDe/pointwise) with cost/quality tradeoff

**Visualizations:**
- Flip-rate heatmap: how often argmax changes across option orders
- Per-option probability stability: violin plot of P(option) across rotations
- Cost vs quality frontier: branch count vs flip rate for each mode

**Experiments:**
- **E13 Order Bias** (existing): Expand to larger dataset, test numeric labels (1/2/3/4)
- **E18 Full Permutations**: Test k! orderings for k≤4 (currently only k rotations)
- **E19 Label Scheme**: Compare letter (A/B/C/D) vs numeric (1/2/3/4) vs word (First/Second) labels

### Model Quality (W4/W8)

**Data to surface:**
- Model size and type (0.5B/1.5B/3B, instruct vs base)
- Accuracy vs base rate per dataset (flag when at chance)
- Per-question literal vs intended interpretation signal
- Criteria usage: which questions have explicit criteria, which rely on model priors

**Controls needed:**
- Model selector (extend existing, add 3B when available)
- Criteria editor per question (True means X, False means Y)
- Base vs instruct model toggle (test whether RLHF hurts token prob calibration)
- Confidence threshold for 'uncertain' flagging

**Visualizations:**
- Accuracy by question type (with criteria vs without)
- Literal vs contextual interpretation examples (side-by-side comparisons)
- Model size vs accuracy curves per dataset

**Experiments:**
- **E11 Quality** (existing): Run on base Qwen2.5 (non-instruct) to test Kadavath hypothesis
- **E20 Criteria Ablation**: Same questions with/without contrastive criteria, measure Δaccuracy and ΔECE
- **E21 Few-Shot**: Add 2-3 exemplars in shared prefix, measure transfer

### Pointwise Biases (W5/W6)

**Data to surface:**
- Per-level yes-bias in pointwise Score (how often each level is chosen)
- Complementary Noul consistency: P(X) + P(not X) for detected pairs
- Level confusion matrix for Score (how often adjacent levels are conflated)
- Min label mass warnings (when softmax mass leaks off labels)

**Controls needed:**
- Contrastive level descriptions toggle (level i sees i±1 in prompt)
- Complementary normalization toggle (enforce P(X)+P(¬X)=1)
- Semantic complementarity detector (flag potential opposite questions)
- Per-level bias correction (vector scaling on log-odds)

**Visualizations:**
- Score level separation plot (mean P(yes) per level, with error bars)
- Complementary consistency scatter (P(X) vs P(not X), diagonal = perfectly consistent)
- Yes-bias heatmap across Score scales

**Experiments:**
- **E22 Pointwise Score Separation**: Measure level confusion at 0.5B/1.5B/3B
- **E23 Complementary Pairs**: Generate N opposite Noul pairs, measure sum deviation from 1.0
- **E24 Contrastive Prompts**: Same Score questions with/without contrastive descriptions

### Template Robustness (W7/W10)

**Data to surface:**
- Current template version and SHA fingerprint
- Template stability score: variance of P across paraphrases
- Per-question answer flip rate across templates
- Supported option count (currently capped at 25, target 255)

**Controls needed:**
- Template lock toggle (freeze current template, require explicit update)
- Template variant selector (for A/B testing approved alternatives)
- Large-k mode selector (pointwise-only for k>25, two-stage when implemented)
- IIA validation button (test that adding irrelevant option does not change P(A)/P(B))

**Visualizations:**
- Template sensitivity heatmap (questions x templates, color = flip/no-flip)
- Variance plot: std(P) across template variants per question
- k-scaling plot: accuracy/ECE/latency vs number of options

**Experiments:**
- **E25 Template Sensitivity**: 50-100 lexical variants, measure flip rate and variance
- **E26 Large-k Scaling**: Measure quality and cost at k=25/50/100/255
- **E27 IIA Validation**: Add irrelevant options, assert P(A)/P(B) ratio stable

### Long Context (W11)

**Data to surface:**
- State length and model's max_position_embeddings (32768 for Qwen2.5)
- Position-dependent accuracy curve (answer early/middle/late)
- Label mass at different state lengths (flag collapse off labels)
- Effective context window (where accuracy/ECE degrade)

**Controls needed:**
- State truncation control (test on first/last N tokens)
- Position randomization toggle (move answer location in synthetic tests)
- Windowing toggle (when/if implemented, with window size param)

**Visualizations:**
- Needle-in-haystack heatmap (state length x answer position)
- Lost-in-the-middle curve (accuracy vs position in long state)
- ECE vs state length

**Experiments:**
- **E28 Position-Controlled Eval**: GDPR states at 1k/2k/4k/8k, answer at 0%/25%/50%/75%/100%
- **E29 Label Mass Decay**: Plot min_label_mass warnings vs state length
- **E30 Retrieval vs Reasoning**: Needle retrieval ("what is X?") vs inference ("does X imply Y?")

### Serving Efficiency (W12/W13/W14)

**Data to surface:**
- Latency breakdown: state prefill / branch prefill / readout (from packed mode trace)
- Attention mask sparsity (% of blocks masked out)
- Cache hit rate and state reuse statistics
- Memory usage: model weights / KV cache / attention scratch

**Controls needed:**
- Attention implementation selector (eager/sdpa/flex, when available)
- dtype selector (fp32/bf16/int8)
- Cache toggle and size limit
- Prefill optimization toggle (two-level tree for Score)

**Visualizations:**
- Latency waterfall: state vs branches, by readout mode
- Attention mask visualization (existing /hood page, extend to three-level tree)
- Cache efficiency plot: hit rate vs cache size, with working set curve

**Experiments:**
- **E3 Latency** (existing): Expand to measure per-phase breakdown and cache impact
- **E31 Two-Level Tree**: Measure token count reduction for pointwise Score
- **E32 Cache Hit Rate**: Synthetic workload with varying state overlap, measure speedup

### Measurement Quality (W15-W18)

**Data to surface:**
- Dataset version: manifest SHA, row count, download date
- Experiment provenance: model, prompt fingerprint, random seed, split membership
- Bootstrap confidence intervals (on all reported metrics)
- Baseline comparison: minijev vs ClaudeJudge vs Jev (when available)

**Controls needed:**
- Dataset refresh button with integrity check
- Experiment seed selector (reproduce or randomize)
- CI width selector (90%/95%/99%)
- Baseline toggle (show/hide frontier model comparison)

**Visualizations:**
- Confidence interval plots for all metrics (accuracy, ECE, Brier, latency)
- Multi-model comparison table with statistical tests
- Timing variance plot (median, IQR, outliers across repetitions)

**Experiments:**
- **E11 Quality**: Re-run with fixed held-out split and preregistered template
- **E33 Statistical Power**: Sample-size analysis for target effect sizes
- **E34 Baseline Comparison**: ClaudeJudge verbalized confidence vs minijev token probs

---

## Research Gaps

The following gaps block further progress on specific weaknesses:

1. **W2/W3 Calibration Transfer**: Zero calibration evidence for Choice/Score beyond single-dataset fits. Need: multi-domain calibration sets (n≥300 per domain), cross-domain transfer experiments, and per-(primitive, mode, model) fitted temperatures. Current status: Noul done, Choice partially measured, Score unmeasured.

2. **W4 Capacity Floor**: No systematic measurement of where 0.5B/1.5B/3B become viable. Need: accuracy vs model size curves across multiple tasks, identification of task types that require certain capacity, and validation against larger open models (7B-14B). Blocked on GPU access and larger labeled datasets.

3. **W5 Pointwise Score Separation**: Only 10 documented Jev examples, insufficient for training or validation. Need: n≥200 labeled Score tasks with ground truth across diverse domains (severity, quality, urgency, sentiment intensity). Currently no ordinal regression benchmark exists in the repo.

4. **W6 Complementary Consistency**: Only one Observed Jev datapoint (P+P'=1.19). Need: measure minijev's own baseline on N≥100 opposite Noul pairs, test whether existing temperature scaling reduces violations, and validate normalization formulas on held-out pairs. Gap: no synthetic paired dataset exists.

5. **W7 Template Stability**: Current template chosen with test cases visible (contaminated). Need: n≥50 template variants tested on held-out split, variance decomposition (lexical vs semantic vs format changes), and label-free selection criteria. Also need: automated template generation to reduce manual effort.

6. **W8 Literal Reading**: Only 2 documented failure cases with Jev reference values. Need: n≥100 vague questions with human-adjudicated ground truth (not just Jev agreement), systematic categorization of vagueness types (ambiguous threshold, context-dependent, subjective), and criteria templates validated per category.

7. **W10 Large-k Scaling**: No labeled dataset with >25 options exists in repo. Need: k=50/100/255 evaluation sets, validation that pointwise mode preserves quality at large k, and cost-benefit analysis vs two-stage approach. Also need: test whether IIA property holds in practice (does adding option change P(A)/P(B)?).

8. **W11 Long Context Degradation**: Qwen2.5 is 32k-native (max_position_embeddings=32768) but untested beyond 1k tokens. Need: needle-in-haystack at 2k/4k/8k/16k/32k with answer at varying positions, measure whether ECE degrades with length independent of accuracy, and test if windowing helps or hurts. Blocked on memory limits (dense mask OOMs at 32k).

9. **W15 Sample Size Requirements**: Current n=200-400 gives wide CIs (±0.06-0.08 on ECE). Need: power analysis for target effect sizes, minimum sample sizes per primitive/domain, and validation that bootstrap CI assumptions hold (no strong autocorrelation, representative sampling). Gap: no pre-registered analysis plan exists.

10. **W17 Dataset Stability**: BoolQ and AG News pulled from HF datasets-server with no version pin. Need: pin revision hashes, implement per-row checksums, and monitor for upstream changes. Gap: silent dataset drift would invalidate all comparisons but is currently undetected.

11. **W18 Baseline Mechanisms**: No comparison against frontier models or other methods. Need: ClaudeJudge (verbalized confidence), Jev (when API key available), and open alternatives (GPT-4o, Gemini, Qwen2.5-7B). Gap: unclear whether minijev's approach is competitive or how to trade off cost/accuracy/calibration.

12. **Cross-Weakness Interactions**: No systematic measurement of how fixes compose. Need: test whether training-based fixes (W1/W4 LoRA) invalidate calibration (W2/W3), whether large k (W10) interacts with position bias (W1), and whether long context (W11) amplifies literal reading (W8). Gap: each weakness analyzed in isolation.

---

## Next Actions

**Week 1-2: Fix Measurement Foundation (W15-W18)**
1. Pin BoolQ/AG News with revision hashes and implement per-row checksums (W17-A)
2. Document current template in preregistration file, fix train/val/test splits with disjoint index ranges (W15-B)
3. Add warm-up runs and increase timing repetitions to 5 with median+IQR reporting (W16-A)
4. Expand BoolQ to n=500, AG News to n=300 with stratified sampling (W15-A)

**Week 3: Deploy Quick Wins (W1, W2/W3)**
1. Change MINIJEV_CHOICE_MODE to pointwise for 1.5B (one line in minijev.env plus docs)
2. Commit fitted Choice temperatures from E11 to minijev.env (MINIJEV_TEMP_CHOICE_LISTWISE, _POINTWISE, _AVERAGED)
3. Update web UI to show per-mode calibration status
4. Add plaintext tooltips explaining the choice_mode tradeoff

**Week 4: Package Structure (W19 Phase 1)**
1. Create src/minijev layout with api/, judges/, prompt/, engine/, primitives/, calibrate/
2. Add [project.scripts] entry point 'minijev = minijev.cli:app'
3. Migrate experiments.py to its proper location
4. Update all imports, tests, Dockerfile, docker-compose, Tiltfile, README
5. Run full test suite and experiment regression

**Week 5: Calibration Expansion (W2/W3)**
1. Fit temperatures for Score on labeled ordinal dataset (Yelp-5 or SST-5)
2. Implement per-mode calibration (separate temps for listwise/pointwise/averaged Choice)
3. Add ordinal metrics for Score (ranked probability score, CRPS)
4. Extend Calibration.tsx with per-primitive controls
5. Run E17 (calibration transfer across domains)

**Week 6: Two-Level Prefix Tree (W14)**
1. Refactor proposal_block to separate question header from PROPOSED ANSWER
2. Implement three-tier tree in pack(): state → question → levels
3. Extend kv mode to crop at two levels
4. Measure token count reduction on GDPR Scores
5. Validate numerical equivalence (logits match within 1e-3)

**Week 7: Template Robustness Measurement (W7)**
1. Implement template_sensitivity experiment with 50-100 lexical/semantic/format variants
2. Measure per-question flip rate and variance across templates
3. Identify stable template components via ablation
4. Document recommended templates per primitive with stability scores

**Week 8: Contrastive Prompts (W5, W8)**
1. Add contrastive clauses to Score level descriptions (level i mentions i±1)
2. Implement contrastive Noul criteria templates (True means X, False means Y)
3. Run E24 ablation (with/without contrastive descriptions)
4. Measure impact on level separation and literal reading

**Week 9-10: Baseline Comparison (W18)**
1. Integrate TypeSafe's system-one-adapter for ClaudeJudge
2. Run E34 comparison: token probs vs verbalized confidence
3. Measure accuracy, ECE, Brier, cost, latency for both
4. Document when to use each approach
5. Add multi-model comparison table to web UI

**Week 11: Hash-Based State Cache (W13 MVP)**
1. Implement LRU cache of DynamicCache keyed by (model, prefix_tokens)
2. Measure hit rate on synthetic workload with varying overlap
3. Validate numerical equivalence (cache hit matches fresh prefill)
4. Add cache stats to /v1/health endpoint
5. Make cache opt-in with size limit config

**Month 4+: Long-Term Work** (prioritize based on team needs)
- Model scaling to 3B-7B if GPU available (W4)
- FlexAttention block-sparse attention if torch 2.5+ upgrade feasible (W12)
- LoRA training if GPU + API key + data pipeline exist (W1/W4)
- RadixCache if hash cache shows insufficient hit rate (W13)
- Two-stage Choice if large-k dataset obtained (W10)
- Long-context work if memory limits resolved (W11)

**Ongoing: Measurement Discipline**
1. Never run experiments without fixed held-out splits
2. Report bootstrap CIs on all metrics
3. Pre-register analysis plans before template changes
4. Pin dataset revisions and verify checksums
5. Run equivalence tests (three-mode agreement, cache hit) after any engine change
6. Update WEAKNESSES.md status immediately when fixes ship

**Documentation: Update docs/ as work proceeds**
1. WEAKNESSES.md: Mark W1 'Fixed at 1.5B', W15/W17 'In Progress'
2. RESEARCH.md: Add E25-E34 results as completed
3. DESIGN.md: Update Phase 2 status to 'Done' after packaging
4. WALKTHROUGH.md: Add sections for new experiments
5. README.md: Update quickstart with 'pip install minijev' when available
