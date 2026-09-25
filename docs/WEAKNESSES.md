# minijev — Known weaknesses

*A register of every known weakness of minijev, so that we can fix them one at a time. Each entry says what the
weakness is, the evidence, a candidate fix, and what Jev may do about it. Evidence labels follow RESEARCH.md:
**Measured** (our runs), **Observed** (Jev outputs or third parties), **Stated** (TypeSafe says it), **Inferred** (our
reasoning; it can be wrong).*

Terms used below: a **readout** takes the probabilities of the allowed answer labels from one forward pass. A
**branch** is the tokens of one question after the state. **Listwise** shows all options in one prompt with letter
labels; **pointwise** judges each option or level alone with a yes/no. **ECE** (expected calibration error) is the
average gap between stated confidence and accuracy.

## Summary

| # | Weakness | Area | Severity | Status |
|---|---|---|---|---|
| W1 | The option order changes listwise answers | Method | High | Partly: the default is the order-robust mode chosen on val (E14); listwise still available |
| W2 | Raw probabilities are overconfident | Calibration | High | Partly: fitted for Noul and Choice on train (E14); Score open |
| W3 | Calibration does not transfer beyond its data, and Score has none | Calibration | High | Open |
| W4 | The 0.5B model is at chance on BoolQ | Model | High | Use 1.5B or larger; open |
| W5 | Pointwise items on small models say "yes" to the plausible item | Method | Medium | Open |
| W6 | Opposite questions are not consistent | Method | Medium | Not measured |
| W7 | Answers depend on the prompt wording | Method | Medium | Measured (E15); template frozen |
| W8 | Literal reading of vague questions | Model | Medium | Partly: Noul `criteria` |
| W9 | Contextual calibration over-corrects | Calibration | Low | Avoided (opt-in) |
| W10 | More than 25 options is not supported | Method | Low | Open |
| W11 | Long states are untested | Model | Medium | Open |
| W12 | The packed pass computes the hidden attention blocks | Speed | Low (CPU), Medium (GPU) | Open |
| W13 | No state cache across requests; one request at a time | Speed | Medium | Open |
| W14 | Pointwise Scores repeat the question for every level | Speed | Low | Fixed |
| W15 | Small samples, and tests that are not held out | Measurement | Medium | Partly: held-out test of 300 / 400 (E14); E1–E13 stay exploratory |
| W16 | Timing noise on one laptop | Measurement | Low | Partly: medians, rotation |
| W17 | Datasets are not pinned | Reproducibility | Low | Fixed: frozen splits with per-row SHA-256, checked on every load |
| W18 | No cloud-model baseline | Measurement | Medium | Open (needs an API key) |
| W19 | The app is a POC: no packaging, no persistence, Docker untested | Engineering | Low | Partly: package, CLI and API exist (src/minijev); Docker build untested |

---

## Method

### W1. The option order changes listwise answers
- **What:** in listwise mode the model sees "A) billing B) technical C) sales" and picks a letter. A small model
  likes some letters more than others, whatever the options are. So moving an option can change the winner.
- **Evidence (Measured):** on 8 documented Choices, a rotation of the options changed the winner in 54% of the
  rotations at 0.5B and 9% at 1.5B (RESEARCH.md §7.3 R4). E13 measures it on 120 AG News articles with known answers
  (RESEARCH.md §7.3 R10): as-is, the answer changed with the order for 20% of the articles at
  0.5B and 8% at 1.5B.
- **Fixes, measured in E13 (WALKTHROUGH.md §6.10):**
  - *All orders averaged:* ask once per rotation in the same pass and average each option's probability. Flips
    fall to 3% (0.5B) and 2% (1.5B), and ECE falls at 0.5B. It costs one branch per option;
    the text is still read once. The app offers it as "asked: all orders averaged". Best at 0.5B.
  - *Debiased:* divide out the model's measured liking for each letter (PriDe, Zheng et al. 2024). One branch.
    Flips only fall to 18% and 8%: the position effect depends on the content.
  - *Pointwise:* judge each option alone. 0% flips, by construction; one branch per option. Best at 1.5B:
    accuracy 0.908 and ECE 0.073, against 0.840 and 0.143 as-is.
- **What Jev may do (Inferred):** Jev judges Score levels "separately", without "a level's number or its neighbours"
  (Stated, row 12): that removes position effects for Scores. Large Choices go through an independent scoring stage
  first (Stated, row 11). Jev's probability maps come back in a different key order from the request (Observed,
  row 24), which fits internal option shuffling or averaging. Training on shuffled options (RESEARCH.md §3.9) would
  also remove the bias. RESEARCH.md §8 test 1 can tell these apart with a Jev key.

### W5. Pointwise items on small models say "yes" to the plausible item
- **What:** a pointwise branch asks "is this level right? yes/no" with no comparison. A small model says yes to
  anything plausible, so neighbouring levels get similar scores.
- **Evidence (Measured):** at 0.5B, pointwise Score rated every bug report "workaround exists", from a misaligned icon
  to a full outage. 1.5B separated them (RESEARCH.md §7.3 R5).
- **Candidate fix:** contrastive level descriptions ("… but not …"); a larger model; training on pointwise data.
- **What Jev may do (Inferred):** train the pointwise judgement directly (RLCD, RESEARCH.md §3.5), so that "yes" is
  calibrated for each item.

### W6. Opposite questions are not consistent
- **What:** each question is its own branch and never sees another one. Nothing forces P(X) + P(not X) = 1.
- **Evidence:** Jev shows it: P(refund) + P(not refund) = 1.19 (Stated, jaggedness page, row 26). minijev has the same
  structure, so it has the same risk (Inferred; not measured).
- **Candidate fix:** measure it with pairs of opposite Nouls (RESEARCH.md §8 test 9); if needed, a consistency
  step that asks both and renormalizes.
- **What Jev may do (Inferred):** nothing yet; its own docs list it as a known weakness.

### W7. Answers depend on the prompt wording
- **What:** a different system line or question template changes the probabilities.
- **Evidence (Measured):** the answer format alone changed the picked option in E10 (69% agreement at 0.5B between
  "answer with the name" and the letter readout, WALKTHROUGH.md §6.3). The template was chosen while the documented
  cases were visible, so R5 is not a held-out test.
- **Evidence (Measured, E15):** 81 templates (3 wordings of the system line, the state label, the question label and
  the answer line) on 200 BoolQ val questions at 0.5B. Accuracy goes from 0.64 to 0.74 (SD 0.019); one template's
  95% interval is ±0.065, so no template is clearly more accurate. But single answers move: on average 9.8% of answers
  flip against the production template (at most 21%), and 39.5% of the questions flip under at least one template.
- **Which part matters most (Measured, E15):** the system line. Over the 27 templates that use each line, the mean ECE
  is 0.173 for the production line and 0.126–0.129 for the two others; the production line also gives the highest
  mean P(yes) (0.697 against a true rate of 0.620). The question label changes the least (mean flips 9.6–9.9%).
- **Fix, in part:** the template is frozen and pre-registered (`poc/templates/v1-preregistration.json`), and E14 fits
  the temperature for this template, which absorbs most of the over-confidence. A different template must be chosen
  on val and registered again.
- **Still open:** a template that is less biased to "yes"; the same grid at 1.5B.
- **What Jev may do (Inferred):** a hidden template of ≈270–300 tokens per request (Observed, row 22), tuned in
  training, so users never write the prompt.

### W10. More than 25 options is not supported
- **What:** labels are single letters (A–Z without I), so a Choice has at most 25 options. Jev allows 255.
- **Candidate fix:** two stages, as Jev does: score every option pointwise, keep the best few, then one listwise
  Choice (DESIGN.md §5.2).
- **What Jev does:** exactly that (Stated, launch blog, row 11).

## Calibration

### W2. Raw probabilities are overconfident
- **Evidence (Measured):** raw ECE 0.10–0.22 on BoolQ and AG News at both sizes (E1, E11). When 0.5B was "95%+ sure",
  it was right 86% of the time.
- **Fix:** temperature scaling. It reduced ECE to 0.045–0.076 without changing any answer (E11). The dials and
  `poc/minijev.env` apply it.
- **What Jev may do (Inferred):** calibrate in training (probabilities "optimized against outcomes", Stated,
  row 6), plus a final temperature.

### W3. Calibration exists only for Noul, and does not transfer
- **What:** we have fitted temperatures only for yes/no questions on BoolQ. A temperature fitted on one kind of data
  can be wrong on another.
- **Evidence:** SemIf measured temperatures from 1.2 to 2.5 across workloads (Observed). The pointwise Score softmax
  uses a different scale from Noul log-odds, so a Noul temperature does not apply to it (Inferred, DESIGN.md §5.5).
- **Candidate fix:** calibration sets for Choice and Score; fit per primitive and per domain; report the transfer.
- **What Jev may do (Inferred):** train on many synthetic domains so that one calibration holds broadly
  (RESEARCH.md §3.9).

### W9. Contextual calibration over-corrects
- **Evidence (Measured):** subtracting an "N/A"-state prior made ECE worse at both sizes (0.161 → 0.350 at 0.5B).
- **Status:** it is opt-in only; temperature scaling is the default.

## Model

### W4. The 0.5B model is at chance on BoolQ
- **Evidence (Measured):** accuracy 0.630 [0.565–0.695] against a 0.62 base rate (E11). 1.5B reaches 0.820.
- **Candidate fix:** use 1.5B or a larger model; fine-tune (distillation, RESEARCH.md §3.9).
- **What Jev may do (Inferred):** a larger model (≈15–20B active parameters, RESEARCH.md §3.9) trained for decisions.

### W8. Literal reading of vague questions
- **Evidence (Measured):** both sizes counted "used Python occasionally" as *strong in Python* (0.90–0.93; Jev 0.14),
  and "charged twice, can someone look into this?" as *not* a refund request (RESEARCH.md §7.3 R5).
- **Candidate fix:** Noul `criteria` ("Yes means …") make the question precise; a larger model.
- **What Jev may do:** it lists literal reading as a known weakness (Stated, row 25).

### W11. Long states are untested
- **What:** our runs use states of at most 1,000 tokens. Jev allows 32k tokens and warns about "context rot".
- **Candidate fix:** a long-context test: the same question with the answer early, in the middle, and late in a long
  state.

## Speed and serving

### W12. The packed pass computes the hidden attention blocks
- **What:** the packed mode uses a dense attention mask, so it still computes the blocks that the mask hides. Cost
  grows with the square of the total length.
- **Candidate fix:** block-sparse attention (FlexAttention, FlashAttention varlen) or cascade attention on a GPU.
- **What Jev may do (Inferred):** exactly such kernels (RESEARCH.md §3.9).

### W13. No state cache across requests; one request at a time
- **What:** the API computes every state again for every request, and a lock allows one forward pass at a time.
- **Candidate fix:** keep the KV cache of recent states; batch concurrent requests.
- **What Jev may do (Inferred):** the cookbook latencies fit a cross-request state cache (RESEARCH.md §3.9);
  RESEARCH.md §8 test 7 can confirm it.

### W14. Pointwise Scores repeat the question for every level
- **What:** a 5-level Score sends its question text 5 times.
- **Fix (Measured):** a two-level tree (state → question head → level). The question text runs once; each level
  branch sees the state, the head and itself. `MINIJEV_SHARE_QUESTION=true` in `poc/minijev.env` turns it on;
  `Settings()` leaves it off, so the recorded experiments do not change.
- **Rule:** a head is shared only when the tokens are identical to the joined text. If a token merges across the
  split, that question keeps the flat layout.
- **Result (Measured, Qwen2.5-0.5B, GDPR article cut to 512 tokens, 587-token prefix):** the 3 GDPR Score questions
  use 542 branch tokens instead of 691 (−21.6%); the full 13-question request uses 1,543 input tokens instead of
  1,692 (−8.8%). The largest logit difference against the flat layout is 1.9e-5 (packed mode). Tests check naive, kv
  and packed within 1e-3 (`tests/test_engine.py`).

## Measurement

### W15. Small samples, and tests that are not held out
- **Evidence:** agreement with Jev rests on 8–15 cases per type. E1–E13 fitted and reported on overlapping items, and
  the prompt was chosen while the data was visible (W7).
- **Fix, in part:** frozen train / val / test splits (docs/DATA.md). E14 fits on train, chooses on val and reports on
  test: 300 BoolQ questions and 400 AG News articles, so accuracy intervals are about ±0.05 and ±0.035.
- **Still open:** the E1–E13 results remain exploratory; re-running them on the splits would make them defensible.

### W16. Timing noise on one laptop
- **Evidence (Measured):** runs vary by about ±10–20% (WALKTHROUGH.md §6.3). Medians of 3 runs in rotating order
  reduce, but do not remove, the effect.

### W17. Datasets are not pinned
- **What:** the datasets came from the Hugging Face datasets-server API, which has no revision pin, so the rows could
  change between runs without anyone noticing.
- **Fix (Measured):** `poc/datasets/splits_v2.json` freezes every sampled item by its source index and the SHA-256 of
  its row, and records the SHA-256 of each source file. `data.load_split()` re-checks every row and stops on a change;
  `data.py check` verifies all claims of docs/DATA.md. Downloads check TLS certificates.

### W18. No cloud-model baseline
- **What:** every comparison uses the same small model. We do not know how a frontier model compares on the same
  data (DESIGN.md E2).
- **Candidate fix:** run TypeSafe's MIT adapter with an `ANTHROPIC_API_KEY` on the E11 sets.

## Engineering

### W19. The app is a POC
- **Done:** the code is the `minijev` package (`src/minijev/`): `pip install -e .` or `uv sync` in poc/, the
  `minijev ask` and `minijev serve` commands, and the API. The old and the new code give byte-identical logits (Measured).
- **Open:** the last response is lost when a page reloads; the Docker build is not tested (the Docker daemon was not
  running); the Compare page asks every question as a Choice.
