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
| W1 | The option order changes listwise answers | Method | High | Fixes measured (E13) |
| W2 | Raw probabilities are overconfident | Calibration | High | Partly: fixed for Noul, open for Choice and Score |
| W3 | Calibration exists only for Noul, and does not transfer | Calibration | High | Open |
| W4 | The 0.5B model is at chance on BoolQ | Model | High | Use 1.5B or larger; open |
| W5 | Pointwise items on small models say "yes" to the plausible item | Method | Medium | Open |
| W6 | Opposite questions are not consistent | Method | Medium | Not measured |
| W7 | Answers depend on the prompt wording | Method | Medium | Open |
| W8 | Literal reading of vague questions | Model | Medium | Partly: Noul `criteria` |
| W9 | Contextual calibration over-corrects | Calibration | Low | Avoided (opt-in) |
| W10 | More than 25 options is not supported | Method | Low | Open |
| W11 | Long states are untested | Model | Medium | Open |
| W12 | The packed pass computes the hidden attention blocks | Speed | Low (CPU), Medium (GPU) | Open |
| W13 | No state cache across requests; one request at a time | Speed | Medium | Open |
| W14 | Pointwise Scores repeat the question for every level | Speed | Low | Open |
| W15 | Small samples, and tests that are not held out | Measurement | Medium | Partly: intervals added |
| W16 | Timing noise on one laptop | Measurement | Low | Partly: medians, rotation |
| W17 | Datasets are not pinned | Reproducibility | Low | Open |
| W18 | No cloud-model baseline | Measurement | Medium | Open (needs an API key) |
| W19 | The app is a POC: no packaging, no persistence, Docker untested | Engineering | Low | Open |

---

## Method

### W1. The option order changes listwise answers
- **What:** in listwise mode the model sees "A) billing B) technical C) sales" and picks a letter. A small model
  likes some letters more than others, whatever the options are. So moving an option can change the winner.
- **Evidence (Measured):** on 8 documented Choices, a rotation of the options changed the winner in 54% of the
  rotations at 0.5B and 9% at 1.5B (RESEARCH.md §7.3 R4). E13 measures it on 120 AG News articles with known answers
  (RESEARCH.md §7.3 R10).
- **Candidate fixes, all measured in E13:**
  - *All orders averaged:* ask once per rotation in the same pass and average each option's probability. It costs
    one branch per option, but the text is still read once. The app offers it as "asked: all orders averaged".
  - *Debiased:* divide out the model's measured liking for each letter (PriDe, Zheng et al. 2024). One branch.
  - *Pointwise:* judge each option alone. It cannot depend on the order, by construction.
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
- **Candidate fix:** freeze the templates, then fit calibration; keep a held-out set; test several templates.
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
- **Candidate fix:** share the question as a second prefix level in the tree (state → question → level).

## Measurement

### W15. Small samples, and tests that are not held out
- **Evidence:** agreement with Jev rests on 8–15 cases per type; E11 uses 200 and 120 items. Intervals are now
  reported; the documented cases were visible while we chose the template (W7).
- **Candidate fix:** larger labelled sets; a held-out split fixed before any template change.

### W16. Timing noise on one laptop
- **Evidence (Measured):** runs vary by about ±10–20% (WALKTHROUGH.md §6.3). Medians of 3 runs in rotating order
  reduce, but do not remove, the effect.

### W17. Datasets are not pinned
- **What:** BoolQ and AG News come from the Hugging Face datasets-server API, which has no revision pin. The GDPR
  article is pinned.
- **Candidate fix:** store a checksum of the downloaded rows; fail loudly when they change.

### W18. No cloud-model baseline
- **What:** every comparison uses the same small model. We do not know how a frontier model compares on the same
  data (DESIGN.md E2).
- **Candidate fix:** run TypeSafe's MIT adapter with an `ANTHROPIC_API_KEY` on the E11 sets.

## Engineering

### W19. The app is a POC
- No pydantic API or CLI package (DESIGN.md §12 Phase 2).
- The last response is lost when a page reloads.
- The Docker path is not tested.
- The Compare page asks every question as a Choice, so a Noul or a Score is not compared in its own form there.
