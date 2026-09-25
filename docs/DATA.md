# minijev — Data card

*What data minijev uses, where it comes from, how it is split, what each split is for, and how to check every claim
on this page. `uv run python data.py check` (in `poc/`) verifies the claims from the files; the Data page of the web
app shows the same facts. Evidence labels follow RESEARCH.md: **Stated**, **Observed**, **Inferred**, **Measured**.*

Terms: **train**, **val** and **test** are the three disjoint parts of the data. **Calibration** fits a temperature so
that stated confidence matches accuracy. **ECE** (expected calibration error) is the average gap between stated
confidence and accuracy. A **readout** takes the probabilities of the allowed answer labels from one forward pass.

## 1. Summary

| | BoolQ | AG News |
|---|---|---|
| What it is | A yes/no question about a Wikipedia paragraph | A news article; its topic is one of World, Sports, Business, Sci/Tech |
| Used for | Noul (yes/no) questions | Choice (multiple-choice) questions |
| Source | `google/boolq`, validation split, 3,270 rows | `fancyzhx/ag_news`, test split, 7,600 rows |
| Licence | CC BY-SA 3.0 (Stated, dataset card) | "unknown" on the card; the source says "for research purposes … and any other non-commercial activity" (Stated) |
| Our sample | 1,000 questions: train 500 · val 200 · test 300 | 1,200 articles: train 600 · val 200 · test 400 |
| Label balance | 62% "yes", as in the full split | 25% per topic in every split |

**What this data is not.** It is not training data for a model: minijev has not fine-tuned any model (Measured: the
repository has no training code, and no adapter or fine-tuned model is on disk). The only thing fitted on this data
is a handful of calibration temperatures. Training data for a real use case is future work (section 9).

## 2. Sources and provenance

| | BoolQ | AG News |
|---|---|---|
| Hugging Face repository | `google/boolq`, revision `35b264d0` (2024-01-22) | `fancyzhx/ag_news`, revision `eb185aad` (2024-03-07) |
| Original work | Clark et al. 2019, *BoolQ* | Zhang et al. 2015; articles from the AG news corpus (2004–2005) |
| Download | datasets-server.huggingface.co/rows, 100 rows per request, TLS certificates checked | same |
| Local file | `poc/data/boolq_validation.json`, SHA-256 `1b8fa2e9…` | `poc/data/agnews_test_full.json`, SHA-256 `7ff1b8ea…` |
| Fetched | 2026-09-24 | 2026-09-24 |

The full SHA-256 of each source file and of every sampled row is in `poc/datasets/splits_v2.json`. Loading a split
re-checks each row; a changed row stops the run with an error (Measured, `data.py`).

Why these datasets (Inferred): they are public, labelled, and widely used, so anyone can re-run our numbers; they
match two of minijev's question types (yes/no and multiple choice). We use BoolQ's validation split and AG News's test
split because those are the parts that the dataset authors kept for evaluation; the models we test were never
fine-tuned on them by us.

## 3. How the sample was drawn

The splits were drawn once, on 2026-09-24, from seed 2026, by `data.py build`. The file is frozen: the build refuses
to overwrite it. The splits file has SHA-256 `54bb491d210778dc…`.

- **BoolQ is grouped by passage.** Several questions can share one Wikipedia paragraph (3,270 questions share
  2,938 passages; one passage has 9 questions). All questions about one passage are in the same split. Otherwise
  the model would see a test passage while we fit on train, and the test would be easier than it looks.
- **BoolQ keeps the natural label balance.** The full split is 62.2% "yes". Each passage group goes to the split that
  has room and whose "yes" rate stays closest to 62.2%: train 65.2%, val 62.0%, test 62.0%.
- **AG News is balanced by topic.** Every split has the same number of articles per topic, drawn at random from all
  7,600 rows (1,900 per topic). An earlier sample used 4 blocks of 100 rows and was unbalanced (121 Sci/Tech,
  79 Business).
- **Duplicates.** Exact duplicate AG News texts are removed before sampling; the test split has none (Measured).
- **No overlap.** No source row, no BoolQ passage and no AG News text appears in two splits (Measured, `data.py check`).

## 4. What each split is for

| Split | Used for | Never used for |
|---|---|---|
| **train** | Fitting the calibration temperatures (Noul: temperature and Platt; Choice: one temperature per readout mode) | Reporting results |
| **val** | Choosing between methods: temperature or Platt for Noul; listwise, averaged or pointwise for Choice. Rule: the lowest negative log-likelihood (a proper scoring rule) | Fitting |
| **test** | Reporting the final numbers, once | Fitting or choosing anything |

This is enforced, not only promised:
- `data.load_split()` is the only way to read a split, and it records every access.
- `experiments.py heldout` (E14) fits on train, chooses on val and reports on test. Its results file lists every split
  it read and states what each was used for.
- The calibration file (`poc/calibration/<model>.json`) records the dataset, split and size it was fitted on, and the
  checksum of the splits file.
- Readouts never use the labels, so computing them for val and test is not tuning on them.
- **Option order.** Each AG News article shows its four topics in its own random order, seeded by the article's
  source index (reproducible). Listwise questions depend on the order (docs/WEAKNESSES.md W1); one fixed order for
  every item would hide that sensitivity and could flatter listwise when val chooses the mode.

## 5. Statistics per split

| Dataset | Split | n | Labels | Words per text (median / max) |
|---|---|---:|---|---|
| BoolQ | train | 500 | 326 yes · 174 no · 455 passages | 85 / 813 |
| BoolQ | val | 200 | 124 yes · 76 no · 181 passages | 80 / 304 |
| BoolQ | test | 300 | 186 yes · 114 no · 267 passages | 84 / 511 |
| AG News | train | 600 | 150 per topic | 38 / 106 |
| AG News | val | 200 | 50 per topic | 37 / 74 |
| AG News | test | 400 | 100 per topic | 37 / 137 |

BoolQ questions have a median of 8 words. How precise the test numbers can be (Inferred, binomial approximation):
at n = 300 an accuracy near 0.65 has a 95% interval of about ±0.05; at n = 400 an accuracy near 0.85 has about
±0.035. Differences smaller than that are not claims we can defend.

## 6. How to check this page

```bash
cd poc
uv run python data.py check     # every claim in sections 2–5, from the files; exits 1 if any fails
uv run python data.py summary   # the statistics in section 5
```

`data.py check` verifies: the source files match their SHA-256; every sampled row matches its SHA-256 and label;
the split sizes; no shared row, passage or text between splits; the BoolQ "yes" rate of each split; equal topics in
each AG News split.

## 7. Which results are held out

| Results | Data | Held out? |
|---|---|---|
| E1–E13 (calibration, jevdocs, permutation, quality, fan-out, order bias, …) | Earlier samples of BoolQ and AG News, drawn before these splits | **No.** They are exploratory: temperatures were fitted and results reported on overlapping items, and prompts were chosen while the data was visible |
| E14 (`experiments.py heldout`) | `datasets/splits_v2.json` | **Yes**: fitted on train, chosen on val, reported on test |

Only E14 numbers should be used to defend a claim about calibration or accuracy.

## 8. Limits

- **Pretraining contamination is unknown.** BoolQ (2019) and AG News (2015) are public. Qwen2.5 may have seen them
  in pretraining; nobody outside the Qwen team knows. If it did, accuracy would look better than on new data
  (Inferred).
- **Narrow domains.** English only; Wikipedia paragraphs and 2004–2005 news. A temperature fitted here may not hold for
  support tickets or code (docs/WEAKNESSES.md W3).
- **Label quality.** Both datasets were labelled by their authors; we did not re-check the labels. The label error rate
  of our sample is unknown.
- **Small splits.** See the precision note in section 5.
- **No Score data.** There is no labelled ordinal set yet, so Score answers are never calibrated.
- **Licence.** AG News is for non-commercial research only. It must not become training data for a product.
- **Prompt.** The question template was chosen during E1–E13, while earlier samples were visible. It was not tuned on
  these splits; any future template change must be chosen on val only.

## 9. Future training data (not built yet)

The use case in view is agent routing: does a request need a heavy or a cheap model, and does a tool call need a
human. Rules for that data, so that it can be defended in the same way:
1. **Real before synthetic.** Logs of real sessions are the best source. They contain code and possibly secrets, so
   decide first what may be stored, and redact before anything is saved.
2. **Hold out first.** Freeze train, val and test before any model or prompt sees the data; test is never read to
   choose anything.
3. **Blind generation.** Synthetic items come from a strong model that has not seen the test set; label a sample by
   hand to measure its error rate.
4. **Record everything.** Source, date, licence, labeller and checksum per item, as in this card.
