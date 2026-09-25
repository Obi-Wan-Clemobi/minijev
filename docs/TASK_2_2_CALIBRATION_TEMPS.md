# Task 2.2 — Calibration temperatures (W2/W3)

*How minijev's calibration temperatures are fitted, stored and used. Every value is fitted on labelled data; none is
estimated or typed by hand.*

## Where the values come from

`experiments.py heldout` (E14) fits them, for each model, on the frozen splits of `poc/datasets/splits_v2.json`
(docs/DATA.md):

| Primitive | Data | Fitted on | Chosen on | Reported on |
|---|---|---|---|---|
| Noul (yes/no) | BoolQ | train (500): a temperature, and Platt (a·z + b) | val (200): the one with the lower negative log-likelihood | test (300) |
| Choice | AG News | train (600): one temperature per readout mode (listwise, averaged, pointwise) | val (200): the mode with the lowest negative log-likelihood | test (400) |
| Score | SST-5 | not fitted yet: SST-5 train (300) is kept for it | val (200): plain or contrastive levels (E16) | test (300) |

The result is `poc/calibration/<model>.json`. It records the fitted values, the chosen Noul calibrator and Choice
mode, and its provenance: the dataset, split and size per primitive, the splits-file checksum, the prompt fingerprint
and the date.

## How the app uses them

- `poc/minijev.env` sets `MINIJEV_CALIBRATION=fitted` and `MINIJEV_CHOICE_MODE=selected`: `ask()` and the API use the
  fitted file of the loaded model, and the Choice mode that val chose.
- `MINIJEV_TEMP_NOUL`, `MINIJEV_BIAS_NOUL`, `MINIJEV_TEMP_CHOICE` and `MINIJEV_TEMP_SCORE` stay `fitted` unless you type a
  number, which then overrides the fitted value for that primitive.
- `Settings.calibrator(qtype, mode)` in `src/minijev/settings.py` returns the temperature, the bias, and the source
  (`manual`, `fitted` or `none`). The web page uses the same rule (`calibrator()` in `web/lib/scoring.ts`); a test on each
  side checks the same cases.
- The experiments use `Settings()`, which is uncalibrated, so the measured results never depend on this file.
- The Calibration panel of the web app shows the provenance, a "fitted" or "manual" badge per dial, and Reset returns a
  dial to the fitted value.

## Checks

```bash
cd poc
uv run pytest tests/test_maths.py -k calibration    # the fitted file is used, and an explicit dial wins
uv run python data.py check                         # the data the values were fitted on
cd ../web && npm test                               # the same rule in the browser
```
