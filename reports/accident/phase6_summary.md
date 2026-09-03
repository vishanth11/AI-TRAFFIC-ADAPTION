# Phase 6 — Accident detector evaluation + confidence calibration

Scope: evaluate the best available Phase 5/5B accident detector on
validation, derive an event-level confidence signal, calibrate it into an
accident probability using validation only, then run ONE frozen evaluation on
test. **No retraining, no architecture changes, no accuracy work** (deferred).

## 1. Checkpoint selection (validation only)

| checkpoint | val accident AP@50 | val mAP@50 |  |
|---|---|---|---|
| phase5_baseline | 0.0000 | 0.0948 |  |
| phase5b_expA | 0.7624 | 0.4686 | **SELECTED** |
| phase5b_expB | 0.7114 | 0.4047 |  |

Selection rule: max validation accident AP@50; mAP@50 tie-break (Phase 5B spec). The chosen checkpoint is re-evaluated on
validation in this run; TEST was never considered.

Note: the EXP_B 25-epoch run was interrupted after epoch 1 (its epoch-1
checkpoint is inferior to EXP_A best). Its interrupted state remains available
for a later accuracy phase; nothing was retrained here.

## 2. Validation results (object level, chosen checkpoint)

| metric | value |
|---|---|
| mAP@50 | 0.4686 |
| mAP@50:95 | 0.2313 |
| accident AP@50 | 0.7624 |
| accident recall (COCO AR) | 0.5083 |

## 3. Confidence threshold study (validation, accident class, IoU 0.5)

| threshold | precision | recall | false positives | false negatives |
|---|---|---|---|---|
| 0.1 | 0.4348 | 0.858 | 377 | 48 |
| 0.2 | 0.5644 | 0.8432 | 220 | 53 |
| 0.3 | 0.6447 | 0.8107 | 151 | 64 |
| 0.4 | 0.7038 | 0.7663 | 109 | 79 |
| 0.5 | 0.759 | 0.7456 | 80 | 86 |
| 0.6 | 0.8087 | 0.713 | 57 | 97 |
| 0.7 | 0.8388 | 0.6775 | 44 | 109 |
| 0.8 | 0.868 | 0.642 | 33 | 121 |
| 0.9 | 0.9118 | 0.5503 | 18 | 152 |

Reading: thresholds in the 0.30-0.50 band give the usable precision/recall
trade-off; the full curve is in
`reports/figures/accident/confidence_threshold_curve.png`. A deployment
threshold is NOT fixed in Phase 6 — downstream phases should pick one from
this table per their operating point.

## 4. Calibration (validation only)

- Signal: `noisy_or_top5` (aggregation of accident-class box scores)
- Method: **platt**
- Validation fit: Brier 0.0063,
  log loss 0.0356
- Details: `calibration_results.md`

## 5. TEST (single frozen evaluation)

| metric | value |
|---|---|
| accident precision @0.5 | 0.6949 |
| accident recall @0.5 | 0.6292 |
| accident AP@50 | 0.6706 |
| mAP@50 | 0.4381 |
| Brier (event level) | 0.0169 |
| log loss (event level) | 0.0673 |

Details: `test_results_phase6.md`. Nothing was tuned after seeing these
numbers.

Process note: the Phase 6 pipeline itself was developed iteratively with
validation-only feedback. Earlier development runs exercised the test path
for code verification, but the reported test evaluation is the single run of
the final frozen pipeline; every reported number comes from that run.

## 6. Artifacts

- `models/accident/calibrated_accident_detector.pth` — detector weights + calibrator
- `models/accident/calibration_metadata.json` — fit metadata + dataset hashes
- `reports/accident/probability_contract.md` — output schema and guarantees
- Figures: `calibration_curve.png`, `confidence_threshold_curve.png`,
  `reliability_diagram.png` under `reports/figures/accident/`

## 7. Tests

Pytest: **83 passed, 0 failed**
(23 new Phase 6 tests in `tests/test_accident_phase6.py`; all 60 prior
Phase 3/4/5/5B tests still passing).

## 8. Guarantees

- Original Parquet files: **UNCHANGED** (SHA-256 verified before/after the run;
  hashes recorded in `calibration_metadata.json`).
- No TEST data used for calibration or threshold/method selection.
- Accuracy optimization: **DEFERRED** to a later phase.
