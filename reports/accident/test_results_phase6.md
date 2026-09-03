# Phase 6 — Final TEST evaluation

This evaluation ran **once**, after the checkpoint, signal aggregation, and
calibrator were frozen on validation. No parameter was tuned after these
numbers were produced.

- Pipeline: `C:\Users\Varun\Desktop\acci\models\accident\phase5b_expA_best.pth` + platt calibration on the
  `noisy_or_top5` signal.
- Test images: **325** (324 accident-positive).

## Detection metrics (test, object level, IoU 0.5 basis)

| metric | value |
|---|---|
| mAP@50 | 0.4381 |
| mAP@50:95 | 0.2116 |
| accident AP@50 | 0.6706 |
| accident AP@50:95 | 0.3344 |
| accident recall (COCO AR) | 0.4409 |
| non_accident AP@50 | 0.2056 |

## Accident class precision / recall (test, score threshold 0.50)

The 0.50 threshold was fixed from the validation study and applied unchanged.

| metric | value |
|---|---|
| accident precision | 0.6949 |
| accident recall | 0.6292 |

## Calibrated probability metrics (test, event level)

| metric | value |
|---|---|
| images | 325 |
| accident-positive images | 324 |
| Brier score | 0.0169 |
| log loss | 0.0673 |
| expected calibration error (10 bins) | 0.0176 |

For reference, the validation fit metrics of the same frozen calibrator were:
Brier 0.0063, log loss 0.0356.

## Reliability table (test)

| bin low | bin high | count | mean predicted | observed frequency |
|---|---|---|---|---|
| 0.0 | 0.1 | 5 | 0.0523 | 1.0 |
| 0.9 | 1.0 | 320 | 0.9938 | 0.9969 |

## Descriptive failure-mode notes (no tuning derived from these)

Event-level errors concentrate in two modes:

- **Missed images** (signal 0.0 -> floor probability 0.052): 5 of
  324 accident-positive test images received no
  accident-class detection above the 0.05 score floor. These dominate the
  log loss.
- **The single accident-negative test image** received probability
  0.994 (a confident false positive at event level).

These observations are descriptive only; per the Phase 6 rules nothing was
changed after the test evaluation. Detector accuracy work is deferred.

## Figures

- `reports/figures/accident/reliability_diagram.png` (validation and test panels)
