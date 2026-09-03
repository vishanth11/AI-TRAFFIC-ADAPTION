# Phase 6 — Accident probability calibration results

Calibration was fitted and selected **exclusively on the validation split**.
TEST was touched only after the calibrator was frozen (see
`test_results_phase6.md`).

## Data (validation)

- Images: **316** (314 accident-positive, prevalence 99.4%)
- Image-level label: 1 if the image contains >= 1 accident ground-truth box.
- Detector score threshold for building the signal: 0.05 (consistent with the
  Phase 5/5B evaluation pipeline).

**Weak-identification caveat.** The accident dataset is built almost
entirely of accident-containing images, so the image-level label is positive
for ~99% of both calibration and test images. Event-level calibration is
therefore dominated by prevalence: Brier/log loss/ECE on this split are close
to what a well-behaved constant predictor achieves, and the discrimination
power of the detector is better read from the object-level AP and the
threshold study (see `phase6_summary.md` §2-3). The calibrator should be
re-fit on any deployment data whose accident prevalence differs materially
from this split.

## Detector confidence vs calibrated probability

These are two different quantities:

1. **`accident_detection_confidence`** — the raw image/event-level accident
   signal aggregated from accident-class box scores of the detector
   (selected aggregation: **noisy_or_top5**). It is a ranking score in
   [0, 1] and is **not** a probability.
2. **`calibrated_accident_probability`** — the signal mapped through the
   fitted calibrator, interpretable as P(the image contains an accident).

## Signal aggregation (validation, uncalibrated)

| signal | Brier (uncal.) | log loss (uncal.) |
|---|---|---|
| max | 0.0743 | 0.2334 |
| top3_mean | 0.2084 | 0.5755 |
| noisy_or_top5 | 0.0427 | 0.1488 |

Selection rule: lowest uncalibrated validation Brier -> **noisy_or_top5**.

## Calibration methods (validation)

| method | Brier | log loss | ECE | CV Brier mean (5-fold) | CV Brier std |
|---|---|---|---|---|---|
| platt | 0.0063 | 0.0356 | 0.0 | 0.0063 | 0.0076 |
| isotonic | 0.0061 | 0.0277 | 0.0 | 0.0064 | 0.0074 |

Selection rule: lowest validation Brier; when the Brier gap is
<= 0.002 (near tie), held-out 5-fold CV decides — lower CV Brier mean first,
then lower CV log-loss mean (isotonic step functions can score well on
in-sample Brier yet produce extreme held-out probabilities) — with Platt
preferred on an exact tie (smoother, parametric form). Here the Brier gap is
0.0002 (a near tie) and Platt's CV log loss is 0.0382
vs isotonic's 0.1058, so the selected method is **platt**.

## Fitted calibrator

```json
{
  "method": "platt",
  "a": 0.48996558230802595,
  "b": 3.8711533728013694
}
```

Validation fit metrics: Brier 0.0063, log loss
0.0356, ECE 0.0000.

## Reliability table (validation, selected method)

| bin low | bin high | count | mean predicted | observed frequency |
|---|---|---|---|---|
| 0.9 | 1.0 | 316 | 0.9937 | 0.9937 |

## Figures

- `reports/figures/accident/calibration_curve.png`
- `reports/figures/accident/confidence_threshold_curve.png`
- `reports/figures/accident/reliability_diagram.png`

## Leakage guard

- Calibration fitting: validation only.
- Threshold study and signal/method selection: validation only.
- TEST evaluations performed in Phase 6: exactly 1, after freezing.
