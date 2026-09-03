# Accident probability contract (Phase 6)

This document defines the meaning, construction, and guarantees of the two
confidence quantities produced by `models/accident/calibrated_accident_detector.pth`.

## Output schema

```json
{
    "detector_version": "accident_calibrated_detector_v1",
    "image_size": [640, 640],
    "detections": [
        {"bbox": [x1, y1, x2, y2], "category": "accident", "confidence": 0.83}
    ],
    "accident_detection_confidence": 0.83,
    "calibrated_accident_probability": 0.71,
    "calibration": {"method": "platt", "signal": "noisy_or_top5", "fit_split": "validation"}
}
```

## Definitions

### `accident_detection_confidence` — detector confidence (NOT a probability)

The raw event-level accident signal: the **noisy_or_top5** aggregation of the
accident-class box scores (above a 0.05 detector floor) produced by the
Faster R-CNN detector. Range [0, 1]; 0.0 means the detector emitted no
accident box above the floor.

- It is a *ranking* score. Higher means the detector is more confident that
  some accident object is present.
- It is **not calibrated**: it does NOT mean "an accident is present with this
  probability". Treat it exactly like any detector score.

### `calibrated_accident_probability` — P(image contains an accident)

The signal mapped through the fitted **platt** calibrator (fitted on
validation image-level labels only; see `calibration_metadata.json`). Range
[0, 1] by construction.

- Semantics: an estimate of the probability that the image/event contains at
  least one accident object, in the sense of the validation prevalence of
  images with equal signal.
- It is a property of the **image/event**, not of any single box.
- Validity is conditional on the deployment distribution resembling the
  dataset distribution (CCTV frames at similar resolution/content). Under
  distribution shift, recalibrate on a NEW validation-like split; never reuse
  TEST for fitting.

## Guarantees

- Deterministic for a fixed image and checkpoint (eval mode, CPU).
- Calibrated output is always in [0, 1]; monotone in the raw signal for the
  selected calibrator.
- The calibrator was fitted **only on validation**; TEST was evaluated once,
  after freezing.
- Original Parquet dataset files are never modified by the pipeline.

## Non-guarantees

- The probability does not encode severity, number of accidents, or
  localization quality.
- Per-box `confidence` values in `detections` remain uncalibrated detector
  scores.
- Nothing in this contract fixes a deployment decision threshold; downstream
  consumers choose their own operating point from the validation threshold
  study (`phase6_summary.md` §3).

## Update policy

To change the signal, method, checkpoint, or thresholds: refit calibration on
validation (or a fresh validation-like split), re-run the threshold study,
bump `detector_version`, and re-freeze. Do not adjust any parameter based on
TEST performance.
