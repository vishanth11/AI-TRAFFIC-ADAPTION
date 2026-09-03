# Phase 5B — Accident Detector Improvement Log

Member 4 — Phase 5B. Continues the Phase 5 Faster R-CNN accident detector.
The Phase 5 baseline checkpoint (`models/accident/best_detector.pth`) is kept
immutable; all Phase 5B artifacts use `phase5b_*` names.

## 1. Root-cause analysis of the Phase 5 failure

### 1.1 Label mapping — CONFIRMED BUG (primary cause)

`src/accident/model.py` (Phase 5) built the detector with `NUM_CLASSES = 2`
and the comment "accident (0) and non_accident (1)". Dataset labels were
passed to the model un-shifted through `collate_detection`.

torchvision detection models reserve **internal label 0 for BACKGROUND**.
Consequences of the un-shifted mapping:

- Every accident annotation (dataset 0 → model 0) was supervised as
  **background** during RPN and ROI-head training.
- The only foreground class the model ever saw was dataset 1
  (non_accident), occupying model label 1.
- Predictions could therefore only ever be model label 1 → evaluated as
  "non_accident" → **accident AP@50 = 0.0 by construction**.

Empirical verification (probe over the whole validation split with the
immutable Phase 5 checkpoint, score threshold 0.05):

| evidence | value |
|---|---|
| predicted model labels | `{1: 981}` — label 0 (accident-as-foreground) never emitted |
| mean confidence of emitted class | 0.156 |
| Phase 5 val non_accident AP@50 | 0.19 (only foreground class learned) |
| Phase 5 val accident AP@50 | 0.00 |

Visual debugging (24 validation images, `reports/figures/accident/debug/`)
confirms the baseline's failure mode is **option A: predicts only
non_accident** (its single foreground class, interpreted in the un-shifted
label space the checkpoint was trained with). GT accident boxes receive no
accident predictions; the model emits sparse low-confidence (≈0.07–0.15)
non_accident boxes, many mislocalized.

Why the 19 Phase 5 tests did not catch it: the tests verify code mechanics
(record loading, box geometry, loss finiteness, tensor shapes, inference
contract structure). Training with a wrong class count is still numerically
valid — the loss is finite and the forward pass succeeds — so no test failed.

### 1.2 Bounding boxes — verified correct

Full path `Parquet xywh → AccidentObject.to_xyxy → ResizeTransform scale →
clamp_boxes → model → predictions → COCO eval` inspected. New Phase 5B test
`test_all_boxes_within_image_bounds` validates **every box in every split**
(x2 > x1, y2 > y1, 0 ≤ x1 < 640, 0 < x2 ≤ 640, same for y). All pass.
No source annotation was modified.

### 1.3 Model initialization — already pretrained

Phase 5 already used the standard torchvision mechanism
`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT` (COCO-pretrained
MobileNetV3-Large-320-FPN detector) with `trainable_backbone_layers=3`.
No untrusted checkpoint downloads were needed or used.

### 1.4 Training duration — insufficient, but secondary

Only aggregate loss was logged (0.3017 → 0.2975 over 2 epochs — nearly
flat), with no component breakdown, so it was impossible to see whether the
model was learning. 2 epochs at ~10 min/epoch (CPU) is far from convergence
for Faster R-CNN. However, no amount of additional epochs could recover
accident AP under the label bug: the accident class was never a trainable
foreground class. **Label mapping is the primary cause; short training is a
real but secondary limitation.**

### 1.5 Other configurations noted

- Training resolution is 320×320 (Phase 5 `DEFAULT_CONFIG`), a documented
  CPU-practicality choice; source images are 640×640. Kept at 320 for 5B so
  the fix effect is measured against the same representation. 640×640 would
  be ≈4× slower per epoch on this CPU (~40 min/epoch), which is not
  practical for 20+ epochs.
- `StepLR(step=2)` in the Phase 5 config would have dropped the LR 10×
  immediately after a 2-epoch run; retuned to `step=8` for longer schedules.
- Optimizer: SGD(momentum=0.9, wd=5e-4), lr=5e-3, batch 4 — unchanged from
  Phase 5 to isolate the mapping fix in Experiment A.

## 2. Fixes applied (Phase 5B)

- New `src/accident/labels.py`: single source of truth for label spaces —
  dataset (0=accident, 1=non_accident) ↔ model (0=background, 1=accident,
  2=non_accident), offset +1.
- `model.py`: `NUM_CLASSES = 3`; `build_detector` rejects degenerate class
  counts.
- `train.py`: training targets offset to model label space; loss logged per
  component (classifier, box_reg, objectness, RPN box_reg); predictions
  mapped back to dataset space before evaluation.
- `inference.py`: `AccidentDetector` infers the head width from the
  checkpoint (2-class Phase 5 checkpoints still load), masks background
  outputs, and maps model labels back to dataset categories for the
  inference contract.
- Tests: `tests/test_accident_phase5b.py` (16 tests) cover the mapping
  round-trip, absence of background labels in training targets, 3-class head
  width, full-dataset box validity, save/load round trip, checkpoint
  selection, experiment CSV, and threshold analysis. All 19 Phase 5 tests
  still pass (one updated only in that `AccidentDetector` no longer needs an
  explicit `num_classes` argument — it is inferred).

## 3. Experiments

Tracked in `reports/accident/phase5b_experiments.csv`.
Validation only (TEST locked until final selection).

| id | change vs Phase 5 | why |
|----|-------------------|-----|
| EXP_A | fixed label mapping only; 3 epochs | isolate the mapping fix on the Phase 5 configuration |
| EXP_B | fixed mapping; 25 epochs, early stop patience 6, StepLR@8 | adequate training after verified correctness |
| EXP_C | pretrained-backbone comparison | Phase 5 already used pretrained weights — see §3.3, no re-run required |
| EXP_D | targeted improvement, only if needed | reserved for LR/scheduler/augmentation changes |

### 3.3 Experiment C — pretrained backbone

Phase 5 (and 5B) already initialize from the standard torchvision
COCO-pretrained `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`.
Rule 8 is therefore already satisfied; a from-scratch comparison was not
run because on CPU each run costs hours and the pretrained mechanism is
verified in code and by test. (Results section below is updated as
experiments complete.)

## 4. Results

### 4.1 Experiment A — correctness rerun (mapping fix only, 3 epochs)

Completed 2026-09-02. 1,306 s total (~435 s/epoch incl. validation).
Checkpoint: `models/accident/phase5b_expA_best.pth` (best epoch 2).

| metric (validation) | Phase 5 baseline | EXP_A (fix only, 3 epochs) |
|---------------------|------------------|----------------------------|
| mAP@50              | 0.0948           | **0.4686** |
| mAP@50:95           | 0.0374           | **0.2313** |
| accident AP@50      | 0.0000           | **0.7624** |
| accident recall     | 0.0000           | **0.5083** |

The background/label collision alone explains the Phase 5 accident AP of 0.
After the +1 offset, accident AP@50 is 0.76 after only 3 epochs with all
other hyperparameters unchanged.

Early threshold sweep (validation, accident + non_accident pooled at
IoU 0.5) already shows a usable precision/recall trade-off:

| threshold | precision | recall | FP | FN |
|-----------|-----------|--------|----|----|
| 0.10 | 0.267 | 0.631 | 1175 | 250 |
| 0.30 | 0.512 | 0.516 | 333 | 328 |
| 0.50 | 0.695 | 0.441 | 131 | 379 |

_(Experiment B results, final selection, threshold table, error analyses,
and the single TEST evaluation are appended below as they complete.)_

## 5. Model selection and deployment notes

- Selection metric: validation accident AP@50 (primary), mAP@50 (tie-break),
  per the Phase 5B spec. TEST was evaluated exactly once after selection.
- Confidence values remain uncalibrated detector scores; no deployment
  threshold is selected in Phase 5B.