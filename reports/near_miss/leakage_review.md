# Leakage Review — Phase 3 (pre-selection)

Final leakage review before model selection, per the Phase 3 checklist.

## 1. FE_safety_index — provenance RESOLVED, no leakage

Phase 2 flagged this MEDIUM (unverified provenance). Phase 3 investigation
established the exact formula:

```
FE_safety_index = (speed_object_1_kph + speed_object_2_kph)
                  / (2 * max(target_dist_px, 1e-6))
```

- Verified to 3 decimals on all 2,457 rows with `target_dist_px > 0` (max abs
  err 0.0012, consistent with the kph-quantization of the speed columns).
- The 51 zero-distance rows use a 1e-6 floor, which produces the quantized
  extremes (multiples of 250,000 up to 2.85e7) noted in Phase 1/2.
- **It is a pure deterministic transform of three INPUT features.** No target
  information is involved → leakage severity: **NONE** (downgraded from MEDIUM).
- Remaining concerns are statistical redundancy (with speeds and
  `target_dist_px`) and numerical instability (the 1e-6 floor), not leakage.
- Empirical confirmation: including it (config B) does not improve any
  validation metric (see model_evaluation.md § Configuration Comparison). A
  leaky feature would show a suspicious validation jump — none observed.

## 2. PET — deployment availability, not statistical leakage

- PET correlates with the label (r = −0.44) and dominates RF importance
  (0.377) — but it is a legitimate surrogate-safety measure computed from the
  interaction itself, not from the label.
- The concern is **deployment availability**: PET requires the completed
  post-encroachment window. A production near-miss system that must alert
  DURING the interaction cannot use it.
- Config D (deployment_realistic: PET, FE_inv_PET, FE_safety_index excluded)
  documents this cost: val ROC 0.931→0.914 (RF), 0.916→0.898 (LR). **No
  model trained here may be described as real-time-deployable** while PET is
  included.

## 3. sample_index — excluded

- Verified absent from every model's feature list (see tests).
- Positional index carries no semantic meaning; even if it leaked into
  training, the split integrity check guarantees no split benefit.

## 4. Target-derived features — none

- No column in X contains or derives from `event_type`. The only target copy
  lives in `event_type` / `split` columns, which are stripped before
  modeling (`train.py` passes explicit feature lists).

## 5. Duplicate rows — contained

- 5 duplicate groups / 10 rows; group-aware split kept every group inside one
  split (verified: 0 crossing groups; 0 identical feature rows across splits;
  0 sample_index overlap).
- Within-train duplicates (4 groups) at most inflate train counts by 4/1,756
  — no evaluation effect.

## 6. Split integrity

- Group-aware, stratified, seeded (42). Details: `split_report.md`. PASS.

## 7. Overfitting note (not leakage, but honesty)

- random_forest_A: train ROC 0.9996 vs validation 0.9305 — the forest nearly
  memorizes train. Validation/test metrics (0.931/0.929) remain consistent,
  so no train-test contamination is indicated, but probability calibration
  (Phase 4) is warranted before the outputs are used as risk scores.

## Verdict

| Item | Severity | Status |
|---|---|---|
| FE_safety_index | NONE (was MEDIUM) | resolved — redundant transform |
| PET | design concern | documented via config D; no real-time claims |
| sample_index | none | excluded from features (tested) |
| Target-derived features | none | none exist |
| Duplicate leakage | none | group-aware split verified |
| Split contamination | none | PASS |
| RF overfit | caution | Phase 4 calibration recommended |

**Pre-selection leakage review: PASS.** Primary candidate
(random_forest_A) selected on validation metrics only.