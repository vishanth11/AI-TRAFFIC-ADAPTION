# Phase 2 — Data Preparation Report (Member 4)

Date: 2026-09-02 · Script: `src/data_preparation.py` (deterministic, no
randomness, relative paths, originals untouched) · Companion:
[data_dictionary.md](data_dictionary.md), Phase 1 reports in `reports/`.

## 1. Datasets Processed

| Dataset | Role |
|---|---|
| filtered_features_DTC_FM.csv + target_labels_DTC_FM.csv | Near-miss (DTC-FM) |
| 4 accident Parquet shards | Accident detection (canonical source, untouched) |

## 2. Row / Image Counts

| | Original | Processed | Loss |
|---|---|---|---|
| DTC-FM samples | 2,508 | 2,508 | 0 |
| Accident train | 2,122 images | 2,122 index rows | 0 |
| Accident validation | 316 | 316 | 0 |
| Accident test | 325 | 325 | 0 |
| Accident total | 2,763 | 2,763 | 0 |

No image bytes were copied; indexes reference `(source_file, row_index)`.

## 3. DTC-FM Alignment Method

- Positional alignment only (no ID column exists in either file).
- Both files verified at 2,508 rows; read in original order; **no sorting, no
  pre-merge drops**.
- `sample_index` (0..2507) inserted as an explicit internal processing index —
  documented as NOT a real dataset identifier.
- Post-write validation: features byte-identical in the exact Parquet artifact;
  labels unchanged; index monotonic; no new missing values.

## 4. DTC-FM Duplicate Findings

5 duplicate groups (10 rows total), reported in
`data/processed/near_miss/duplicate_report.csv`:

| Group | sample_index | event_type | Verdict |
|---|---|---|---|
| 0 | 283, 284 | 1, 1 | exact duplicate (features + label identical) |
| 1 | 456, 457 | 0, 0 | exact duplicate |
| 2 | 593, 595 | 0, 0 | exact duplicate |
| 3 | 596, 597 | 0, 0 | exact duplicate |
| 4 | 1068, 1071 | 0, 0 | exact duplicate |

**Zero conflicting labels** across duplicate groups — no serious data-quality
issue. All rows preserved in the processed dataset (removal deferred; note that
nearby indices suggest spatial/temporal locality in the source ordering —
Phase 3 should ensure duplicate pairs don't straddle a future train/test split).

## 5. DTC-FM Feature Policy

Full decisions in `data/processed/near_miss/feature_policy.json`. Summary:

| Feature | Decision | Reason | Leakage risk |
|---|---|---|---|
| 9 base numeric + 2 categorical features | keep | original measurements | none |
| PET | keep **with flag** | core surrogate-safety measure | none statistically — but **POST-EVENT: unavailable in real-time deployment** |
| FE_inv_PET | retain, flag redundant | verified 1/PET; pure redundancy | none (redundancy ≠ leakage) |
| FE_dist_squared | retain, flag redundant | verified dist²; scale instability | none |
| FE_log_mesafe | retain, flag redundant | verified ln(1+dist_px) | none |
| FE_safety_index | **FLAGGED SUSPICIOUS** | formula unidentified, provenance unverified | **MEDIUM** — Phase 3 must reverse-engineer or exclude |
| target_dist_px (+ derivatives) | keep with flag | camera-geometry dependent | none |

Nothing was deleted; no scaler was fitted; no feature selection was performed.

## 6. DTC-FM Label Mapping Status

**UNCONFIRMED.** No label map exists in the CSVs and no source documentation
was found locally in Phase 1/2. `label_mapping.json` records neutral
`class_0` / `class_1` terminology with explicit unconfirmed status. The
class-conditional evidence (class 1 = shorter PET, smaller distances, larger
angles) remains *suggestive of* "1 = conflict/near-miss" but is NOT treated as
confirmed. **Do not hard-code semantics in Phase 3.**

## 7. Accident Label Mapping

Confirmed from the source dataset card (Hugging Face
`justjuu/traffic-accident-cctv-object-detection`): `objects.category`
`0 = accident`, `1 = non_accident`. Recorded in
`data/processed/accident/label_mapping.json` using the **actual schema field
name** `objects.category`. The map is not embedded in the Parquet files.

## 8. Accident Annotation Structure

- One record = `image{bytes: PNG, path: null}` + `objects{bbox[], category[]}`.
- Bounding boxes: COCO-style `[x, y, w, h]` absolute pixels, verified again
  (100% within 640×640 bounds across all 5,400 boxes — zero violations).
- Representation preserved unchanged in the index artifacts; no normalization
  or conversion performed.
- 1–12 objects per image (avg 1.95); no image lacks annotations.

## 9. Accident Duplicate Findings

26 exact-byte duplicate images inside TRAIN (SHA-256 grouped), reported in
`data/processed/accident/duplicate_report.csv`:
- **0 groups with conflicting annotations**; duplicates carry identical labels
  and identical bounding boxes.
- No duplicates in validation or test; **zero** cross-split duplicates.
- All rows preserved (no deletion — 1.2% of train, benign, likely augmentation
  artifacts).

## 10. Train / Validation / Test Integrity

`data/processed/accident/split_integrity_report.json`:

| Check | Result |
|---|---|
| Method | SHA-256 of embedded PNG bytes (content, not filename) |
| train ∩ validation | 0 duplicates |
| train ∩ test | 0 duplicates |
| validation ∩ test | 0 duplicates |
| Unique images | 2,737 of 2,763 (26 internal train dups) |
| **Result** | **PASS** |

## 11. Class Distributions

Full data in `reports/class_balance_phase2.json`.

**DTC-FM:** class_0 1,867 (74.44%) / class_1 641 (25.56%) — MODERATELY IMBALANCED.

**Accident object level:**

| Split | accident | non_accident | accident % |
|---|---|---|---|
| train | 2,313 | 1,807 | 56.14% |
| validation | 338 | 340 | 49.85% |
| test | 391 | 211 | 64.95% |

**Image level** (≥1 accident-labeled object): train 98.59%, validation 99.37%,
test 99.69% — nearly every frame contains an accident object.

Flags: no severe imbalance anywhere; **train→test accident-rate drift of
+8.8 points** (object level) noted for Phase 6 evaluation; no rebalancing,
oversampling, undersampling, or augmentation performed.

## 12. Data-Quality Issues

1. DTC-FM label semantics unconfirmed (blocking for narrative, not for modeling).
2. `FE_safety_index` provenance unexplained (MEDIUM).
3. PET post-event availability (deployment design issue).
4. DTC-FM has no provided split — Phase 3 must create one without letting
   duplicate pairs straddle splits.
5. Accident train/test class-ratio drift (+8.8 pts).
6. 26 train-internal duplicate images (harmless; documented).
7. CSV round-trip shifts the last ULP of 23 float values (max 9e-16); exact
   Parquet companion provided (`dtc_fm_combined.parquet`).

## 13. Deployment Considerations

**Camera-dependent features.** `target_dist_px`, `FE_dist_squared`, and
`FE_log_mesafe` are measured in camera pixel space. Pixel distance conflates
physical separation with camera geometry (distance, focal length, mounting
angle). A model relying on these features may not transfer to a different
camera without recalibration. Metric features (`min_dist_dual_check_m`, speeds,
PET) are the safer core; pixel features should be treated as camera-specific
auxiliary signals. No camera calibration was attempted in Phase 2.

**Post-event features.** PET (and its transform `FE_inv_PET`) is only
computable after an interaction completes — fine for offline near-miss
analytics, unavailable for live alerting. If Member 4's deployment target is
real-time, Phase 3 should plan a pre-encroachment feature variant.

**Accident module.** ~99% of frames contain an accident object; the system's
value is localization (which object) and confidence, not frame-level binary
classification.

## 14. Files Created

```
data/processed/
├── near_miss/
│   ├── dtc_fm_combined.csv          2,508 rows, sample_index + 15 features + event_type
│   ├── dtc_fm_combined.parquet      exact-float copy
│   ├── duplicate_report.csv         5 groups, 10 rows, 0 conflicts
│   ├── label_mapping.json           UNCONFIRMED (class_0/class_1)
│   └── feature_policy.json          per-feature keep/flag decisions
└── accident/
    ├── train_index.parquet          2,122 rows
    ├── validation_index.parquet       316 rows
    ├── test_index.parquet             325 rows
    ├── duplicate_report.csv         26 duplicate images, 0 annotation conflicts
    ├── split_integrity_report.json  PASS
    ├── label_mapping.json           0=accident, 1=non_accident (objects.category)
    └── corruption_report.json       0 issues (IHDR all 2,763 + decoded sample every 100th)

reports/
├── phase2_data_preparation.md       this report
├── data_dictionary.md
└── class_balance_phase2.json

src/data_preparation.py              deterministic, re-runnable end-to-end
```

Original datasets: **unmodified** (verified — no writes under `dataset/`).

## 15. Post-Preparation Validation (all PASS)

- DTC-FM: 2,508 rows; features exact vs original (Parquet artifact); labels
  unchanged; sample_index 0..2507; no new missing values.
- Accident: index counts match scans (2,122 / 316 / 325; total 2,763);
  annotation structure preserved verbatim; zero cross-split contamination.

## 16. Unresolved Issues

1. DTC-FM label semantics — requires external documentation from the dataset provider.
2. `FE_safety_index` formula — requires source code/documentation or exclusion in Phase 3.
3. DTC-FM splitting strategy — deferred (must respect duplicate pairs; consider
   stratification on class and object-pair types).

## 17. Recommendations for Phase 3 (Near-Miss Model)

1. Load `data/processed/near_miss/dtc_fm_combined.parquet` (exact floats).
2. Resolve label semantics BEFORE interpreting any model output; train with
   neutral class_0/class_1.
3. First model: exclude `FE_safety_index` (or run with/without as an
   ablation); keep base features + optionally the verified transforms.
4. Create a stratified train/validation split (e.g. 80/20) with a fixed seed,
   ensuring duplicate pairs stay on one side of the split; hold out a test set
   the same way.
5. Handle moderate imbalance via class weights rather than resampling.
6. Encode `class_object_1`/`class_object_2` (one-hot; watch the 2-row HCV rarity).
7. Report both class_0/class_1 metrics and, once semantics are confirmed,
   map them to operational terms.

**No model training was performed. No inference was performed. No original
dataset was modified. STOP HERE — Phase 3 requires explicit instruction.**