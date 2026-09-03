# Dataset Inventory — Member 4, Phase 1

Generated: 2026-09-02 · Scope: `dataset/` (originals untouched, read-only audit)

## Files Found

| # | File | Relative Path | Type | Size | Rows | Likely Role | Status |
|---|------|---------------|------|------|------|-------------|--------|
| 1 | filtered_features_DTC_FM.csv | dataset/filtered_features_DTC_FM.csv | CSV | 296.4 KB | 2,508 × 15 | Near-miss / traffic-conflict features | Verified |
| 2 | target_labels_DTC_FM.csv | dataset/target_labels_DTC_FM.csv | CSV | 12.3 KB | 2,508 × 1 | Near-miss / conflict binary labels | Verified |
| 3 | train-00000-of-00002.parquet | dataset/train-00000-of-00002.parquet | Parquet (PNG + annotations) | 361.5 MB | 1,061 | Accident CCTV detection — train shard 1/2 | Verified |
| 4 | train-00001-of-00002.parquet | dataset/train-00001-of-00002.parquet | Parquet (PNG + annotations) | 369.7 MB | 1,061 | Accident CCTV detection — train shard 2/2 | Verified |
| 5 | validation-00000-of-00001.parquet | dataset/validation-00000-of-00001.parquet | Parquet (PNG + annotations) | 120.8 MB | 316 | Accident CCTV detection — validation split | Verified |
| 6 | test-00000-of-00001.parquet | dataset/test-00000-of-00001.parquet | Parquet (PNG + annotations) | 115.1 MB | 325 | Accident CCTV detection — test split | Verified |

Totals: **2 CSV datasets** (near-miss), **4 Parquet shards** (accident detection), ~1.01 GB Parquet media.

## Verified Roles (from contents, not filenames)

### filtered_features_DTC_FM.csv
15 columns of traffic-interaction measurements for object pairs: speeds, angle,
PET, accelerations, minimum distance, object counts, pixel distance, plus 4
`FE_*` engineered features and two object-class columns. **Features table.**

### target_labels_DTC_FM.csv
Single column `event_type` with binary values {0.0, 1.0}. **Label vector.**

### Parquet shards
Hugging Face image-object-detection schema: `image{bytes: PNG, path}` +
`objects{bbox: list<float32[4]>, category: list<int64>}`. All four shards
byte-identical schema. **Accident CCTV object-detection dataset.**

## Originals Untouched
All files remain in `dataset/` unmodified. No renames, moves, conversions, or
deletions were performed. Audit scripts live in `audit_scripts/`; outputs in `reports/`.