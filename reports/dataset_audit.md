# Member 4 Dataset Audit — Phase 1 Master Summary

Date: 2026-09-02 · Status: **PHASE 1 COMPLETE** · No models trained, no
inference run, no dataset modifications.

Detailed reports:
- [dataset_inventory.md](dataset_inventory.md) — file-by-file inventory
- [near_miss_dataset_audit.md](near_miss_dataset_audit.md) — DTC-FM deep audit
- [accident_dataset_audit.md](accident_dataset_audit.md) — CCTV Parquet deep audit
- [data_quality_report.md](data_quality_report.md) — quality grades & risks

## Definitive Dataset Mapping

| Dataset | Member 4 Module | Purpose | Confidence |
|---------|------------------|---------|------------|
| filtered_features_DTC_FM.csv | Near-Miss / Traffic Conflict Detection | Input features for interaction-level conflict classification (2,508 × 15) | High |
| target_labels_DTC_FM.csv | Near-Miss / Traffic Conflict Detection | Binary `event_type` labels, positionally aligned with features | High |
| train-0000{0,1}-of-00002.parquet | Accident Detection | CCTV object-detection training shards (2,122 images) | High |
| validation-00000-of-00001.parquet | Accident Detection | Validation split (316 images) | High |
| test-00000-of-00001.parquet | Accident Detection | Test split (325 images) | High |

The two dataset families serve **different modules** and must not be combined:
DTC-FM → surrogate-safety / near-miss analytics; CCTV Parquet → vision-based
accident detection.

## Key Facts

- **DTC-FM**: 2,508 interaction records; 13 numeric features + 2 categorical;
  4 engineered `FE_*` features (3 verified as pure transforms, 1 unexplained);
  labels 74.4% / 25.6% (moderately imbalanced); aligned positionally (no IDs).
- **CCTV Parquet**: 2,763 PNG images @640×640; 5,400 COCO-style [x,y,w,h]
  bounding boxes; 2 classes (0 = accident, 1 = non_accident per source README);
  schemas identical across all splits; zero cross-split contamination;
  26 exact duplicate images within train.

## Recommended Next Phase

Phase 1 stops here. The expected sequence remains:

```
PHASE 1  Dataset Audit            ← COMPLETE
PHASE 2  Data Preparation         (next: DTC-FM positional merge + FE_* decisions;
                                   Parquet loader strategy + local label-map file)
PHASE 3  Near-Miss Model
PHASE 4  Near-Miss Evaluation
PHASE 5  Accident Detection Model
PHASE 6  Accident Model Evaluation
PHASE 7  Risk Fusion
PHASE 8  Confidence Engine
PHASE 9  Safety Event Engine
PHASE 10 Member 5 Integration
```

Awaiting instructions before any Phase 2 work.