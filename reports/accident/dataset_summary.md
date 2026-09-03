# Accident Dataset Summary
This report summarizes the object-level accident dataset used for Phase 5 training.
## Image-level and Object-level Distributions
| Split | Images | Objects | Accident | Non-accident | Accident % | Mean objs/img | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2122 | 4120 | 2313 | 1807 | 56.14% | 1.94 | 1 | 1 | 11 |
| validation | 316 | 678 | 338 | 340 | 49.85% | 2.15 | 1 | 1 | 11 |
| test | 325 | 602 | 391 | 211 | 64.95% | 1.85 | 1 | 1 | 12 |

## Key Observations
- Object-level class distribution is far more balanced than image-level prevalence.
- A naive image-level classifier predicting 'accident' for every image would be misleading.
- The meaningful task is **object-level accident localization**.

## Duplicate Handling
- Training duplicates preserved for the baseline: 26 duplicate rows (13 unique hashes).
- Cross-split leakage: 0.
- Original Parquet files were not modified.
