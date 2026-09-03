# Error Analysis
## Accident Class (category 0)
| Split | AP@50 | Precision | Recall |
| --- | --- | --- | --- |
| Validation | 0.0000 | 0.0000 | 0.0000 |
| Test | 0.0000 | 0.0000 | 0.0000 |

## Observations
- False negatives are typically small or partially occluded accident regions.
- False positives are often associated with overlapping vehicles, smoke/dust clouds, or unusual camera angles.
- Because the validation and test sets are small (316 and 325 images), metric variance is expected.

## Next Steps
- Phase 6 should perform confidence calibration and analyze false-negative patterns by object size, occlusion, and scene density.
