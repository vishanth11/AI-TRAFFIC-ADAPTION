# Data Quality Report — Member 4, Phase 1

Overall quality grades per dataset, with the evidence behind each grade.

| Dataset | Grade | Key evidence |
|---|---|---|
| filtered_features_DTC_FM.csv | **GOOD** | 0 missing, 0 infinite, no constants, plausible physical ranges, moderate imbalance only |
| target_labels_DTC_FM.csv | **GOOD** | 0 missing, clean binary {0,1}, 2,508 rows aligned positionally with features |
| train-*.parquet (both shards) | **GOOD** | Identical schema, 0 missing bytes, 0 empty annotations; 26 internal duplicate PNGs (1.2%) |
| validation-*.parquet | **GOOD** | Clean, balanced classes (49.9/50.2) |
| test-*.parquet | **GOOD** | Clean; mildly accident-heavy (65%) vs train (56%) |

## Checks Performed

### Missing / invalid data
- CSVs: 0 missing cells across 37,620 feature values + 2,508 labels. 0 infinite values.
- Parquet: 0 images with empty bytes across all 2,763 records; 0 images with zero annotations.

### Duplicates
- CSV features: 5 fully duplicated rows (0.20%) — plausibly genuine identical
  interaction configurations; verify before dropping in Phase 2.
- Parquet: zero cross-split duplicate images (SHA-256 verified over all 2,763 PNGs);
  26 exact-byte duplicates inside train only.

### Schema consistency
- All 4 Parquet shards: byte-identical Arrow schema. No inconsistencies.

### Corrupted-looking records
- None found. All PNG headers parse; all boxes within image bounds (x+w ≤ 640, y+h ≤ 640 at 100%).

### Suspicious labels
- DTC-FM: label meanings not embedded in files; distribution and class-conditional
  statistics are consistent with 1 = conflict/near-miss, but **semantics require
  confirmation from the dataset documentation**.
- Parquet: class map not embedded in files; confirmed via the source dataset's
  public README (0 = accident, 1 = non_accident). Keep a local copy of that mapping
  in Phase 2.

### Severe imbalance
- None. DTC-FM labels moderately imbalanced (74.4/25.6). Parquet object classes
  balanced-to-mildly-imbalanced per split.

### Leakage
- DTC-FM: `FE_safety_index` formula unconfirmed (MEDIUM); PET is post-interaction
  by nature (MEDIUM, design consideration); three `FE_*` features are pure
  transforms of base columns (LOW).
- Parquet: no identifiers/timestamps to leak; no cross-split duplicates.

### Split contamination
- None detected (0 cross-split duplicate images; CSVs are a single dataset with
  no pre-existing split).

## Dataset-Level Risks for Later Phases

1. **DTC-FM alignment is positional** — any Phase 2 row filtering must preserve
   feature/label index alignment.
2. **`FE_safety_index`** — unexplained composite with clipped extremes
   (max 28.5M); exclude or reverse-engineer before modeling.
3. **Pixel-distance features** — camera-dependent; metric features (PET, meters)
   are the safer core.
4. **Train/test class-ratio drift (Parquet)** — 56% vs 65% accident objects;
   use per-split stratified metrics in evaluation.
5. **DTC-FM class imbalance (moderate)** — consider class weights in Phase 3;
   no resampling decisions made in Phase 1.