# Near-Miss Split Report — Phase 3

Script: `src/near_miss/split.py` · Seed: **42** · Method: group-aware deficit-based
greedy stratified assignment.

## Method

- Duplicate feature rows were grouped by full feature-tuple hashing (Phase 2
  found 5 groups of 2 rows; labels within each group are identical).
- Groups — not rows — were assigned to splits: groups within each class were
  shuffled with `RandomState(42)` and greedily given to the split whose class
  quota was furthest behind (deficit-minimizing).
- Priority order honored: 1) no duplicate leakage, 2) class balance,
  3) approximate 70/15/15.

## Final Counts

| Split | Samples | % | class_0 | class_1 | class_1 % |
|---|---|---|---|---|---|
| train | 1,756 | 70.02% | 1,307 | 449 | 25.57% |
| validation | 376 | 14.99% | 280 | 96 | 25.53% |
| test | 376 | 14.99% | 280 | 96 | 25.53% |
| **total** | **2,508** | 100% | 1,867 | 641 | 25.56% |

Exact 70/15/15 was unreachable only insofar as 2-row duplicate groups straddle
boundaries; realized proportions are within 0.02 pp of target. Class ratios are
preserved in every split (74.4/25.6).

## Integrity Verification (all PASS)

| Check | Result |
|---|---|
| Duplicate groups crossing splits | **0** |
| Identical feature rows crossing splits | **0** (independent of grouping) |
| sample_index overlap across splits | **0** |
| Every sample assigned exactly once | 1,756 + 376 + 376 = 2,508 |
| Class distribution preserved per split | yes |

## Files

- `data/processed/near_miss/splits/{train,validation,test}.parquet`
- `data/processed/near_miss/split_assignments.csv` (sample_index, split, event_type)

Re-running `split.py` reproduces these assignments exactly (fixed seed, no
library split shortcut).