# Near-Miss Dataset Audit — DTC-FM (Member 4, Phase 1)

Files audited (read-only):
- `dataset/filtered_features_DTC_FM.csv` — 2,508 rows × 15 cols, 0 missing
- `dataset/target_labels_DTC_FM.csv` — 2,508 rows × 1 col (`event_type`), 0 missing

Audit script: `audit_scripts/csv_audit.py`

---

## 1. Data Quality Snapshot

| Check | Result |
|---|---|
| Rows / columns | 2,508 × 15 (features); 2,508 × 1 (labels) |
| Missing values | 0 in both files |
| Infinite values | None |
| Constant columns | None |
| Fully duplicated feature rows | 5 (0.20%) |
| Duplicate labels file | 2,506 (expected — label-only file) |
| Numeric ranges | All physically plausible (see §3) |

## 2. Feature ↔ Label Relationship

- Row counts match exactly: **2,508 = 2,508**.
- The labels file has **no ID column** — only `event_type`.
- Therefore the join key is **row position** (positional alignment), not an ID.
- No merge was performed; alignment was verified by count and by sanity of
  class-conditional statistics (§5).

**Recommended Phase 2 action:** after any row filtering/de-duplication, merge
strictly by row index in a single operation, and re-verify `len(X) == len(y)`
afterwards. The 5 duplicated feature rows are likely genuine identical
interaction configurations, not copies — verify before dropping.

## 3. Feature Groups (from column names)

| Group | Columns |
|---|---|
| Speed | `speed_object_1_kph`, `speed_object_2_kph` |
| Acceleration/deceleration | `acceleration_obj1_mps2`, `acceleration_obj2_mps2` |
| Geometry / distance | `angle_degrees`, `min_dist_dual_check_m`, `target_dist_px` |
| Surrogate safety (conflict) | `PET` (post-encroachment time, s) |
| Scene / temporal context | `object_count_at_frame1` |
| Object type (categorical) | `class_object_1`, `class_object_2` ∈ {Car, VRU, Ped, LCV, HCV} |
| Engineered (`FE_`) | `FE_inv_PET`, `FE_safety_index`, `FE_log_mesafe`, `FE_dist_squared` |

No TTC/PET-family features beyond PET, no timestamps, no trajectory IDs, no
video/frame IDs are present.

Object class distributions (`class_object_1` / `class_object_2`):
Car 59.3% / 40.7%, VRU 20.5% / 36.5%, Ped 15.7% / 19.6%, LCV 4.4% / 3.3%,
HCV 0.08% (2 rows) / absent.

## 4. Engineered-Feature Verification (reverse-engineered, evidence-based)

| Feature | Verified formula | Max abs error |
|---|---|---|
| `FE_inv_PET` | `1 / PET` | 3.6e-06 |
| `FE_dist_squared` | `target_dist_px²` | 5.8e-11 |
| `FE_log_mesafe` | `ln(1 + target_dist_px)` ("mesafe" = distance) | 8.9e-16 |
| `FE_safety_index` | Composite: behaves as `inv_PET/dist_px × k` where k varies per row (1.3–75.6); additional input term not identified. Extreme values look clipped at round caps (28.5M, 14M, 12.5M). | n/a — **flagged for investigation** |

Note: the pixel-distance features (`target_dist_px`, `FE_dist_squared`,
`FE_log_mesafe`) are camera-frame dependent. PET and metric distance are the
physically meaningful safety measures; pixel distance mixes scale with camera
geometry.

## 5. Label Semantics

- Column: `event_type`; 2 classes: `0.0` = 1,867 (74.44%), `1.0` = 641 (25.56%).
- The files themselves contain **no label map** — semantics are not confirmed by
  the data alone.
- Circumstantial evidence that `1` = conflict/near-miss class: class-1 rows have
  shorter PET (1.85 s vs 2.38 s), much smaller minimum distance (2.04 m vs 6.26 m),
  much smaller pixel distance (26 px vs 104 px), and larger encounter angle
  (47.8° vs 29.5°) — i.e. the physically more dangerous class is labeled 1.
- **Label semantics require confirmation from the dataset documentation.**
  Do not hard-code 0/1 meanings in Phase 2 until confirmed.

## 6. Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0.0 | 1,867 | 74.44% |
| 1.0 | 641 | 25.56% |

Verdict: **MODERATELY IMBALANCED** (~2.9:1). No resampling in Phase 1.

## 7. Leakage Audit (preliminary)

| Column | Reason for concern | Severity | Recommended action |
|---|---|---|---|
| `FE_inv_PET` | Direct algebraic transform of `PET` (verified 1/PET). Duplicates PET exactly; no new information, but not target-derived. | LOW | Keep one of PET / FE_inv_PET, or let regularization handle collinearity (r = 0.92). |
| `FE_dist_squared` | `target_dist_px²` (verified). Pure duplicate information, adds scale instability. | LOW | Drop or standardize in Phase 2. |
| `FE_log_mesafe` | `ln(1 + target_dist_px)` (verified). Duplicated information. | LOW | Same as above. |
| `FE_safety_index` | Formula not fully identified; composite of engineered terms with unknown provenance. If it was constructed using label statistics or a safety rule that encodes the outcome, it could leak. Correlation with label is only +0.18 (weaker than raw PET), so direct label encoding is unlikely. | MEDIUM | Reverse-engineer formula or drop before modeling; do not treat as a trusted feature until understood. |
| `PET` | PET is a *retrospective* surrogate measure — it can only be computed after the interaction completes. In a real-time near-miss system it is not available at decision time. For an offline classifier this is a design consideration, not leakage per se. | MEDIUM | Document as "post-interaction feature"; if the final system must run live, plan a pre-encroachment feature variant in a later phase. |
| `target_dist_px` / pixel units | Camera-geometry dependent; not leakage but a domain-shift risk if CCTV calibration changes. | LOW | Note for Phase 2 normalization strategy. |

No ID columns exist, so no ID-based leakage is possible. No timestamps or
future-event columns are present.

## 8. Data Quality Verdict

**GOOD** — 0 missing, 0 infinite, plausible ranges, moderate imbalance, 5 benign
duplicate rows, no ID/ID-based leakage. Only concerns are collinear engineered
features and the unconfirmed `FE_safety_index` formula.