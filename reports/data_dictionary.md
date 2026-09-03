# Data Dictionary — Member 4

Companion to `reports/phase2_data_preparation.md`. Column semantics from the
Phase 1/2 audits; label semantics marked where unconfirmed.

## Part 1 — DTC-FM Near-Miss Dataset

Processed file: `data/processed/near_miss/dtc_fm_combined.csv` (+ `.parquet` exact-float copy).

| Column | Type | Unit | Meaning | Missing | Origin | Leakage concern | Deployment concern |
|---|---|---|---|---|---|---|---|
| sample_index | int64 | — | **Internal processing index 0..2507 (row position). NOT a real dataset ID.** | 0 | generated Phase 2 | none | none |
| speed_object_1_kph | int64 | km/h | Speed of object 1 in the interaction | 0 | original | none | none |
| speed_object_2_kph | int64 | km/h | Speed of object 2 in the interaction | 0 | original | none | none |
| angle_degrees | float64 | degrees | Encounter angle between objects (0.01–89.98) | 0 | original | none | none |
| PET | float64 | s | Post-Encroachment Time between object pairs (0.53–3.00) | 0 | original | none statistical | **POST-EVENT: only measurable after the interaction completes — unavailable for real-time decisions** |
| acceleration_obj1_mps2 | float64 | m/s² | Longitudinal acceleration of object 1 (−5.0–5.0, quantized ~0.56 steps) | 0 | original | none | none |
| acceleration_obj2_mps2 | float64 | m/s² | Longitudinal acceleration of object 2 (−5.0–4.44) | 0 | original | none | none |
| min_dist_dual_check_m | float64 | m | Minimum separation distance under dual checking (0.5–25.71) | 0 | original | none | none |
| object_count_at_frame1 | int64 | count | Objects detected in the reference frame (5–57) | 0 | original | none | none |
| target_dist_px | float64 | px | Distance between objects in camera pixel space (0–613.4) | 0 | original | none | **camera-dependent: not transferable across camera geometries** |
| FE_inv_PET | float64 | 1/s | Engineered, verified `1/PET` | 0 | derived (verified) | redundancy only | inherits PET post-event limitation |
| FE_safety_index | float64 | — | Engineered, **formula UNIDENTIFIED**; ≈ `(1/PET)/dist_px × unknown term`; extremes clipped at round caps (max 2.85e7) | 0 | derived (unverified) | **MEDIUM — unverified provenance** | unknown |
| FE_log_mesafe | float64 | ln(px) | Engineered, verified `ln(1 + target_dist_px)` ("mesafe" = distance) | 0 | derived (verified) | redundancy only | camera-dependent |
| FE_dist_squared | float64 | px² | Engineered, verified `target_dist_px²` | 0 | derived (verified) | redundancy only | camera-dependent |
| class_object_1 | str | — | Object 1 type ∈ {Car, VRU, Ped, LCV, HCV} | 0 | original categorical | none | none |
| class_object_2 | str | — | Object 2 type ∈ {Car, VRU, Ped, LCV} (no HCV) | 0 | original categorical | none | none |
| event_type | int64 | — | **Binary target. `0 = class_0`, `1 = class_1`. Semantics UNCONFIRMED** — no label map in the files; class-conditional statistics are *consistent with* 1 = conflict/near-miss but this is not documentation-confirmed | 0 | original label | — | — |

### Notes
- Class counts: class_0 = 1,867 (74.44%), class_1 = 641 (25.56%) — moderately imbalanced.
- Acceleration values are quantized (~0.56 m/s² steps, likely per-frame estimation).
- `class_object_1 = HCV` occurs in only 2 rows; treat as rare category in Phase 3.
- No train/val/test split exists in the source; splitting is deferred to a later phase.

## Part 2 — Accident CCTV Object-Detection Dataset

Canonical source: the 4 Parquet shards in `dataset/` (never copied into
processed outputs). Processed indexes: `data/processed/accident/{train,validation,test}_index.parquet`.

| Field | Type | Meaning | Structure |
|---|---|---|---|
| image | struct<bytes: binary, path: string> | CCTV frame, embedded PNG | 640×640 px; `path` is always null (bytes only) |
| image.bytes | binary | PNG-encoded image | verified decodable on a sampled basis; IHDR-validated on all 2,763 |
| objects | struct<bbox, category> | annotation container | one struct per image |
| objects.bbox | list<fixed_size_list<float32, 4>> | per-object box, **COCO-style [x, y, w, h] absolute pixels** | 1–12 boxes/image; 100% within image bounds (x+w ≤ 640, y+h ≤ 640) |
| objects.category | list<int64> | per-object class id | `0 = accident`, `1 = non_accident` (per source dataset card; map not embedded in files) |

### Index fields (processed artifact)

| Column | Type | Meaning |
|---|---|---|
| split | str | train / validation / test |
| source_file | str | original parquet shard name |
| row_index | int64 | 0-based row WITHIN source_file — internal reference is `(source_file, row_index)`; **not a dataset ID** |
| image_sha256 | str | content hash of the embedded PNG (dup/integrity key) |
| image_format, width, height | str/int32 | IHDR-derived; all PNG 640×640 |
| n_objects | int32 | annotation count for the image |
| categories | list<int64> | copy of objects.category |
| bbox | list<list<float32[4]>> | copy of objects.bbox (unchanged representation) |

### Usage in future pipeline
- Phase 5 (accident model) reads images by resolving `(source_file, row_index)`
  against the canonical shards; the index supplies annotations without loading bytes.
- No bounding-box normalization was performed (Phase 2 policy: preserve
  representation unless a downstream model requires conversion).
- Image-level accident prevalence is ~99% (nearly every frame contains ≥1
  accident-labeled object), so image-level classification is trivially biased;
  the meaningful task is object-level detection/localization.