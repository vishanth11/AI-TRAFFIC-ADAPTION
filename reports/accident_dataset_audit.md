# Accident Dataset Audit — CCTV Parquet (Member 4, Phase 1)

Files audited (read-only, metadata + row-group streaming only — no full load):
- `dataset/train-00000-of-00002.parquet` (1,061 rows) + `train-00001-of-00002.parquet` (1,061 rows)
- `dataset/validation-00000-of-00001.parquet` (316 rows)
- `dataset/test-00000-of-00001.parquet` (325 rows)

Audit scripts: `audit_scripts/parquet_audit.py`, `audit_scripts/parquet_labels_audit.py`,
`audit_scripts/bbox_format_probe.py`

---

## 1. Dataset Identification

**Classification: D. Accident + object detection** (image-level accident-scene
CCTV frames with per-object bounding-box annotations). It is not
classification-only (each record carries objects), not trajectory data, and not
video (independent frames).

- Schema: standard Hugging Face image-object-detection format —
  `image: struct<bytes: binary, path: string>` and
  `objects: struct<bbox: list<float32[4]>, category: list<int64>>`.
- Images: embedded **PNG** bytes, all **640 × 640 px** (sampled n=2,763 via IHDR headers).
- `image.path` is `None` everywhere (bytes only).
- Provenance (split sizes 2,122 / 316 / 325 match exactly): Hugging Face dataset
  [`justjuu/traffic-accident-cctv-object-detection`](https://huggingface.co/datasets/justjuu/traffic-accident-cctv-object-detection)
  (CC0-1.0), sourced from a Roboflow "Accident and Non-accident" CCTV dataset,
  letterboxed to 640×640, with horizontal-flip (50%) and ±20% brightness augmentation.

## 2. Annotation Structure

Record = `image + objects{bbox[], category[]}`.

- **Bounding boxes**: COCO-style **[x, y, w, h] in absolute pixels**.
  Verified empirically: 100% of sampled boxes satisfy x+w ≤ 640 and y+h ≤ 640,
  while only 16.7% would be valid as corner coordinates — decisive evidence for
  width/height format, not [x1, y1, x2, y2].
- **Category**: integer class ids 0 and 1 only.
- **Class map** (from the dataset's public README — **not embedded in the files**):
  `0 = accident`, `1 = non_accident`.
- Every image has ≥ 1 annotated object (no zero-object images).
- No timestamps, frame IDs, confidence values, or image IDs are present in the schema.

## 3. Split Inventory

| Split | Shards | Images | Objects | Objects/image |
|---|---|---|---|---|
| train | 2 | 2,122 | 4,120 | 1.94 |
| validation | 1 | 316 | 678 | 2.15 |
| test | 1 | 325 | 602 | 1.85 |
| **total** | 4 | **2,763** | **5,400** | 1.95 |

Objects per image range 1–12; multi-object images are common (train: 2+ objects
in 994/2,122 images).

## 4. Train/Validation/Test Schema Compatibility

All four shards have a **byte-identical Arrow schema**
(`image{bytes, path}`, `objects{bbox: fixed_size_list<float32,4>, category: list<int64>}`)
and compatible label structure. Row groups are small (16–100 rows), enabling
memory-efficient streaming. **Fully compatible — no remediation needed.**

## 5. Class Distribution (object level)

| Split | class 0 — accident | class 1 — non_accident | Ratio | Verdict |
|---|---|---|---|---|
| train | 2,313 (56.14%) | 1,807 (43.86%) | 1.28:1 | BALANCED |
| validation | 338 (49.85%) | 340 (50.15%) | 1:1 | BALANCED |
| test | 391 (64.95%) | 211 (35.05%) | 1.85:1 | MILDLY IMBALANCED |

Notes: no class is rare, but train (56%) and test (65%) accident rates differ by
~9 points — worth remembering when comparing split metrics in a later phase.
The label distribution here is at the *object* level; with ≥1 object per image,
image-level "contains an accident object" rates follow the same ordering.

## 6. Duplicate / Split-Contamination Check

SHA-256 of every embedded PNG across all splits (2,763 images hashed):

| Check | Result |
|---|---|
| Unique images overall | 2,737 (of 2,763) |
| train ∩ validation duplicates | **0** |
| train ∩ test duplicates | **0** |
| validation ∩ test duplicates | **0** |
| train internal duplicates | 26 (exact-byte PNG copies) |
| validation / test internal duplicates | 0 |

**No cross-split contamination.** The 26 train-internal duplicates are exact
byte copies (possibly augmentation artifacts, e.g. a flip/brightness variant
that produced identical bytes). 1.2% of train — benign, but Phase 2 may dedupe
or at least be aware of them for evaluation hygiene.

## 7. Leakage Audit

| Item | Reason for concern | Severity | Recommended action |
|---|---|---|---|
| Augmentation in train only | Horizontal-flip/brightness augmentation is applied to train but the class balance differences across splits (56% vs 65% accident) suggest splits were not stratified identically. Not leakage, but an evaluation caution. | LOW | Compare per-class metrics per split in a later evaluation phase. |
| Exact duplicate train images | 26 duplicated PNGs within train only; none cross split. | LOW | Dedupe or ignore in Phase 2. |
| Letterboxed source images | All frames pre-resized to 640×640 with padding — scale information is partially baked in. | LOW | Note for any future model-input decisions. |
| No IDs/timestamps | Nothing that could leak labels via identifiers. | — | No action. |

## 8. Data Quality Verdict

**GOOD** — identical schemas across splits, zero missing image bytes, zero
empty annotations, no cross-split duplicates, balanced classes, clean COCO-style
annotations. Minor notes: 26 train-internal duplicate images, test split mildly
more accident-heavy than train, class map not embedded in the files.