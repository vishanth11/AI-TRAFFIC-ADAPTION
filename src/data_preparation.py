"""Phase 2 — Member 4 data preparation (deterministic, read-only on originals).

Produces lightweight processed artifacts for Phase 3+:

  data/processed/near_miss/
      dtc_fm_combined.csv     positional-merged features+labels with sample_index
      duplicate_report.csv    feature-row duplicate groups + label agreement
      label_mapping.json      neutral class_0/class_1 (semantics unconfirmed)
      feature_policy.json     keep/flag decision + reason per feature

  data/processed/accident/
      train_index.parquet     one row per image (split, source, sha256, dims,
      validation_index.parquet  annotations) — NO image bytes copied
      test_index.parquet
      duplicate_report.csv    train exact-byte duplicate groups w/ annotation diff
      split_integrity_report.json
      label_mapping.json
      corruption_report.json  (empty list when all sampled images decode)

Originals under dataset/ are never modified. No randomness is used anywhere.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = Path("dataset")
OUT_NM = Path("data/processed/near_miss")
OUT_ACC = Path("data/processed/accident")
REPORT_DIR = Path("reports")

FEATURES_CSV = DATASET_DIR / "filtered_features_DTC_FM.csv"
LABELS_CSV = DATASET_DIR / "target_labels_DTC_FM.csv"

ACCIDENT_SPLITS = {
    "train": [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}
EXPECTED_DTC_ROWS = 2508
DECODE_SAMPLE_STRIDE = 100  # decode every Nth image as a representative sample


# --------------------------------------------------------------------------
# Near-miss (DTC-FM)
# --------------------------------------------------------------------------

def build_near_miss() -> None:
    feats = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)

    # --- alignment verification (positional; no sorting, no pre-drops) ---
    assert len(feats) == EXPECTED_DTC_ROWS, f"features rows {len(feats)} != {EXPECTED_DTC_ROWS}"
    assert len(labels) == EXPECTED_DTC_ROWS, f"label rows {len(labels)} != {EXPECTED_DTC_ROWS}"
    assert list(labels.columns) == ["event_type"]
    assert labels["event_type"].isin([0.0, 1.0]).all(), "unexpected label values"

    combined = feats.copy()
    combined.insert(0, "sample_index", np.arange(len(combined), dtype=np.int64))
    combined["event_type"] = labels["event_type"].astype(np.int64).values

    # --- duplicate analysis (report only; nothing dropped) ---
    dup_mask = combined.duplicated(
        subset=[c for c in feats.columns], keep=False
    )
    groups = combined.loc[dup_mask, feats.columns.tolist()].apply(
        lambda r: "|".join(map(str, r)), axis=1
    )
    group_ids = pd.factorize(groups)[0]
    report = pd.DataFrame({
        "duplicate_group": group_ids,
        "sample_index": combined.loc[dup_mask, "sample_index"].values,
        "event_type": combined.loc[dup_mask, "event_type"].values,
    }).sort_values(["duplicate_group", "sample_index"])

    label_conflicts = (
        report.groupby("duplicate_group")["event_type"].nunique() > 1
    )
    report.attrs["n_groups"] = int(report["duplicate_group"].nunique())
    report.attrs["n_conflicting_groups"] = int(label_conflicts.sum())
    report.to_csv(OUT_NM / "duplicate_report.csv", index=False)

    # --- cleaned combined dataset (original order preserved) ---
    combined.to_csv(OUT_NM / "dtc_fm_combined.csv", index=False)
    # exact-float companion (CSV round-trip can shift the last ULP of ~23 values)
    combined.to_parquet(OUT_NM / "dtc_fm_combined.parquet", index=False)

    write_near_miss_label_mapping()
    write_feature_policy()
    return report


def write_near_miss_label_mapping() -> None:
    payload = {
        "dataset": "DTC-FM",
        "target_column": "event_type",
        "mapping_status": "unconfirmed",
        "class_0": "class_0 (semantics unconfirmed; evidence from class-conditional "
                   "statistics is consistent with a 'non-conflict / normal' class, "
                   "NOT confirmed by source documentation)",
        "class_1": "class_1 (semantics unconfirmed; evidence from class-conditional "
                   "statistics is consistent with a 'conflict / near-miss' class, "
                   "NOT confirmed by source documentation)",
        "source": "No label map present in the CSV files. Web documentation search "
                  "performed in Phase 3 (2026-09-02): no public dataset, paper or "
                  "documentation named 'DTC-FM' could be located; no source "
                  "documentation available locally. Neutral class_0/class_1 "
                  "terminology MUST be used until the mapping is confirmed.",
        "class_distribution": {
            "class_0": {"count": 1867, "pct": 74.44},
            "class_1": {"count": 641, "pct": 25.56},
        },
        "note": "sample_index in dtc_fm_combined.csv is an internal processing "
                "index (row position 0..2507), NOT a real dataset identifier.",
    }
    (OUT_NM / "label_mapping.json").write_text(json.dumps(payload, indent=2))


def write_feature_policy() -> None:
    policy = {
        "policy_version": "phase2",
        "decisions": [
            {
                "feature": "speed_object_1_kph", "type": "original",
                "decision": "keep", "reason": "base physical measurement (kph)",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "speed_object_2_kph", "type": "original",
                "decision": "keep", "reason": "base physical measurement (kph)",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "angle_degrees", "type": "original",
                "decision": "keep", "reason": "encounter geometry, computable pre-conflict",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "PET", "type": "original",
                "decision": "keep_with_flag",
                "reason": "core surrogate-safety measure; retained for offline "
                          "modeling but flagged: post-encroachment time can only be "
                          "measured AFTER the interaction completes",
                "leakage_risk": "none_statistical",
                "deployment_availability": "POST_EVENT_ONLY — not available in a "
                                           "real-time system at decision time; "
                                           "deployment/design issue, not target leakage",
            },
            {
                "feature": "acceleration_obj1_mps2", "type": "original",
                "decision": "keep", "reason": "base kinematic measurement",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "acceleration_obj2_mps2", "type": "original",
                "decision": "keep", "reason": "base kinematic measurement",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "min_dist_dual_check_m", "type": "original",
                "decision": "keep", "reason": "minimum separation distance (m)",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "object_count_at_frame1", "type": "original",
                "decision": "keep", "reason": "scene context",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "target_dist_px", "type": "original",
                "decision": "keep_with_flag",
                "reason": "raw pixel distance; camera-geometry dependent",
                "leakage_risk": "none",
                "deployment_availability": "camera_dependent — may not transfer "
                                           "across camera geometries",
            },
            {
                "feature": "class_object_1", "type": "original_categorical",
                "decision": "keep", "reason": "object type; requires encoding in Phase 3",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "class_object_2", "type": "original_categorical",
                "decision": "keep", "reason": "object type; requires encoding in Phase 3",
                "leakage_risk": "none", "deployment_availability": "available_live",
            },
            {
                "feature": "FE_inv_PET", "type": "engineered_verified",
                "formula": "1 / PET",
                "decision": "retain_temporarily_flag_redundant",
                "reason": "pure mathematical transform, verified; exact duplicate of "
                          "PET information (r=0.92) — statistical redundancy, NOT leakage",
                "leakage_risk": "none_statistical_redundancy_only",
                "deployment_availability": "inherits PET post-event limitation",
            },
            {
                "feature": "FE_dist_squared", "type": "engineered_verified",
                "formula": "target_dist_px ** 2",
                "decision": "retain_temporarily_flag_redundant",
                "reason": "verified pure transform; heavy-tailed scale instability; "
                          "statistical redundancy, NOT leakage",
                "leakage_risk": "none_statistical_redundancy_only",
                "deployment_availability": "camera_dependent",
            },
            {
                "feature": "FE_log_mesafe", "type": "engineered_verified",
                "formula": "ln(1 + target_dist_px)",
                "decision": "retain_temporarily_flag_redundant",
                "reason": "verified pure transform; monotone copy of target_dist_px; "
                          "statistical redundancy, NOT leakage",
                "leakage_risk": "none_statistical_redundancy_only",
                "deployment_availability": "camera_dependent",
            },
            {
                "feature": "FE_safety_index", "type": "engineered_verified_phase3",
                "formula": "(speed_object_1_kph + speed_object_2_kph) / "
                           "(2 * max(target_dist_px, 1e-06))",
                "decision": "keep_with_flag",
                "reason": "PROVENANCE ESTABLISHED in Phase 3: exact deterministic "
                          "transform of speed_object_1_kph, speed_object_2_kph and "
                          "target_dist_px (verified to 3 decimals on all 2,457 "
                          "dist>0 rows; the 51 dist==0 rows use a 1e-6 floor, "
                          "producing quantized extremes up to 2.85e7). It is an "
                          "inverse-pixel-distance x combined-speed composite with "
                          "inconsistent units (kph/px) — redundant with existing "
                          "features and numerically unstable, but NOT target leakage",
                "leakage_risk": "none (pure input-feature transform); "
                                "redundancy + instability only",
                "deployment_availability": "available_live (speeds + pixel distance), "
                                           "camera_dependent",
                "action": "Phase 3 trains with/without it (configs A/B) to measure "
                          "its effect; no test-based keep/drop decision",
            },
        ],
        "principles": [
            "derived != leakage: transforms are flagged for redundancy, not accused of leakage",
            "PET concern is deployment availability (post-event), not statistical leakage",
            "no feature deleted; decisions are advisory for Phase 3",
        ],
    }
    (OUT_NM / "feature_policy.json").write_text(json.dumps(policy, indent=2))


# --------------------------------------------------------------------------
# Accident (Parquet)
# --------------------------------------------------------------------------

def png_ihdr(data: bytes) -> tuple[str, int, int] | None:
    """Return (format, width, height) from a PNG header, or None if invalid."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return "PNG", w, h


def scan_accident_shards() -> dict[str, pd.DataFrame]:
    """Stream every shard once: hash, dims, annotations -> per-split index frames."""
    indexes: dict[str, list[dict]] = {s: [] for s in ACCIDENT_SPLITS}
    for split, files in ACCIDENT_SPLITS.items():
        for fname in files:
            pf = pq.ParquetFile(DATASET_DIR / fname)
            local_row = 0
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg)
                images = tbl.column("image").to_pylist()
                objects = tbl.column("objects").to_pylist()
                for row_in_rg, (img, obj) in enumerate(zip(images, objects)):
                    data = (img or {}).get("bytes") or b""
                    fmt, w, h = png_ihdr(data) if data else ("EMPTY", 0, 0)
                    cats = list((obj or {}).get("category") or [])
                    boxes = [list(map(float, b)) for b in (obj or {}).get("bbox") or []]
                    indexes[split].append({
                        "split": split,
                        "source_file": fname,
                        "row_index": local_row,
                        # row_index is 0-based WITHIN source_file (internal
                        # reference = source_file + row_index; not a dataset ID)
                        "image_sha256": hashlib.sha256(data).hexdigest() if data else "",
                        "image_format": fmt,
                        "width": w,
                        "height": h,
                        "n_objects": len(cats),
                        "categories": cats,
                        "bbox": boxes,
                    })
                    local_row += 1
    out = {}
    for split, rows in indexes.items():
        schema = pa.schema([
            ("split", pa.string()), ("source_file", pa.string()),
            ("row_index", pa.int64()), ("image_sha256", pa.string()),
            ("image_format", pa.string()), ("width", pa.int32()),
            ("height", pa.int32()), ("n_objects", pa.int32()),
            ("categories", pa.list_(pa.int64())),
            ("bbox", pa.list_(pa.list_(pa.float32(), 4))),
        ])
        tbl = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(tbl, OUT_ACC / f"{split}_index.parquet")
        out[split] = tbl.to_pandas()
    return out


def accident_label_mapping() -> None:
    payload = {
        "dataset": "traffic-accident-cctv-object-detection",
        "task": "accident + object detection",
        "label_column": "objects.category",
        "mapping": {"0": "accident", "1": "non_accident"},
        "source": "dataset documentation (Hugging Face dataset card README, "
                  "justjuu/traffic-accident-cctv-object-detection); class map is "
                  "NOT embedded in the parquet files themselves",
        "bbox_format": "COCO-style [x, y, w, h] in absolute pixels on 640x640 images",
    }
    (OUT_ACC / "label_mapping.json").write_text(json.dumps(payload, indent=2))


def accident_duplicate_report(idx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Group train images by exact-byte hash; compare labels+annotations."""
    rows = []
    train = idx["train"]
    for h, grp in train.groupby("image_sha256"):
        if len(grp) < 2:
            continue
        ref_cats = list(grp["categories"].iloc[0])
        ref_bbox = [list(b) for b in grp["bbox"].iloc[0]]
        cats_identical = all(list(c) == ref_cats for c in grp["categories"])
        bbox_identical = all(
            [list(b) for b in boxes] == ref_bbox for boxes in grp["bbox"]
        )
        for _, r in grp.iterrows():
            rows.append({
                "image_sha256": h,
                "source_file": r["source_file"],
                "row_index": r["row_index"],
                "categories": json.dumps([int(c) for c in r["categories"]]),
                "annotations_identical": cats_identical and bbox_identical,
                "label_conflict": not cats_identical,
            })
    rep = pd.DataFrame(rows).sort_values(["image_sha256", "row_index"])
    rep.to_csv(OUT_ACC / "duplicate_report.csv", index=False)
    return rep


def split_integrity_report(idx: dict[str, pd.DataFrame]) -> dict:
    sets = {s: set(idx[s]["image_sha256"]) for s in idx}
    tv = sets["train"] & sets["validation"]
    tt = sets["train"] & sets["test"]
    vt = sets["validation"] & sets["test"]
    n_cross = len(tv) + len(tt) + len(vt)
    payload = {
        "method": "SHA-256 of embedded PNG bytes (content hash, not filename)",
        "train_images": len(idx["train"]),
        "validation_images": len(idx["validation"]),
        "test_images": len(idx["test"]),
        "unique_images": len(sets["train"] | sets["validation"] | sets["test"]),
        "cross_split_duplicates": n_cross,
        "detail": {
            "train_and_validation": len(tv),
            "train_and_test": len(tt),
            "validation_and_test": len(vt),
        },
        "result": "PASS" if n_cross == 0 else "FAIL",
    }
    (OUT_ACC / "split_integrity_report.json").write_text(json.dumps(payload, indent=2))
    return payload


def fetch_image_bytes(source_file: str, row_index: int) -> bytes:
    """Resolve a split-global row_index to its row group and fetch image bytes."""
    pf = pq.ParquetFile(DATASET_DIR / source_file)
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if offset + n > row_index:
            tbl = pf.read_row_group(rg, columns=["image"])
            return tbl.column("image")[row_index - offset].as_py()["bytes"]
        offset += n
    raise IndexError(f"row_index {row_index} beyond {source_file}")


def validate_sample_images(idx: dict[str, pd.DataFrame]) -> list[dict]:
    """Decode a representative sample; verify dims + bounds on the full index."""
    corrupt: list[dict] = []
    try:
        from PIL import Image
        have_pil = True
    except ImportError:
        have_pil = False
        print("WARNING: Pillow unavailable — decode check skipped (IHDR checks only)")

    for split, df in idx.items():
        bad_dims = df[(df["width"] != 640) | (df["height"] != 640)]
        for _, r in bad_dims.iterrows():
            corrupt.append({"source_file": r["source_file"], "row_index": int(r["row_index"]),
                            "issue": f"unexpected dims {r['width']}x{r['height']}"})
        empty = df[df["image_format"] == "EMPTY"]
        for _, r in empty.iterrows():
            corrupt.append({"source_file": r["source_file"], "row_index": int(r["row_index"]),
                            "issue": "empty image bytes"})
        for _, r in df.iloc[::DECODE_SAMPLE_STRIDE].iterrows():
            data = fetch_image_bytes(r["source_file"], int(r["row_index"]))
            if have_pil:
                try:
                    im = Image.open(__import__("io").BytesIO(data))
                    im.load()
                    if im.size != (r["width"], r["height"]):
                        corrupt.append({"source_file": r["source_file"],
                                        "row_index": int(r["row_index"]),
                                        "issue": "decoded dims != IHDR dims"})
                except Exception as exc:  # noqa: BLE001 - report any decode failure
                    corrupt.append({"source_file": r["source_file"],
                                    "row_index": int(r["row_index"]),
                                    "issue": f"decode failed: {exc}"})

    # bbox bounds check on the full index (annotation references valid image)
    for split, df in idx.items():
        for i, boxes in enumerate(df["bbox"]):
            for b in boxes:
                x, y, w, h = b
                if x < 0 or y < 0 or x + w > 640.5 or y + h > 640.5:
                    corrupt.append({"source_file": df["source_file"].iloc[i],
                                    "row_index": int(df["row_index"].iloc[i]),
                                    "issue": f"bbox out of bounds: {b}"})

    (OUT_ACC / "corruption_report.json").write_text(
        json.dumps({"sampled_stride": DECODE_SAMPLE_STRIDE,
                    "issues_found": len(corrupt), "corrupt_records": corrupt}, indent=2))
    return corrupt


def class_balance(idx: dict[str, pd.DataFrame]) -> dict:
    labels = pd.read_csv(LABELS_CSV)["event_type"].value_counts().sort_index()
    payload = {
        "near_miss_DTC_FM": {
            "class_0": {"count": int(labels[0.0]), "pct": round(labels[0.0] / 2508 * 100, 2)},
            "class_1": {"count": int(labels[1.0]), "pct": round(labels[1.0] / 2508 * 100, 2)},
            "verdict": "MODERATELY_IMBALANCED",
        },
        "accident_object_level": {},
        "accident_image_level": {},
    }
    for split, df in idx.items():
        all_cats = [c for cats in df["categories"] for c in cats]
        n0 = sum(1 for c in all_cats if c == 0)
        n1 = sum(1 for c in all_cats if c == 1)
        n_img_acc = sum(1 for cats in df["categories"] if 0 in cats)
        payload["accident_object_level"][split] = {
            "accident": n0, "non_accident": n1, "total": n0 + n1,
            "accident_pct": round(n0 / max(n0 + n1, 1) * 100, 2),
        }
        payload["accident_image_level"][split] = {
            "images": len(df),
            "images_with_accident_object": n_img_acc,
            "pct": round(n_img_acc / len(df) * 100, 2),
        }
    return payload


def main() -> None:
    OUT_NM.mkdir(parents=True, exist_ok=True)
    OUT_ACC.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    dup_rep = build_near_miss()
    print(f"near-miss: combined dataset written; duplicate groups="
          f"{dup_rep.attrs['n_groups']}, conflicting-label groups="
          f"{dup_rep.attrs['n_conflicting_groups']}")

    idx = scan_accident_shards()
    accident_label_mapping()
    acc_dup = accident_duplicate_report(idx)
    integrity = split_integrity_report(idx)
    corrupt = validate_sample_images(idx)
    balance = class_balance(idx)

    print(f"accident: index rows per split = "
          f"{ {s: len(d) for s, d in idx.items()} }")
    print(f"accident: train duplicate images = "
          f"{acc_dup['image_sha256'].nunique()}, conflicting annotations = "
          f"{int((~acc_dup['annotations_identical']).groupby(acc_dup['image_sha256']).any().sum())}")
    print(f"accident: split integrity = {integrity['result']}")
    print(f"accident: corrupt/invalid records found = {len(corrupt)}")

    (REPORT_DIR / "class_balance_phase2.json").write_text(json.dumps(balance, indent=2))
    print("class balance written to reports/class_balance_phase2.json")

    # ---------------- post-preparation validation ----------------
    combined = pd.read_csv(OUT_NM / "dtc_fm_combined.csv")
    combined_exact = pd.read_parquet(OUT_NM / "dtc_fm_combined.parquet")
    orig_f = pd.read_csv(FEATURES_CSV)
    orig_l = pd.read_csv(LABELS_CSV)["event_type"].astype(np.int64).values
    num_cols = orig_f.select_dtypes(include=[np.number]).columns

    def csv_roundtrip_ok() -> bool:
        if not combined[orig_f.columns].dtypes.equals(orig_f.dtypes):
            if not combined[orig_f.select_dtypes(exclude=[np.number]).columns].equals(
                    orig_f[orig_f.select_dtypes(exclude=[np.number]).columns]):
                return False
        return bool(np.allclose(
            combined[num_cols].to_numpy(dtype=float),
            orig_f[num_cols].to_numpy(dtype=float),
            rtol=0, atol=1e-9, equal_nan=True))

    checks = {
        "near_miss_rows": len(combined) == EXPECTED_DTC_ROWS,
        "near_miss_features_exact_in_parquet":
            combined_exact[orig_f.columns].equals(orig_f),
        "near_miss_feature_values_unchanged_csv_tol1e-9": csv_roundtrip_ok(),
        "near_miss_labels_unchanged": (combined["event_type"].values == orig_l).all(),
        "near_miss_sample_index_ok":
            (combined["sample_index"].values == np.arange(EXPECTED_DTC_ROWS)).all(),
        "near_miss_no_new_missing": int(combined.isna().sum().sum()) == 0,
    }
    for split, df in idx.items():
        pq_rows = pq.ParquetFile(OUT_ACC / f"{split}_index.parquet").metadata.num_rows
        checks[f"{split}_index_count_matches_scan"] = pq_rows == len(df)
    checks["accident_total_images"] = sum(len(d) for d in idx.values()) == 2763
    checks["accident_no_cross_split_contamination"] = integrity["result"] == "PASS"
    print("\nvalidation:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    if not all(checks.values()):
        raise SystemExit("VALIDATION FAILED — inspect outputs before proceeding")


if __name__ == "__main__":
    main()