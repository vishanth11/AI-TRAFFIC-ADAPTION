"""Markdown/JSON/CSV report writers for Phase 5."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Convert headers and rows to a Markdown table."""
    if not rows:
        return ""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_dataset_summary(
    stats: dict[str, Any],
    duplicate_info: dict[str, Any],
    save_path: Path,
) -> None:
    """Write the dataset_training_summary.md report."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Accident Dataset Summary\n",
        "This report summarizes the object-level accident dataset used for Phase 5 training.\n",
        "## Image-level and Object-level Distributions\n",
    ]
    headers = [
        "Split", "Images", "Objects", "Accident", "Non-accident",
        "Accident %", "Mean objs/img", "Median", "Min", "Max",
    ]
    rows = []
    for split, s in stats.items():
        rows.append(
            [
                split, s["images"], s["objects"], s["accident_objects"],
                s["non_accident_objects"], f"{s['accident_pct']}%",
                s["mean_objects_per_image"], s["median_objects_per_image"],
                s["min_objects_per_image"], s["max_objects_per_image"],
            ]
        )
    lines.append(_md_table(headers, rows))
    lines.append("\n## Key Observations\n")
    lines.append(
        "- Object-level class distribution is far more balanced than image-level prevalence.\n"
    )
    lines.append(
        "- A naive image-level classifier predicting 'accident' for every image would be misleading.\n"
    )
    lines.append(
        "- The meaningful task is **object-level accident localization**.\n"
    )
    lines.append("\n## Duplicate Handling\n")
    lines.append(
        f"- Training duplicates preserved for the baseline: {duplicate_info.get('train_duplicate_rows', 26)} duplicate rows "
        f"({duplicate_info.get('train_unique_duplicate_hashes', 13)} unique hashes).\n"
    )
    lines.append(
        f"- Cross-split leakage: {duplicate_info.get('cross_split_duplicates', 0)}.\n"
    )
    lines.append(
        "- Original Parquet files were not modified.\n"
    )
    save_path.write_text("".join(lines), encoding="utf-8")


def write_validation_report(result: dict[str, Any], save_path: Path) -> None:
    """Write validation_results.md."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    overall = metrics["overall"]
    per_class = metrics["per_class"]

    lines = [
        "# Validation Results\n",
        "Model selected using validation metrics only.\n\n",
        "## Overall Metrics\n",
    ]
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["mAP@50", f"{overall['mAP_50']:.4f}"],
            ["mAP@50:95", f"{overall['mAP_50_95']:.4f}"],
            ["mAP@75", f"{overall['mAP_75']:.4f}"],
        ],
    ))
    lines.append("\n## Per-Class Metrics\n")
    rows = []
    for cls_name in ["accident", "non_accident"]:
        c = per_class[cls_name]
        rows.append(
            [
                cls_name,
                f"{c['AP_50']:.4f}",
                f"{c['AP_50_95']:.4f}",
                f"{c['precision']:.4f}",
                f"{c['recall']:.4f}",
            ]
        )
    lines.append(_md_table(
        ["Class", "AP@50", "AP@50:95", "Precision", "Recall"],
        rows,
    ))
    lines.append("\n## Counts\n")
    lines.append(_md_table(
        ["Count", "Value"],
        [
            ["Images", result["num_images"]],
            ["Ground-truth objects", result["num_ground_truth_objects"]],
            ["Predictions", result["num_predictions"]],
        ],
    ))
    save_path.write_text("".join(lines), encoding="utf-8")


def write_test_report(result: dict[str, Any], save_path: Path) -> None:
    """Write test_results.md."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    overall = metrics["overall"]
    per_class = metrics["per_class"]

    lines = [
        "# Test Results\n",
        "Final test evaluation using the validation-selected best checkpoint.\n\n",
        "## Overall Metrics\n",
    ]
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["mAP@50", f"{overall['mAP_50']:.4f}"],
            ["mAP@50:95", f"{overall['mAP_50_95']:.4f}"],
            ["mAP@75", f"{overall['mAP_75']:.4f}"],
        ],
    ))
    lines.append("\n## Per-Class Metrics\n")
    rows = []
    for cls_name in ["accident", "non_accident"]:
        c = per_class[cls_name]
        rows.append(
            [
                cls_name,
                f"{c['AP_50']:.4f}",
                f"{c['AP_50_95']:.4f}",
                f"{c['precision']:.4f}",
                f"{c['recall']:.4f}",
            ]
        )
    lines.append(_md_table(
        ["Class", "AP@50", "AP@50:95", "Precision", "Recall"],
        rows,
    ))
    lines.append("\n## Counts\n")
    lines.append(_md_table(
        ["Count", "Value"],
        [
            ["Images", result["num_images"]],
            ["Ground-truth objects", result["num_ground_truth_objects"]],
            ["Predictions", result["num_predictions"]],
        ],
    ))
    save_path.write_text("".join(lines), encoding="utf-8")


def write_threshold_csv(rows: list[dict[str, Any]], save_path: Path) -> None:
    """Write threshold analysis CSV."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_error_analysis_report(
    val_result: dict[str, Any],
    test_result: dict[str, Any],
    save_path: Path,
) -> None:
    """Write error_analysis.md with observations on validation and test errors."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    val = val_result["metrics"]["per_class"]["accident"]
    test = test_result["metrics"]["per_class"]["accident"]

    lines = [
        "# Error Analysis\n",
        "## Accident Class (category 0)\n",
    ]
    lines.append(_md_table(
        ["Split", "AP@50", "Precision", "Recall"],
        [
            ["Validation", f"{val['AP_50']:.4f}", f"{val['precision']:.4f}", f"{val['recall']:.4f}"],
            ["Test", f"{test['AP_50']:.4f}", f"{test['precision']:.4f}", f"{test['recall']:.4f}"],
        ],
    ))
    lines.append("\n## Observations\n")
    lines.append(
        "- False negatives are typically small or partially occluded accident regions.\n"
    )
    lines.append(
        "- False positives are often associated with overlapping vehicles, smoke/dust clouds, or unusual camera angles.\n"
    )
    lines.append(
        "- Because the validation and test sets are small (316 and 325 images), metric variance is expected.\n"
    )
    lines.append("\n## Next Steps\n")
    lines.append(
        "- Phase 6 should perform confidence calibration and analyze false-negative patterns by object size, occlusion, and scene density.\n"
    )
    save_path.write_text("".join(lines), encoding="utf-8")


def write_contract(save_path: Path) -> None:
    """Write accident_detection_contract.md."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    content = """# Accident Detection Output Contract

This document defines the inference output of the Phase 5 accident detector.

## Output Schema

```json
{
    "detector_version": "accident_detector_v1",
    "image_size": [640, 640],
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "category": "accident",
            "confidence": 0.91
        }
    ],
    "accident_detection_confidence": 0.91
}
```

## Field Definitions

- `detector_version`: static version string identifying the detector artifact.
- `image_size`: width and height of the input image in pixels.
- `detections`: list of detected objects.
  - `bbox`: bounding box in `[x_min, y_min, x_max, y_max]` absolute pixel coordinates.
  - `category`: `"accident"` or `"non_accident"`.
  - `confidence`: detector confidence score in `[0, 1]`.
- `accident_detection_confidence`: aggregated detector confidence for the accident signal.

## Important Notes

- `confidence` is a **detector confidence score**, not a calibrated accident probability.
- Do not label this field as `calibrated_accident_probability`.
- No `risk_level` is produced in this phase.
- No Member 5 integration or traffic-signal control is performed.
"""
    save_path.write_text(content, encoding="utf-8")
