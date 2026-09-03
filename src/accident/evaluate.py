"""Evaluation helpers for validation and test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import AccidentParquetDataset, collate_detection
from .metrics import evaluate_coco, threshold_analysis
from .model import build_detector
from .train import gather_predictions
from .utils import FIGURE_DIR, MODEL_DIR, REPORT_DIR, get_device


def evaluate_split_from_checkpoint(
    split: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    dataset_dir: Path | None = None,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a split using a saved checkpoint.

    Returns ground-truth records, prediction records, COCO metrics, and
    threshold analysis.
    """
    device = get_device()
    image_size = tuple(config["image_size"])

    ds = AccidentParquetDataset(
        split=split, target_size=image_size, dataset_dir=dataset_dir
    )
    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_detection,
    )

    model = build_detector(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)

    gt_records, pred_records = gather_predictions(model, loader, device, score_threshold)
    metrics = evaluate_coco(gt_records, pred_records, image_size)
    thresholds = threshold_analysis(gt_records, pred_records, image_size=image_size)

    return {
        "split": split,
        "metrics": metrics,
        "threshold_analysis": thresholds,
        "num_images": len(ds),
        "num_ground_truth_objects": sum(len(rec["boxes"]) for rec in gt_records),
        "num_predictions": sum(len(rec["boxes"]) for rec in pred_records),
    }


def run_test_evaluation(
    config: dict[str, Any],
    checkpoint_path: Path = MODEL_DIR / "best_detector.pth",
    dataset_dir: Path | None = None,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    """Run the ONE final test evaluation after validation-based selection."""
    print("Running final test evaluation with best validation checkpoint...")
    result = evaluate_split_from_checkpoint(
        split="test",
        checkpoint_path=checkpoint_path,
        config=config,
        dataset_dir=dataset_dir,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    test_path = report_dir / "test_results.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"Test mAP@50={result['metrics']['overall']['mAP_50']:.4f} — "
        f"accident AP={result['metrics']['per_class']['accident']['AP_50']:.4f} — "
        f"accident recall={result['metrics']['per_class']['accident']['recall']:.4f}"
    )
    return result
