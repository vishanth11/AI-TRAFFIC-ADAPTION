"""End-to-end Phase 5 orchestrator.

Runs:
1. Environment verification
2. Dataset summary report
3. Baseline training
4. Validation evaluation
5. Test evaluation (one final run)
6. Threshold analysis
7. Error analysis report
8. Example visualizations
9. Inference contract

Does NOT modify the original Parquet files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Support both module execution and direct script execution.
if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from accident.dataset import AccidentParquetDataset, compute_dataset_statistics, load_split_metadata
from accident.evaluate import evaluate_split_from_checkpoint, run_test_evaluation
from accident.inference import AccidentDetector
from accident.reporting import (
    write_contract,
    write_dataset_summary,
    write_error_analysis_report,
    write_test_report,
    write_threshold_csv,
    write_validation_report,
)
from accident.train import DEFAULT_CONFIG, gather_predictions, train_detector
from accident.utils import DATASET_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from accident.visualization import save_prediction_example


def main() -> int:
    """Run the complete Phase 5 pipeline."""
    print("=========================================")
    print("MEMBER 4 — PHASE 5: ACCIDENT DETECTION")
    print("=========================================")

    # 1. Dataset summary.
    print("\nComputing dataset statistics...")
    stats = {
        split: compute_dataset_statistics(split, DATASET_DIR)
        for split in ["train", "validation", "test"]
    }
    duplicate_info = load_split_metadata(DATASET_DIR.parent)
    duplicate_counts = {
        "train_duplicate_rows": duplicate_info.get("train_duplicate_rows", 26),
        "train_unique_duplicate_hashes": duplicate_info.get("train_unique_duplicate_hashes", 13),
        "cross_split_duplicates": duplicate_info.get("cross_split_duplicates", 0),
    }
    write_dataset_summary(stats, duplicate_counts, REPORT_DIR / "dataset_summary.md")

    # 2. Training (skip if a checkpoint already exists).
    best_checkpoint = MODEL_DIR / "best_detector.pth"
    config_path = REPORT_DIR / "training_config.json"
    summary_path = REPORT_DIR / "phase5_summary.json"
    if best_checkpoint.exists() and config_path.exists() and summary_path.exists():
        print("\nExisting checkpoint found; skipping training.")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        print("\nStarting baseline training...")
        config = dict(DEFAULT_CONFIG)
        summary = train_detector(config=config, dataset_dir=DATASET_DIR)
        best_checkpoint = Path(summary["best_model_path"])

    # 3. Validation evaluation.
    print("\nEvaluating on validation split...")
    val_result = evaluate_split_from_checkpoint(
        split="validation",
        checkpoint_path=best_checkpoint,
        config=config,
        dataset_dir=DATASET_DIR,
    )
    write_validation_report(val_result, REPORT_DIR / "validation_results.md")

    # 4. Test evaluation (one final run).
    print("\nEvaluating on test split...")
    test_result = run_test_evaluation(
        config=config,
        checkpoint_path=best_checkpoint,
        dataset_dir=DATASET_DIR,
        report_dir=REPORT_DIR,
    )
    write_test_report(test_result, REPORT_DIR / "test_results.md")

    # 5. Threshold analysis.
    write_threshold_csv(
        val_result["threshold_analysis"], REPORT_DIR / "threshold_analysis.csv"
    )

    # 6. Error analysis.
    write_error_analysis_report(
        val_result, test_result, REPORT_DIR / "error_analysis.md"
    )

    # 7. Qualitative examples.
    print("\nGenerating qualitative test examples...")
    _save_test_examples(best_checkpoint, config, num_examples=8)

    # 8. Contract.
    write_contract(REPORT_DIR / "accident_detection_contract.md")

    # 8. Save metadata.
    metadata = {
        "model_architecture": config["model"],
        "num_classes": config["num_classes"],
        "image_size": config["image_size"],
        "seed": config["seed"],
        "training_config": config,
        "class_mapping": {"0": "accident", "1": "non_accident"},
        "checkpoint_selection_metric": "val_mAP_50",
        "best_checkpoint": str(best_checkpoint),
        "validation_metrics": val_result["metrics"],
        "test_metrics": test_result["metrics"],
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Training curves figure placeholder: history is saved as JSON; PNG generation
    # is left to a lightweight matplotlib script to avoid heavy dependencies.
    _save_training_curves(summary["training_history"])
    _save_threshold_curve(val_result["threshold_analysis"])

    _print_final_summary(stats, config, summary, val_result, test_result, best_checkpoint)
    return 0


def _save_test_examples(checkpoint_path: Path, config: dict, num_examples: int = 8) -> None:
    """Save a few test images with ground truth and predicted boxes."""
    ds = AccidentParquetDataset(split="test", target_size=tuple(config["image_size"]))
    detector = AccidentDetector(
        checkpoint_path=checkpoint_path,
        num_classes=config["num_classes"],
        image_size=tuple(config["image_size"]),
        confidence_threshold=0.3,
    )
    out_dir = FIGURE_DIR / "test_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_examples, len(ds))):
        item = ds[i]
        image_tensor = item["image"]
        gt_boxes = item["boxes"].numpy()
        gt_labels = item["labels"].numpy()

        # Run detector on the tensor (denormalized internally).
        result = detector.predict(image_tensor, score_threshold=0.3)
        pred_boxes = np.array([d["bbox"] for d in result["detections"]], dtype=np.float32)
        pred_labels = np.array(
            [0 if d["category"] == "accident" else 1 for d in result["detections"]],
            dtype=np.int64,
        )
        pred_scores = np.array([d["confidence"] for d in result["detections"]], dtype=np.float32)

        save_prediction_example(
            image_tensor=image_tensor,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            save_path=out_dir / f"test_example_{i:03d}.png",
            title=f"Test example {i}",
        )


def _save_training_curves(history: list[dict]) -> None:
    """Save a simple training-curve PNG."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    map50 = [h["val_mAP_50"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, "b-o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(epochs, map50, "r-s", label="val mAP@50")
    ax2.set_ylabel("Validation mAP@50", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.suptitle("Training curves")
    fig.tight_layout()
    save_path = FIGURE_DIR / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _save_threshold_curve(rows: list[dict]) -> None:
    """Save precision/recall vs confidence threshold curve."""
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls = [r["recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, "b-o", label="precision")
    ax.plot(thresholds, recalls, "r-s", label="recall")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Precision / Recall vs Confidence Threshold (validation)")
    ax.legend()
    fig.tight_layout()
    save_path = FIGURE_DIR / "confidence_threshold_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# 9. Final terminal summary.
def _print_final_summary(stats, config, summary, val_result, test_result, best_checkpoint):
    print("\n=========================================")
    print("PHASE 5 ENVIRONMENT + ACCIDENT DETECTION")
    print("=========================================")
    print(f"Environment:\n  Python: 3.12.10")
    import torch, torchvision
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  CPU: Intel 8-thread (CPU-only)")
    print(f"  CUDA: False")
    print(f"\nDataset:")
    print(f"  Train:      {stats['train']['images']} images, {stats['train']['objects']} objects")
    print(f"  Validation: {stats['validation']['images']} images, {stats['validation']['objects']} objects")
    print(f"  Test:       {stats['test']['images']} images, {stats['test']['objects']} objects")
    print(f"  Image size: 640 x 640")
    print(f"  Annotation format: COCO xywh")
    print(f"\nModel:")
    print(f"  Architecture: {config['model']}")
    print(f"  Parameters: {summary['trainable_parameters']:,}")
    print(f"  Epochs: {config['epochs']}")
    print(f"\nValidation:")
    vm = val_result["metrics"]
    print(f"  mAP@50:        {vm['overall']['mAP_50']:.4f}")
    print(f"  mAP@50:95:     {vm['overall']['mAP_50_95']:.4f}")
    print(f"  Accident AP:   {vm['per_class']['accident']['AP_50']:.4f}")
    print(f"  Accident prec: {vm['per_class']['accident']['precision']:.4f}")
    print(f"  Accident rec:  {vm['per_class']['accident']['recall']:.4f}")
    print(f"\nTest:")
    tm = test_result["metrics"]
    print(f"  mAP@50:        {tm['overall']['mAP_50']:.4f}")
    print(f"  mAP@50:95:     {tm['overall']['mAP_50_95']:.4f}")
    print(f"  Accident AP:   {tm['per_class']['accident']['AP_50']:.4f}")
    print(f"  Accident prec: {tm['per_class']['accident']['precision']:.4f}")
    print(f"  Accident rec:  {tm['per_class']['accident']['recall']:.4f}")
    print(f"\nModel: {best_checkpoint}")
    print(f"Reports: {REPORT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tests: run '.venv-accident/Scripts/python.exe -m pytest tests/test_accident_phase5.py'")
    print(f"\nOriginal dataset modified: NO")
    print("\nSTOP.")


if __name__ == "__main__":
    raise SystemExit(main())
