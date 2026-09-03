"""Phase 5B post-training analysis: errors, qualitative results, test eval.

Tools for the selected Phase 5B checkpoint:
    - false-negative / false-positive visual + tabular analysis (validation)
    - qualitative prediction panels (GT + predictions + confidence)
    - the single locked TEST evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from accident import CLASS_NAMES
from accident.dataset import AccidentParquetDataset
from accident.metrics import _compute_iou, threshold_analysis
from accident.model import build_detector
from accident.phase5b import THRESHOLD_SWEEP, evaluate_metrics
from accident.utils import FIGURE_DIR, MODEL_DIR, REPORT_DIR, get_device, seed_everything
from accident.visualization import save_prediction_example


def match_boxes(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy IoU matching per class.

    Returns (matches, unmatched_gt_indices, unmatched_pred_indices) where
    matches are (gt_idx, pred_idx, iou) triples.
    """
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    order = np.argsort(-pred_scores) if len(pred_scores) else []
    for p in order:
        best_iou, best_g = 0.0, -1
        for g in range(len(gt_boxes)):
            if g in used_gt or int(gt_labels[g]) != int(pred_labels[p]):
                continue
            iou = _compute_iou(pred_boxes[p], gt_boxes[g])
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_g >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_g)
            used_pred.add(int(p))
            matches.append((best_g, int(p), float(best_iou)))
    unmatched_gt = [g for g in range(len(gt_boxes)) if g not in used_gt]
    unmatched_pred = [int(p) for p in range(len(pred_boxes)) if p not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def box_size_bucket(box: np.ndarray) -> str:
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    area = w * h
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def collect_error_examples(
    model: torch.nn.Module,
    split: str = "validation",
    image_size: tuple[int, int] = (320, 320),
    score_threshold: float = 0.05,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    """Gather FN/FP statistics and example indices for one split."""
    from accident.utils import DATASET_DIR

    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    device = get_device()
    ds = AccidentParquetDataset(split=split, target_size=image_size, dataset_dir=dataset_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, collate_fn=collate_fn_ref()
    )
    model.eval()

    fn_rows: list[dict[str, Any]] = []
    fp_rows: list[str] = []
    fn_examples: list[int] = []
    fp_examples: list[int] = []

    for item_iter, (images, targets) in enumerate(loader):
        outputs = model([img.to(device) for img in images])
        for t, o in zip(targets, outputs):
            image_id = int(t["image_id"].item())
            gt_boxes = t["boxes"].numpy()
            gt_labels = t["labels"].numpy()
            keep = o["scores"] >= score_threshold
            pred_boxes = o["boxes"][keep].cpu().numpy()
            # Model label space -> dataset space.
            pred_labels = (o["labels"][keep].cpu().numpy() - 1)
            pred_scores = o["scores"][keep].cpu().numpy()

            matches, unmatched_gt, unmatched_pred = match_boxes(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
            )
            for g in unmatched_gt:
                if int(gt_labels[g]) != 0:
                    continue  # FN analysis focuses on accident objects
                best_iou = 0.0
                best_score = 0.0
                for p in range(len(pred_boxes)):
                    if int(pred_labels[p]) != 0:
                        continue
                    iou = _compute_iou(pred_boxes[p], gt_boxes[g])
                    if iou > best_iou:
                        best_iou = iou
                        best_score = float(pred_scores[p])
                fn_rows.append(
                    {
                        "image_id": image_id,
                        "size_bucket": box_size_bucket(gt_boxes[g]),
                        "width": float(gt_boxes[g][2] - gt_boxes[g][0]),
                        "height": float(gt_boxes[g][3] - gt_boxes[g][1]),
                        "center_x": float((gt_boxes[g][0] + gt_boxes[g][2]) / 2),
                        "center_y": float((gt_boxes[g][1] + gt_boxes[g][3]) / 2),
                        "best_pred_iou": round(best_iou, 4),
                        "best_accident_score": round(best_score, 4),
                    }
                )
                if len(fn_examples) < 12:
                    fn_examples.append(image_id)
            accident_fps = [
                p for p in unmatched_pred if int(pred_labels[p]) == 0
            ]
            for p in accident_fps:
                fp_rows.append(
                    {
                        "image_id": image_id,
                        "score": round(float(pred_scores[p]), 4),
                        "size_bucket": box_size_bucket(pred_boxes[p]),
                        "center_x": float((pred_boxes[p][0] + pred_boxes[p][2]) / 2),
                        "center_y": float((pred_boxes[p][1] + pred_boxes[p][3]) / 2),
                    }
                )
                if len(fp_examples) < 12:
                    fp_examples.append(image_id)

    return {
        "fn_rows": fn_rows,
        "fp_rows": fp_rows,
        "fn_example_ids": sorted(set(fn_examples)),
        "fp_example_ids": sorted(set(fp_examples)),
    }


def collate_fn_ref():
    from accident.dataset import collate_detection

    return collate_detection


def save_error_visuals(
    model: torch.nn.Module,
    split: str,
    image_ids: list[int],
    out_dir: Path,
    image_size: tuple[int, int] = (320, 320),
    score_threshold: float = 0.05,
    prefix: str = "fn",
    dataset_dir: Path | None = None,
) -> None:
    """Save GT+prediction panels for the given example image ids."""
    from accident.utils import DATASET_DIR

    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    device = get_device()
    ds = AccidentParquetDataset(split=split, target_size=image_size, dataset_dir=dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for image_id in image_ids:
            item = ds[image_id]
            output = model([item["image"].to(device)])[0]
            keep = output["scores"] >= score_threshold
            pred_boxes = output["boxes"][keep].cpu().numpy()
            pred_labels = output["labels"][keep].cpu().numpy() - 1
            pred_scores = output["scores"][keep].cpu().numpy()
            save_prediction_example(
                image_tensor=item["image"],
                gt_boxes=item["boxes"].numpy(),
                gt_labels=item["labels"].numpy(),
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                save_path=out_dir / f"{prefix}_{image_id:04d}.png",
                title=f"{split} image {image_id} (thr={score_threshold})",
            )


def run_test_evaluation_phase5b(
    checkpoint_path: Path,
    config: dict[str, Any],
    report_dir: Path = REPORT_DIR,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    """The single locked TEST evaluation for the selected Phase 5B model."""
    from accident.utils import DATASET_DIR

    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    seed_everything(config["seed"])
    device = get_device()
    image_size = tuple(config["image_size"])

    model = build_detector(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    ds = AccidentParquetDataset(split="test", target_size=image_size, dataset_dir=dataset_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_ref()
    )
    metrics = evaluate_metrics(model, loader, device, image_size, config.get("score_threshold", 0.05))
    thresholds = threshold_analysis(
        metrics["gt_records"], metrics["pred_records"],
        thresholds=THRESHOLD_SWEEP, image_size=image_size,
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "config": {k: v for k, v in config.items()},
        "num_images": len(ds),
        "num_ground_truth_objects": int(sum(len(r["boxes"]) for r in metrics["gt_records"])),
        "metrics": {
            "mAP_50": metrics["map50"],
            "mAP_50_95": metrics["map50_95"],
            "accident": {
                "AP_50": metrics["accident_ap50"],
                "AP_50_95": metrics["accident_ap50_95"],
                "precision": metrics["accident_precision"],
                "recall": metrics["accident_recall"],
            },
            "non_accident": {"AP_50": metrics["non_accident_ap50"]},
        },
        "threshold_analysis": thresholds,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "phase5b_test_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> int:
    """Run the full analysis suite for the selected Phase 5B checkpoint."""
    checkpoint = MODEL_DIR / "phase5b_fasterrcnn_best.pth"
    with open(REPORT_DIR / "phase5b_selected_config.json", encoding="utf-8") as f:
        config = json.load(f)
    image_size = tuple(config["image_size"])
    score_threshold = config.get("score_threshold", 0.05)
    seed_everything(config["seed"])

    device = get_device()
    model = build_detector(num_classes=config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)

    val_dir = FIGURE_DIR / "phase5b_val_qualitative"
    errors = collect_error_examples(model, "validation", image_size, score_threshold)

    save_error_visuals(
        model, "validation", errors["fn_example_ids"],
        FIGURE_DIR / "phase5b_false_negatives", image_size, score_threshold,
        prefix="fn",
    )
    save_error_visuals(
        model, "validation", errors["fp_example_ids"],
        FIGURE_DIR / "phase5b_false_positives", image_size, score_threshold,
        prefix="fp",
    )

    # Qualitative panels: a mix of correct hits and errors.
    ds = AccidentParquetDataset(split="validation", target_size=image_size)
    val_dir.mkdir(parents=True, exist_ok=True)
    picked = sorted(set(errors["fn_example_ids"][:3] + errors["fp_example_ids"][:3] + [0, 5, 13, 21]))
    save_error_visuals(
        model, "validation", picked, val_dir, image_size, score_threshold, prefix="val",
    )

    with open(REPORT_DIR / "phase5b_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "false_negatives": errors["fn_rows"],
                "false_positives": errors["fp_rows"],
                "fn_size_buckets": {
                    k: sum(1 for r in errors["fn_rows"] if r["size_bucket"] == k)
                    for k in ("small", "medium", "large")
                },
                "fp_size_buckets": {
                    k: sum(1 for r in errors["fp_rows"] if r["size_bucket"] == k)
                    for k in ("small", "medium", "large")
                },
            },
            f, indent=2,
        )
    print(f"FN rows: {len(errors['fn_rows'])}, FP rows: {len(errors['fp_rows'])}")
    print("Analysis artifacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())