"""Visual debugging of the Phase 5 baseline detector.

Generates reports/figures/accident/debug/ containing ground-truth examples and
baseline-model predictions with predicted classes and confidence values, then
summarizes the baseline's failure mode (which classes it emits, confidence
distribution).

Uses the IMMUTABLE Phase 5 checkpoint (models/accident/best_detector.pth),
which has a 2-class head — loaded explicitly with num_classes=2.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from accident import CLASS_NAMES
from accident.dataset import AccidentParquetDataset
from accident.model import build_detector
from accident.utils import FIGURE_DIR, MODEL_DIR, get_device, seed_everything
from accident.visualization import save_ground_truth_example, save_prediction_example

NUM_EXAMPLES = 24
SCORE_THRESHOLD = 0.05


def main() -> int:
    seed_everything(42)
    device = get_device()
    out_dir = FIGURE_DIR / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Ground-truth examples -------------------------------------------------
    ds = AccidentParquetDataset(split="validation", target_size=(320, 320))
    # Pick a spread of images: accident-heavy, non-accident-heavy, mixed.
    indices: list[int] = []
    for i in range(len(ds)):
        item = ds[i]
        labels = item["labels"].numpy()
        n_acc = int((labels == 0).sum())
        n_non = int((labels == 1).sum())
        if n_acc >= 2 and len(indices) % 2 == 0 and len(indices) < NUM_EXAMPLES:
            indices.append(i)
        elif n_non > n_acc and len(indices) < NUM_EXAMPLES:
            indices.append(i)
    indices = indices[:NUM_EXAMPLES]

    for i in indices:
        item = ds[i]
        gt_labels = item["labels"].numpy()
        n_acc = int((gt_labels == 0).sum())
        n_non = int((gt_labels == 1).sum())
        save_ground_truth_example(
            item,
            out_dir / f"gt_{i:04d}.png",
            title=f"GT — image {i}: {n_acc} accident / {n_non} non_accident",
        )

    # --- Baseline predictions --------------------------------------------------
    model = build_detector(num_classes=2, pretrained=False)
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_detector.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    label_counts: Counter = Counter()
    score_by_label: dict[int, list[float]] = {0: [], 1: []}
    summary_rows: list[dict] = []

    with torch.no_grad():
        for i in indices:
            item = ds[i]
            outputs = model([item["image"].to(device)])
            output = outputs[0]
            keep = output["scores"] >= SCORE_THRESHOLD
            pred_boxes = output["boxes"][keep].cpu().numpy()
            # The baseline checkpoint was trained with the un-shifted (buggy)
            # label space, where its single foreground class 1 was supervised
            # ONLY by non_accident boxes. Interpret it with that identity
            # mapping, not the corrected +1 mapping.
            pred_labels = output["labels"][keep].cpu().numpy()
            pred_scores = output["scores"][keep].cpu().numpy()

            for lbl, sc in zip(pred_labels.tolist(), pred_scores.tolist()):
                label_counts[lbl] += 1
                score_by_label[lbl].append(sc)

            save_prediction_example(
                image_tensor=item["image"],
                gt_boxes=item["boxes"].numpy(),
                gt_labels=item["labels"].numpy(),
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                save_path=out_dir / f"baseline_pred_{i:04d}.png",
                title=f"Baseline prediction — image {i} (thr={SCORE_THRESHOLD})",
            )
            summary_rows.append(
                {
                    "image_id": i,
                    "gt_accident": int((item["labels"].numpy() == 0).sum()),
                    "gt_non_accident": int((item["labels"].numpy() == 1).sum()),
                    "pred_accident": int((pred_labels == 0).sum()),
                    "pred_non_accident": int((pred_labels == 1).sum()),
                    "max_score": float(pred_scores.max()) if len(pred_scores) else 0.0,
                }
            )

    all_scores = [s for scores in score_by_label.values() for s in scores]
    report = {
        "checkpoint": "models/accident/best_detector.pth (Phase 5 baseline, num_classes=2)",
        "num_examples": len(indices),
        "score_threshold": SCORE_THRESHOLD,
        "predicted_label_counts_dataset_space": {
            CLASS_NAMES.get(k, str(k)): v for k, v in sorted(label_counts.items())
        },
        "mean_score_by_label": {
            CLASS_NAMES.get(k, str(k)): round(float(np.mean(v)), 4) if v else None
            for k, v in score_by_label.items()
        },
        "max_score_overall": round(float(np.max(all_scores)), 4) if all_scores else None,
        "images": summary_rows,
    }
    with open(out_dir / "baseline_debug_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Predicted class counts (dataset space):", dict(report["predicted_label_counts_dataset_space"]))
    print("Mean score by class:", report["mean_score_by_label"])
    print(f"Debug visuals written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())