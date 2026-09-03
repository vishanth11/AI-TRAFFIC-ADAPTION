"""Phase 5B report generation: training curves, summary, comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from accident.utils import FIGURE_DIR, MODEL_DIR, REPORT_DIR


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_training_curves(history: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, [h["total"] for h in history], "k-o", label="total", lw=2)
    ax1.plot(epochs, [h["loss_classifier"] for h in history], "r-s", label="classifier")
    ax1.plot(epochs, [h["loss_box_reg"] for h in history], "b-^", label="box_reg")
    ax1.plot(epochs, [h["loss_objectness"] for h in history], "g-v", label="objectness")
    ax1.plot(epochs, [h["loss_rpn_box_reg"] for h in history], "m-d", label="rpn_box_reg")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Phase 5B training loss components")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_accident_ap50"] for h in history], "r-o", label="accident AP@50")
    ax2.plot(epochs, [h["val_map50"] for h in history], "b-s", label="mAP@50")
    ax2.plot(epochs, [h["val_accident_recall"] for h in history], "g-^", label="accident recall")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation metric")
    ax2.set_title("Phase 5B validation metrics")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_threshold_curve(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    thr = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr, [r["precision"] for r in rows], "b-o", label="precision")
    ax.plot(thr, [r["recall"] for r in rows], "r-s", label="recall")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Metric (IoU 0.5, all classes)")
    ax.set_title("Phase 5B precision/recall vs confidence (validation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(
    selected: dict,
    test_result: dict | None,
    out_path: Path,
) -> None:
    """Write phase5b_summary.md with the Phase 5 vs 5B comparison table."""
    history = selected["history"]
    best = min(history, key=lambda h: (-h["val_accident_ap50"], -h["val_map50"]))
    val = {
        "map50": selected["val_map50"],
        "map50_95": selected["val_map50_95"],
        "accident_ap50": selected["val_accident_ap50"],
        "accident_precision": selected["val_accident_precision"],
        "accident_recall": selected["val_accident_recall"],
    }
    lines = [
        "# Phase 5B — Improved Accident Detector Summary",
        "",
        f"Selected experiment: **{selected['experiment_id']}** "
        f"(best epoch {selected['best_epoch']} of {selected['epochs_completed']}).",
        "",
        "## Configuration",
        "",
        f"- Architecture: `{selected['config']['model']}`",
        f"- Pretrained: {selected['config']['pretrained']} "
        "(torchvision COCO `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`)",
        f"- Epochs trained: {selected['epochs_completed']} (early stopping patience "
        f"{selected['config'].get('early_stopping_patience', 'n/a')})",
        f"- Batch size: {selected['config']['batch_size']} — Learning rate: {selected['config']['learning_rate']}",
        f"- Image size: {selected['config']['image_size'][0]}x{selected['config']['image_size'][1]} "
        "(Phase 5 CPU-practicality choice retained)",
        f"- Class handling: background offset (dataset 0/1 → model 1/2, background 0)",
        f"- Training time: {selected['total_training_seconds']:.0f} s (CPU only)",
        f"- Checkpoint: `{selected['best_model_path']}`",
        "",
        "## Phase 5 vs Phase 5B",
        "",
        "| Metric | Phase 5 | Phase 5B |",
        "|--------|---------|----------|",
    ]
    ph5_val = {"map50": 0.0948, "map50_95": 0.0374, "accident_ap50": 0.0}
    ph5_test = {"map50": 0.0987, "map50_95": 0.0411, "accident_ap50": 0.0}
    lines.append(f"| Val mAP@50 | {ph5_val['map50']:.4f} | {val['map50']:.4f} |")
    lines.append(f"| Val mAP@50:95 | {ph5_val['map50_95']:.4f} | {val['map50_95']:.4f} |")
    lines.append(f"| Val Accident AP@50 | {ph5_val['accident_ap50']:.4f} | {val['accident_ap50']:.4f} |")
    lines.append(f"| Val Accident Precision | 0.0000 | {val['accident_precision']:.4f} |")
    lines.append(f"| Val Accident Recall | 0.0000 | {val['accident_recall']:.4f} |")
    if test_result is not None:
        tm = test_result["metrics"]
        lines.append(f"| Test mAP@50 | {ph5_test['map50']:.4f} | {tm['mAP_50']:.4f} |")
        lines.append(f"| Test mAP@50:95 | {ph5_test['map50_95']:.4f} | {tm['mAP_50_95']:.4f} |")
        lines.append(
            f"| Test Accident AP@50 | {ph5_test['accident_ap50']:.4f} | {tm['accident']['AP_50']:.4f} |"
        )
        lines.append(f"| Test Accident Precision | 0.0000 | {tm['accident']['precision']:.4f} |")
        lines.append(f"| Test Accident Recall | 0.0000 | {tm['accident']['recall']:.4f} |")
    lines += [
        "",
        "## Root cause of the Phase 5 failure",
        "",
        "The Phase 5 model was built with a 2-class head and dataset labels",
        "(0 = accident, 1 = non_accident) passed to the model un-shifted.",
        "torchvision Faster R-CNN reserves model label 0 for background, so every",
        "accident annotation was supervised as background and the model could only",
        "learn one foreground class (non_accident). Accident AP@50 = 0.0 was",
        "structural. Fixing the mapping (dataset 0/1 → model 1/2) alone raised",
        "validation accident AP@50 from 0.0 to 0.76 within 3 epochs (EXP_A).",
        "",
        "## Threshold analysis (validation, IoU 0.5, all classes)",
        "",
        "| threshold | precision | recall | TP | FP | FN |",
        "|-----------|-----------|--------|----|----|----|",
    ]
    for r in selected["threshold_analysis"]:
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['true_positives']} | {r['false_positives']} | {r['false_negatives']} |"
        )
    lines += [
        "",
        "No deployment threshold is selected in Phase 5B (deferred to the",
        "calibration/fusion phase). Confidence values are uncalibrated detector",
        "scores.",
        "",
        "## Artifacts",
        "",
        f"- Model: `{selected['best_model_path']}`",
        f"- Experiment CSV: `reports/accident/phase5b_experiments.csv`",
        f"- Experiment log: `reports/accident/phase5b_experiment_log.md`",
        f"- Training curves: `reports/figures/accident/phase5b_training_curves.png`",
        f"- Error analysis: `reports/figures/accident/phase5b_false_negatives/`, "
        "`reports/figures/accident/phase5b_false_positives/`",
        f"- Debug visuals (baseline): `reports/figures/accident/debug/`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    selected = load_json(REPORT_DIR / "phase5b_selected_summary.json")
    test_path = REPORT_DIR / "phase5b_test_results.json"
    test_result = load_json(test_path) if test_path.exists() else None

    save_training_curves(selected["history"], FIGURE_DIR / "phase5b_training_curves.png")
    save_threshold_curve(selected["threshold_analysis"], FIGURE_DIR / "phase5b_threshold_curve.png")
    write_summary(selected, test_result, REPORT_DIR / "phase5b_summary.md")
    print("Phase 5B summary and curves written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())