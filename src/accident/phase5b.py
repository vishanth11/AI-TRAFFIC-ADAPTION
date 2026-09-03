"""Phase 5B — improved accident-detector training and experiment tracking.

Builds on the Phase 5 pipeline with the verified fixes:
    - 3-class model head (background + accident + non_accident) with explicit
      label-space mapping (`accident.labels`).
    - Per-component loss tracking.
    - Per-epoch checkpointing, validation evaluation, best-checkpoint
      selection on validation accident AP@50 (primary) with mAP@50 tie-break,
      and optional early stopping.
    - CSV experiment tracking under reports/accident/phase5b_experiments.csv.

The Phase 5 baseline checkpoint (models/accident/best_detector.pth) is never
touched. Phase 5B artifacts go to models/accident/phase5b_*.pth.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import AccidentParquetDataset, collate_detection
from .metrics import evaluate_coco, threshold_analysis
from .model import NUM_CLASSES, build_detector, count_parameters
from .train import gather_predictions
from .utils import DATASET_DIR, MODEL_DIR, REPORT_DIR, get_device, seed_everything

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "model",
    "pretrained",
    "epochs",
    "batch_size",
    "learning_rate",
    "augmentation",
    "class_handling",
    "val_map50",
    "val_map50_95",
    "val_accident_ap50",
    "val_accident_precision",
    "val_accident_recall",
    "training_time",
    "notes",
]

THRESHOLD_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

PHASE5B_CONFIG = {
    "seed": 42,
    "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "num_classes": NUM_CLASSES,
    "image_size": [320, 320],
    "batch_size": 4,
    "epochs": 25,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler_step": 8,
    "lr_scheduler_gamma": 0.1,
    "num_workers": 0,
    "pretrained": True,
    "device": "cpu",
    "early_stopping_patience": 6,
    "min_improvement": 0.002,
    "score_threshold": 0.05,
}


def checkpoint_score(metrics: dict[str, float]) -> tuple[float, float]:
    """Model-selection score: accident AP@50 primary, mAP@50 tie-break."""
    return (round(metrics["accident_ap50"], 4), round(metrics["map50"], 4))


def is_improvement(score: float, best_score: float, min_improvement: float = 0.0) -> bool:
    """True when the new score beats the best by at least min_improvement."""
    return score > best_score + min_improvement


def append_experiment_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one experiment row, creating the CSV with a header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in EXPERIMENT_COLUMNS})


def build_loaders(
    config: dict[str, Any],
    dataset_dir: Path = DATASET_DIR,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders for one experiment."""
    target_size = tuple(config["image_size"])
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "collate_fn": collate_detection,
    }
    train_ds = AccidentParquetDataset(split="train", target_size=target_size, dataset_dir=dataset_dir)
    val_ds = AccidentParquetDataset(split="validation", target_size=target_size, dataset_dir=dataset_dir)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int],
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Return flattened validation metrics used for selection and tracking."""
    gt_records, pred_records = gather_predictions(
        model, val_loader, device, score_threshold=score_threshold
    )
    metrics = evaluate_coco(gt_records, pred_records, image_size)
    accident = metrics["per_class"]["accident"]
    return {
        "map50": metrics["overall"]["mAP_50"],
        "map50_95": metrics["overall"]["mAP_50_95"],
        "accident_ap50": accident["AP_50"],
        "accident_ap50_95": accident["AP_50_95"],
        "accident_precision": accident["precision"],
        "accident_recall": accident["recall"],
        "non_accident_ap50": metrics["per_class"]["non_accident"]["AP_50"],
        "gt_records": gt_records,
        "pred_records": pred_records,
    }


def run_experiment(
    config: dict[str, Any],
    experiment_id: str,
    notes: str,
    augmentation: str = "none",
    class_handling: str = "background_offset",
    dataset_dir: Path = DATASET_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    csv_path: Path | None = None,
    best_model_name: str = "phase5b_fasterrcnn_best.pth",
) -> dict[str, Any]:
    """Train one experiment with per-epoch checkpointing and validation eval.

    The best checkpoint (by validation accident AP@50, mAP@50 tie-break) is
    saved to `model_dir / best_model_name`. A rolling last checkpoint is kept
    so long CPU runs can resume.
    """
    config = dict(config)
    seed_everything(config["seed"])
    device = get_device()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if csv_path is None:
        csv_path = report_dir / "phase5b_experiments.csv"

    train_loader, val_loader = build_loaders(config, dataset_dir=dataset_dir)
    model = build_detector(
        num_classes=config["num_classes"], pretrained=config.get("pretrained", True)
    )
    model.to(device)

    # Optional resume from an interrupted-run checkpoint (see
    # `save_interrupted_checkpoint`). SGD momentum buffers cannot be
    # serialized out of a killed process; they re-warm within a few steps.
    start_epoch = 1
    resume_path = config.get("resume_from")
    if resume_path:
        with open(resume_path, "rb") as f:
            resume_ckpt = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        print(
            f"[{experiment_id}] resuming from {resume_path} "
            f"(epoch {int(resume_ckpt['epoch'])}, best score {resume_ckpt.get('best_score_tuple')})"
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("lr_scheduler_step", 8),
        gamma=config.get("lr_scheduler_gamma", 0.1),
        last_epoch=start_epoch - 1,  # fast-forward the schedule on resume
    )

    image_size = tuple(config["image_size"])
    score_threshold = config.get("score_threshold", 0.05)
    patience = config.get("early_stopping_patience", 0)
    min_improvement = config.get("min_improvement", 0.002)

    best_score_tuple = (-1.0, -1.0)
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if resume_path:
        best_score_tuple = tuple(resume_ckpt.get("best_score_tuple", (-1.0, -1.0)))
        best_epoch = int(resume_ckpt.get("best_epoch", -1))
        history = list(resume_ckpt.get("history", []))

    best_path = model_dir / best_model_name
    last_path = model_dir / best_model_name.replace(".pth", "_last.pth")
    history_path = report_dir / f"phase5b_{experiment_id.lower()}_history.json"

    print(f"[{experiment_id}] training {config['model']} on {device}")
    print(f"[{experiment_id}] trainable parameters: {count_parameters(model):,}")
    start_time = time.time()

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start = time.time()
        losses = train_one_epoch_components(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        metrics = evaluate_metrics(model, val_loader, device, image_size, score_threshold)
        score_tuple = checkpoint_score(metrics)

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{k: round(v, 6) for k, v in losses.items()},
                "val_map50": round(metrics["map50"], 4),
                "val_map50_95": round(metrics["map50_95"], 4),
                "val_accident_ap50": round(metrics["accident_ap50"], 4),
                "val_accident_precision": round(metrics["accident_precision"], 4),
                "val_accident_recall": round(metrics["accident_recall"], 4),
                "val_non_accident_ap50": round(metrics["non_accident_ap50"], 4),
                "epoch_seconds": round(time.time() - epoch_start, 1),
            }
        )
        print(
            f"[{experiment_id}] epoch {epoch}/{config['epochs']} — "
            f"loss={losses['total']:.4f} — val accident AP@50={metrics['accident_ap50']:.4f} — "
            f"val mAP@50={metrics['map50']:.4f} — recall={metrics['accident_recall']:.4f}"
        )

        # Rolling checkpoint so long CPU runs are not lost.
        torch.save(model.state_dict(), last_path)

        if score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_epoch = epoch
            best_metrics = {k: v for k, v in metrics.items() if not k.endswith("_records")}
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
            print(f"[{experiment_id}] new best checkpoint (epoch {epoch})")
        else:
            epochs_without_improvement += 1

        # Flush history every epoch so long CPU runs are observable.
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "config": config,
                    "history": history,
                },
                f,
                indent=2,
            )

        if patience and epochs_without_improvement >= patience:
            print(f"[{experiment_id}] early stopping after {epoch} epochs (no improvement for {patience})")
            break

    total_time = time.time() - start_time

    # Reload the best checkpoint for final reporting.
    if best_metrics is None:
        raise RuntimeError(f"[{experiment_id}] no checkpoint was selected")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_metrics = evaluate_metrics(model, val_loader, device, image_size, score_threshold)

    thresholds = threshold_analysis(
        final_metrics["gt_records"],
        final_metrics["pred_records"],
        thresholds=THRESHOLD_SWEEP,
        image_size=image_size,
    )

    summary = {
        "experiment_id": experiment_id,
        "notes": notes,
        "config": config,
        "augmentation": augmentation,
        "class_handling": class_handling,
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"],
        "trainable_parameters": count_parameters(model),
        "total_training_seconds": round(total_time, 1),
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
        "val_map50": final_metrics["map50"],
        "val_map50_95": final_metrics["map50_95"],
        "val_accident_ap50": final_metrics["accident_ap50"],
        "val_accident_precision": final_metrics["accident_precision"],
        "val_accident_recall": final_metrics["accident_recall"],
        "val_non_accident_ap50": final_metrics["non_accident_ap50"],
        "threshold_analysis": thresholds,
        "history": history,
    }

    # Persist per-experiment artifacts.
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    append_experiment_row(
        csv_path,
        {
            "experiment_id": experiment_id,
            "model": config["model"],
            "pretrained": config.get("pretrained", True),
            "epochs": summary["epochs_completed"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "augmentation": augmentation,
            "class_handling": class_handling,
            "val_map50": round(final_metrics["map50"], 4),
            "val_map50_95": round(final_metrics["map50_95"], 4),
            "val_accident_ap50": round(final_metrics["accident_ap50"], 4),
            "val_accident_precision": round(final_metrics["accident_precision"], 4),
            "val_accident_recall": round(final_metrics["accident_recall"], 4),
            "training_time": round(total_time, 1),
            "notes": notes,
        },
    )
    return summary


def train_one_epoch_components(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """One training epoch; dataset labels are offset to model space (+1)."""
    from .labels import dataset_to_model_labels

    model.train()
    totals = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total": 0.0,
    }
    count = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{**t, "labels": dataset_to_model_labels(t["labels"])} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        totals["total"] += loss.item()
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            totals[key] += loss_dict[key].item()
        count += 1

    return {k: v / count for k, v in totals.items()} if count else totals


def save_interrupted_checkpoint(
    model_checkpoint_path: Path,
    history_path: Path,
    config: dict[str, Any],
    out_path: Path | None = None,
) -> Path:
    """Bundle the latest valid state of an interrupted run for resuming.

    Includes: model state_dict, completed epoch, best validation score so far,
    best checkpoint path, training configuration, seed, and the per-epoch
    history. SGD momentum buffers cannot be recovered from a stopped process
    (optimizer state was held in process memory); this is documented in the
    checkpoint and the momentum buffers re-warm within a few optimizer steps.
    StepLR state is reconstructed deterministically from epoch + schedule.
    """
    if out_path is None:
        out_path = MODEL_DIR / "phase5b_interrupted_checkpoint.pth"
    state = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)

    epoch = 0
    history: list[dict[str, Any]] = []
    best_score_tuple = (-1.0, -1.0)
    best_epoch = -1
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            payload = json.load(f)
        entries = payload.get("history", payload if isinstance(payload, list) else [])
        history = entries
        if entries:
            epoch = int(entries[-1]["epoch"])
            best_entry = max(entries, key=lambda h: (h["val_accident_ap50"], h["val_map50"]))
            best_score_tuple = (best_entry["val_accident_ap50"], best_entry["val_map50"])
            best_epoch = int(best_entry["epoch"])

    ckpt = {
        "model_state_dict": state,
        "epoch": epoch,
        "best_score_tuple": best_score_tuple,
        "best_epoch": best_epoch,
        "config": config,
        "history": history,
        "scheduler": {
            "type": "StepLR",
            "step_size": config.get("lr_scheduler_step", 8),
            "gamma": config.get("lr_scheduler_gamma", 0.1),
            "last_epoch": epoch,
        },
        "optimizer_state": None,  # SGD momentum not recoverable after process stop
        "interrupted": True,
        "notes": "Saved at an epoch boundary during intentional Phase 5B "
        "interruption. Resume via run_experiment config 'resume_from'.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    metadata = {
        "checkpoint_path": str(out_path),
        "model_checkpoint_source": str(model_checkpoint_path),
        "completed_epoch": epoch,
        "best_score_tuple": list(best_score_tuple),
        "best_epoch": best_epoch,
        "history_path": str(history_path),
        "config": config,
        "optimizer_state_included": False,
        "optimizer_note": "SGD momentum buffers lost on process stop; "
        "scheduler reconstructed from epoch. Resume re-warms momentum.",
        "baseline_model_untouched": "models/accident/best_detector.pth",
    }
    with open(MODEL_DIR / "phase5b_checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def load_phase5b_detector(
    checkpoint_path: Path | None = None,
    image_size: tuple[int, int] = (320, 320),
):
    """Load the Phase 5B detector through the inference contract wrapper."""
    from .inference import AccidentDetector

    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / "phase5b_fasterrcnn_best.pth"
    return AccidentDetector(checkpoint_path, num_classes=NUM_CLASSES, image_size=image_size)