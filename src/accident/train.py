"""Training loop for the accident detector.

Uses a lightweight Faster R-CNN with MobileNetV3 backbone. Trains on CPU with
a modest configuration suitable for the 2,122-image accident dataset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT
from .dataset import AccidentParquetDataset, collate_detection
from .labels import dataset_to_model_labels, model_to_dataset_labels
from .metrics import evaluate_coco
from .model import build_detector, count_parameters
from .utils import DATASET_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR, get_device, seed_everything


DEFAULT_CONFIG = {
    "seed": 42,
    "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
    # background (0) + accident (1) + non_accident (2) in model label space.
    "num_classes": 3,
    "image_size": [320, 320],
    "batch_size": 4,
    "epochs": 2,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    # Step of 2 was tuned for the 2-epoch Phase 5 run; a longer schedule needs
    # a later decay to keep learning past the first epochs.
    "lr_scheduler_step": 8,
    "lr_scheduler_gamma": 0.1,
    "num_workers": 0,
    "pretrained": True,
    "device": "cpu",
}


def build_dataloaders(
    config: dict[str, Any],
    dataset_dir: Path | None = None,
) -> dict[str, DataLoader]:
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    """Create train/validation DataLoaders."""
    target_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 0)

    train_ds = AccidentParquetDataset(
        split="train", target_size=target_size, dataset_dir=dataset_dir
    )
    val_ds = AccidentParquetDataset(
        split="validation", target_size=target_size, dataset_dir=dataset_dir
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_detection,
    )
    return {"train": train_loader, "validation": val_loader}


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch and return mean per-component losses.

    Dataset labels (0 = accident, 1 = non_accident) are offset by +1 before
    reaching the model, because torchvision reserves model label 0 for
    background — see `accident.labels`.
    """
    model.train()
    totals = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total": 0.0,
    }
    count = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{**t, "labels": dataset_to_model_labels(t["labels"])} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        for key in totals:
            component = loss if key == "total" else loss_dict.get(key)
            if component is not None:
                totals[key] += component.item()
        count += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {k: v / count for k, v in totals.items()} if count else totals


@torch.no_grad()
def gather_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run inference on a split and return (gt_records, pred_records).

    Ground truth stays in dataset label space (0 = accident, 1 = non_accident).
    Model predictions arrive in model label space (1 = accident, 2 =
    non_accident, 0 = background) and are mapped back to dataset space here.
    """
    model.eval()
    gt_records: list[dict[str, Any]] = []
    pred_records: list[dict[str, Any]] = []

    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images_device = [img.to(device) for img in images]
        outputs = model(images_device)

        for img_idx, (target, output) in enumerate(zip(targets, outputs)):
            image_id = int(target["image_id"].item())
            gt_records.append(
                {
                    "image_id": image_id,
                    "boxes": target["boxes"].cpu().numpy(),
                    "labels": target["labels"].cpu().numpy(),
                }
            )
            keep = output["scores"].cpu() >= score_threshold
            pred_labels = model_to_dataset_labels(output["labels"].cpu()[keep].numpy())
            pred_records.append(
                {
                    "image_id": image_id,
                    "boxes": output["boxes"].cpu()[keep].numpy(),
                    "labels": pred_labels,
                    "scores": output["scores"].cpu()[keep].numpy(),
                }
            )
    return gt_records, pred_records


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int] = (320, 320),
) -> dict[str, Any]:
    """Evaluate a split and return COCO metrics."""
    gt_records, pred_records = gather_predictions(model, loader, device)
    return evaluate_coco(gt_records, pred_records, image_size)


def train_detector(
    config: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    figure_dir: Path = FIGURE_DIR,
) -> dict[str, Any]:
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    """Run the full training and validation loop.

    Returns a summary dict with the best validation metrics, final test metrics,
    and saved artifact paths.
    """
    config = dict(DEFAULT_CONFIG if config is None else config)
    seed_everything(config["seed"])
    device = get_device()

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(config, dataset_dir=dataset_dir)
    train_loader = loaders["train"]
    val_loader = loaders["validation"]

    model = build_detector(
        num_classes=config["num_classes"], pretrained=config.get("pretrained", True)
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_scheduler_step"],
        gamma=config["lr_scheduler_gamma"],
    )

    image_size = tuple(config["image_size"])
    history: list[dict[str, Any]] = []
    best_map50 = -1.0
    best_epoch = -1
    best_model_path = model_dir / "best_detector.pth"

    print(f"Training {config['model']} on {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics = evaluate_split(model, val_loader, device, image_size)
        lr_scheduler.step()

        val_map50 = val_metrics["overall"]["mAP_50"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_losses["total"], 6),
                "train_loss_classifier": round(train_losses["loss_classifier"], 6),
                "train_loss_box_reg": round(train_losses["loss_box_reg"], 6),
                "train_loss_objectness": round(train_losses["loss_objectness"], 6),
                "train_loss_rpn_box_reg": round(train_losses["loss_rpn_box_reg"], 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val_mAP_50": round(val_map50, 4),
                "val_mAP_50_95": round(val_metrics["overall"]["mAP_50_95"], 4),
                "val_accident_AP": round(val_metrics["per_class"]["accident"]["AP_50"], 4),
            }
        )

        print(
            f"Epoch {epoch}/{config['epochs']} — "
            f"train_loss={train_loss:.4f} — "
            f"val mAP@50={val_map50:.4f} — "
            f"val accident AP={val_metrics['per_class']['accident']['AP_50']:.4f}"
        )

        if val_map50 > best_map50:
            best_map50 = val_map50
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    total_time = time.time() - start_time

    # Save final checkpoint (even if not best) for completeness.
    final_path = model_dir / "final_detector.pth"
    torch.save(model.state_dict(), final_path)

    # Save config and history.
    config_path = report_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history_path = report_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Reload best model for final reporting.
    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state)
    best_val_metrics = evaluate_split(model, val_loader, device, image_size)

    summary = {
        "config": config,
        "device": str(device),
        "trainable_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_mAP_50": best_map50,
        "total_training_seconds": round(total_time, 1),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_path),
        "validation_metrics": best_val_metrics,
        "training_history": history,
    }

    # Save a summary JSON for later phases.
    summary_path = report_dir / "phase5_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = train_detector()
    print("\nTraining complete.")
    print(f"Best validation mAP@50: {summary['best_val_mAP_50']:.4f} (epoch {summary['best_epoch']})")
