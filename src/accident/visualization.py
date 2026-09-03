"""Visualization helpers for accident detection."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from . import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT, CLASS_NAMES


PALETTE = {
    CLASS_ID_ACCIDENT: (1.0, 0.0, 0.0),       # red
    CLASS_ID_NON_ACCIDENT: (0.0, 0.5, 1.0),     # blue
}


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) float tensor in [0, 1] to a uint8 RGB array."""
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray | None = None,
    title: str = "",
    line_width: int = 2,
    font_size: int = 10,
) -> plt.Figure:
    """Draw bounding boxes on an RGB image array."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if boxes.size == 0:
        return fig

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        cat_id = int(labels[i])
        color = PALETTE.get(cat_id, (0.5, 0.5, 0.5))
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=line_width, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        label_text = CLASS_NAMES.get(cat_id, str(cat_id))
        if confidences is not None:
            label_text += f" {confidences[i]:.2f}"
        ax.text(
            x1, y1 - 4, label_text, color="white", fontsize=font_size,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
        )
    plt.tight_layout()
    return fig


def save_ground_truth_example(
    item: dict[str, Any],
    save_path: Path,
    title: str = "Ground Truth",
) -> None:
    """Visualize one dataset item and save to disk."""
    image = denormalize_image(item["image"])
    boxes = item["boxes"].numpy()
    labels = item["labels"].numpy()
    fig = draw_boxes(image, boxes, labels, title=title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_prediction_example(
    image_tensor: torch.Tensor,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    save_path: Path,
    title: str = "Predictions",
) -> None:
    """Visualize ground truth (dashed) and predictions (solid) on one image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    image = denormalize_image(image_tensor)
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    # Ground truth boxes (dashed)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        color = PALETTE.get(int(label), (0.5, 0.5, 0.5))
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color,
            facecolor="none", linestyle="--", alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            x1, y2 + 12, f"GT {CLASS_NAMES.get(int(label), label)}",
            color="white", fontsize=9,
            bbox=dict(facecolor=color, alpha=0.5, edgecolor="none", pad=1),
        )

    # Predicted boxes (solid)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        color = PALETTE.get(int(label), (0.5, 0.5, 0.5))
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, f"Pred {CLASS_NAMES.get(int(label), label)} {score:.2f}",
            color="white", fontsize=9,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_grid(
    items: list[dict[str, Any]],
    save_path: Path,
    title: str = "Ground Truth Examples",
    max_cols: int = 5,
) -> None:
    """Save a grid of ground-truth examples."""
    n = len(items)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, item in zip(axes, items):
        image = denormalize_image(item["image"])
        ax.imshow(image)
        boxes = item["boxes"].numpy()
        labels = item["labels"].numpy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            color = PALETTE.get(int(label), (0.5, 0.5, 0.5))
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.axis("off")
        ax.set_title(f"id={item['image_id']} n={item['num_objects']}", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
