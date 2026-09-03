"""Bounding-box and image transforms for accident detection.

The original Parquet annotations are COCO-style xywh in absolute pixels for
640x640 images. This module provides conversions to the xyxy format required
by torchvision detection models, plus resize-aware box rescaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


class AccidentObject:
    """Container for one annotation with conversion helpers."""

    def __init__(self, bbox_xywh: list[float], category_id: int):
        self.bbox_xywh = [float(v) for v in bbox_xywh]
        self.category_id = int(category_id)

    def to_xyxy(self) -> list[float]:
        """Convert COCO [x, y, w, h] to [x_min, y_min, x_max, y_max]."""
        x, y, w, h = self.bbox_xywh
        return [x, y, x + w, y + h]

    @property
    def area(self) -> float:
        return self.bbox_xywh[2] * self.bbox_xywh[3]

    def is_valid(self, image_width: int, image_height: int) -> bool:
        """Check that the box has positive area and lies inside the image."""
        x, y, w, h = self.bbox_xywh
        if w <= 0 or h <= 0:
            return False
        if x < 0 or y < 0:
            return False
        if x + w > image_width or y + h > image_height:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xywh": self.bbox_xywh,
            "bbox_xyxy": self.to_xyxy(),
            "category_id": self.category_id,
            "area": self.area,
        }


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Vectorized COCO xywh -> xyxy conversion."""
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes_xywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = np.copy(boxes_xywh)
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Vectorized xyxy -> COCO xywh conversion."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = np.copy(boxes_xyxy)
    xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    return xywh


def scale_boxes(
    boxes_xyxy: np.ndarray,
    orig_size: tuple[int, int],
    new_size: tuple[int, int],
) -> np.ndarray:
    """Rescale absolute xyxy boxes after image resize.

    Args:
        boxes_xyxy: array of shape (N, 4) in original absolute coordinates.
        orig_size: (width, height) of the source image.
        new_size: (width, height) of the resized image.

    Returns:
        Rescaled boxes in the new image coordinate system.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    sx = new_w / orig_w
    sy = new_h / orig_h
    scaled = np.copy(boxes_xyxy)
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return scaled


class ResizeTransform:
    """Resize image and rescale bounding boxes deterministically."""

    def __init__(self, target_size: tuple[int, int] = (320, 320)):
        self.target_size = target_size  # (width, height)

    def __call__(
        self,
        image: Image.Image,
        boxes_xyxy: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        orig_size = image.size  # (width, height)
        resized = F.resize(image, self.target_size[::-1])  # PIL expects (h, w)
        tensor = F.pil_to_tensor(resized).float() / 255.0
        scaled = scale_boxes(boxes_xyxy, orig_size, self.target_size)
        return tensor, scaled


def clamp_boxes(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp xyxy boxes to image bounds and ensure x2 > x1, y2 > y1."""
    boxes = np.copy(boxes_xyxy)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1e-3)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1e-3)
    return boxes
