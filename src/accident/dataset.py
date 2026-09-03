"""Accident dataset loaders.

Reads the original Parquet files lazily and yields image + annotation records.
The canonical data source remains `dataset/`; this module does not copy or
modify the original files.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT
from .transforms import AccidentObject, ResizeTransform, clamp_boxes
from .utils import DATASET_DIR


class AccidentParquetDataset(Dataset):
    """PyTorch Dataset that streams records from the original Parquet files.

    The dataset returns a dict with:
        - "image_id": int
        - "image": torch.Tensor of shape (3, H, W)
        - "boxes": torch.Tensor of shape (N, 4) xyxy absolute
        - "labels": torch.Tensor of shape (N,) with category ids
        - "orig_size": (W, H)
    """

    def __init__(
        self,
        split: str,
        target_size: tuple[int, int] = (320, 320),
        dataset_dir: Path = DATASET_DIR,
    ):
        self.split = split
        self.target_size = target_size
        self.dataset_dir = Path(dataset_dir)
        self.resize = ResizeTransform(target_size)
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        """Read all Parquet files for this split and flatten records."""
        files = sorted(self.dataset_dir.glob(f"{self.split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No Parquet files found for split '{self.split}' in {self.dataset_dir}"
            )
        records: list[dict[str, Any]] = []
        for path in files:
            table = pq.read_table(path)
            columns = set(table.column_names)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    records.append(self._normalize_row(row, columns))
        return records

    def _normalize_row(
        self, row: dict[str, Any], columns: set[str]
    ) -> dict[str, Any]:
        """Flatten nested Parquet structs into a consistent record dict."""
        image = row["image"]
        objects = row["objects"]

        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            image_path = image.get("path")
        else:
            image_bytes = None
            image_path = str(image)

        if isinstance(objects, dict):
            bboxes = objects.get("bbox", [])
            categories = objects.get("category", [])
        elif isinstance(objects, list):
            # Some variants store a list of per-object structs.
            bboxes = [obj.get("bbox") for obj in objects]
            categories = [obj.get("category") for obj in objects]
        else:
            bboxes = []
            categories = []

        return {
            "image_bytes": image_bytes,
            "image_path": image_path,
            "bboxes": bboxes,
            "categories": categories,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_bytes = record["image_bytes"]
        if image_bytes is None:
            raise ValueError(f"Record {idx} has no embedded image bytes.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = image.size  # (width, height)

        bboxes = record["bboxes"]
        categories = record["categories"]

        objects = [
            AccidentObject(bbox, cat)
            for bbox, cat in zip(bboxes, categories)
        ]

        boxes_xyxy = np.array([obj.to_xyxy() for obj in objects], dtype=np.float32)
        labels = np.array([obj.category_id for obj in objects], dtype=np.int64)

        tensor, scaled = self.resize(image, boxes_xyxy)
        scaled = clamp_boxes(scaled, self.target_size[0], self.target_size[1])

        return {
            "image_id": idx,
            "image": tensor,
            "boxes": torch.from_numpy(scaled).float(),
            "labels": torch.from_numpy(labels).long(),
            "orig_size": orig_size,
            "num_objects": len(objects),
        }


def collate_detection(batch: list[dict[str, Any]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate function for torchvision detection models."""
    images = [item["image"] for item in batch]
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": torch.tensor(item["image_id"], dtype=torch.int64),
        }
        for item in batch
    ]
    return images, targets


def load_split_metadata(processed_dir: Path) -> dict[str, Any]:
    """Load Phase 2 split integrity and duplicate metadata if present."""
    meta_path = processed_dir / "split_integrity_report.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_dataset_statistics(split: str, dataset_dir: Path = DATASET_DIR) -> dict[str, Any]:
    """Compute image/object counts and class distributions for a split."""
    ds = AccidentParquetDataset(split=split, target_size=(640, 640), dataset_dir=dataset_dir)
    total_images = len(ds)
    total_objects = 0
    accident_objects = 0
    non_accident_objects = 0
    objects_per_image: list[int] = []

    for item in ds:
        n = item["num_objects"]
        objects_per_image.append(n)
        total_objects += n
        labels = item["labels"].numpy()
        accident_objects += int((labels == CLASS_ID_ACCIDENT).sum())
        non_accident_objects += int((labels == CLASS_ID_NON_ACCIDENT).sum())

    arr = np.array(objects_per_image, dtype=int)
    return {
        "split": split,
        "images": total_images,
        "objects": total_objects,
        "accident_objects": accident_objects,
        "non_accident_objects": non_accident_objects,
        "accident_pct": round(100.0 * accident_objects / total_objects, 2) if total_objects else 0.0,
        "non_accident_pct": round(100.0 * non_accident_objects / total_objects, 2) if total_objects else 0.0,
        "mean_objects_per_image": round(float(arr.mean()), 2) if len(arr) else 0.0,
        "median_objects_per_image": int(np.median(arr)) if len(arr) else 0,
        "min_objects_per_image": int(arr.min()) if len(arr) else 0,
        "max_objects_per_image": int(arr.max()) if len(arr) else 0,
    }
