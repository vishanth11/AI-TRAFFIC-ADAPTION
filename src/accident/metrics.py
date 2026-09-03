"""Detection metrics using pycocotools.

Provides COCO-style evaluation: mAP@50, mAP@50:95, per-class AP, precision,
and recall. All metrics are computed at the object level, not the image level.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT


def predictions_to_coco(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert ground-truth and prediction records to COCO-format dictionaries.

    Args:
        gt_records: list of dicts with keys "image_id", "boxes" (xyxy), "labels".
        pred_records: list of dicts with keys "image_id", "boxes" (xyxy), "labels", "scores".
        image_size: (width, height) assumed for all images in this evaluation.

    Returns:
        (gt_coco, pred_coco) dictionaries ready for pycocotools.
    """
    width, height = image_size
    categories = [
        {"id": CLASS_ID_ACCIDENT, "name": "accident"},
        {"id": CLASS_ID_NON_ACCIDENT, "name": "non_accident"},
    ]

    images = []
    gt_annotations = []
    pred_annotations = []
    ann_id = 1

    seen_ids = set()
    for rec in gt_records:
        image_id = int(rec["image_id"])
        if image_id not in seen_ids:
            images.append({"id": image_id, "width": width, "height": height})
            seen_ids.add(image_id)
        boxes = rec["boxes"]
        labels = rec["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            gt_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    seen_ids = set()
    for rec in pred_records:
        image_id = int(rec["image_id"])
        if image_id not in seen_ids:
            images.append({"id": image_id, "width": width, "height": height})
            seen_ids.add(image_id)
        boxes = rec["boxes"]
        labels = rec["labels"]
        scores = rec["scores"]
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            pred_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt_coco = {
        "images": images,
        "annotations": gt_annotations,
        "categories": categories,
    }
    pred_coco = {
        "images": images,
        "annotations": pred_annotations,
        "categories": categories,
    }
    return gt_coco, pred_coco


def evaluate_coco(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    image_size: tuple[int, int] = (320, 320),
) -> dict[str, Any]:
    """Compute COCO metrics from ground-truth and prediction records.

    Returns a dictionary with overall and per-class mAP/precision/recall.
    """
    if not gt_records or not pred_records:
        return _empty_metrics()

    gt_coco, pred_coco = predictions_to_coco(gt_records, pred_records, image_size)
    gt_coco_api = COCO()
    gt_coco_api.dataset = gt_coco
    gt_coco_api.createIndex()

    pred_coco_api = COCO()
    pred_coco_api.dataset = pred_coco
    pred_coco_api.createIndex()

    coco_eval = COCOeval(gt_coco_api, pred_coco_api, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall = {
        "mAP_50_95": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }

    per_class = _per_class_metrics(coco_eval)

    return {
        "overall": overall,
        "per_class": per_class,
    }


def _per_class_metrics(coco_eval: COCOeval) -> dict[str, dict[str, float]]:
    """Extract per-class AP, precision, and recall from a COCOeval object."""
    # pycocotools layout: precision[IoU, recall, class, area, maxDets]
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    category_ids = coco_eval.params.catIds
    iou_lo = np.where(coco_eval.params.iouThrs == 0.5)[0][0]

    result: dict[str, dict[str, float]] = {}
    for idx, cat_id in enumerate(category_ids):
        # mAP@50 for this class
        cls_precision = precision[iou_lo, :, idx, 0, 2]
        valid = cls_precision[cls_precision > -1]
        ap50 = float(valid.mean()) if len(valid) else 0.0

        # mAP@50:95 for this class
        cls_precision_all_iou = precision[:, :, idx, 0, 2]
        valid_all = cls_precision_all_iou[cls_precision_all_iou > -1]
        ap5095 = float(valid_all.mean()) if len(valid_all) else 0.0

        # Recall at max detections
        cls_recall = recall[:, idx, 0, 2]
        valid_recall = cls_recall[cls_recall > -1]
        rec = float(valid_recall.mean()) if len(valid_recall) else 0.0

        # Approximate precision as AP-like mean precision at all recall thresholds
        prec = ap50

        cat_name = "accident" if cat_id == CLASS_ID_ACCIDENT else "non_accident"
        result[cat_name] = {
            "AP_50": ap50,
            "AP_50_95": ap5095,
            "precision": prec,
            "recall": rec,
        }
    return result


def _empty_metrics() -> dict[str, Any]:
    return {
        "overall": {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
        },
        "per_class": {
            "accident": {"AP_50": 0.0, "AP_50_95": 0.0, "precision": 0.0, "recall": 0.0},
            "non_accident": {"AP_50": 0.0, "AP_50_95": 0.0, "precision": 0.0, "recall": 0.0},
        },
    }


def threshold_analysis(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    thresholds: list[float] | None = None,
    iou_threshold: float = 0.5,
    image_size: tuple[int, int] = (320, 320),
    class_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Compute precision/recall/FP/FN at a range of confidence thresholds.

    This is a simplified per-threshold analysis that matches each prediction to
    at most one ground-truth box of the same class using IoU. When
    `class_filter` is set, only that dataset class id is counted for both
    predictions and ground truth (e.g. accident-only threshold study).
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Group by image and class.
    gt_by_image: dict[int, dict[int, list[tuple[int, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    for rec in gt_records:
        image_id = int(rec["image_id"])
        for i, (box, label) in enumerate(zip(rec["boxes"], rec["labels"])):
            label = int(label)
            if class_filter is not None and label != class_filter:
                continue
            gt_by_image[image_id][label].append((i, np.asarray(box, dtype=np.float32)))

    rows = []
    for thr in thresholds:
        tp = 0
        fp = 0
        total_pred = 0
        total_gt = sum(
            len(boxes)
            for img in gt_by_image.values()
            for boxes in img.values()
        )
        matched: set[tuple[int, int]] = set()

        for rec in pred_records:
            image_id = int(rec["image_id"])
            for box, label, score in zip(rec["boxes"], rec["labels"], rec["scores"]):
                if score < thr:
                    continue
                label = int(label)
                if class_filter is not None and label != class_filter:
                    continue
                total_pred += 1
                box_arr = np.asarray(box, dtype=np.float32)
                best_iou = 0.0
                best_key: tuple[int, int] | None = None
                for gt_idx, gt_box in gt_by_image[image_id].get(label, []):
                    key = (image_id, gt_idx)
                    if key in matched:
                        continue
                    iou = _compute_iou(box_arr, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_key = key

                if best_key is not None and best_iou >= iou_threshold:
                    matched.add(best_key)
                    tp += 1
                else:
                    fp += 1

        fn = total_gt - len(matched)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        rows.append(
            {
                "threshold": thr,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )
    return rows


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0
