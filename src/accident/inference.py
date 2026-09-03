"""Inference interface for the accident detector.

Defines the Phase 5 output contract. Detector confidence is returned as
`accident_detection_confidence` and is NOT claimed to be a calibrated probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from . import CLASS_NAMES
from .labels import MODEL_LABEL_BACKGROUND, model_to_dataset_labels
from .model import build_detector
from .transforms import ResizeTransform, clamp_boxes
from .utils import get_device


class AccidentDetector:
    """Callable detector wrapper that implements the Phase 5 inference contract.

    The checkpoint stores a 3-class model head (background + accident +
    non_accident in model label space); predictions are mapped back to dataset
    semantics (0 = accident, 1 = non_accident) before output.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_classes: int | None = None,
        image_size: tuple[int, int] = (320, 320),
        confidence_threshold: float = 0.5,
        device: torch.device | None = None,
    ):
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else get_device()
        self.resize = ResizeTransform(target_size=image_size)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # The 2-class Phase 5 baseline and the 3-class Phase 5B checkpoints
        # differ only in the box head width; infer it from the checkpoint.
        inferred = state_dict["roi_heads.box_predictor.cls_score.bias"].shape[0]
        self.num_classes = num_classes if num_classes is not None else inferred

        self.model = build_detector(num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run inference on a single image and return the contract output.

        Output fields:
            - detector_version: static version string
            - image_size: (width, height)
            - detections: list of {bbox, category, confidence}
            - accident_detection_confidence: aggregated detector confidence
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(
                (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).convert("RGB")

        orig_size = image.size
        tensor, _ = self.resize(image, np.zeros((0, 4), dtype=np.float32))
        tensor = tensor.to(self.device)

        outputs = self.model([tensor])
        output = outputs[0]

        thr = score_threshold if score_threshold is not None else self.confidence_threshold
        keep = (output["scores"] >= thr) & (output["labels"] != MODEL_LABEL_BACKGROUND)
        boxes_xyxy = output["boxes"][keep].cpu().numpy()
        # Model labels (1/2) -> dataset semantics (0/1) for category naming.
        labels = model_to_dataset_labels(output["labels"][keep].cpu().numpy())
        scores = output["scores"][keep].cpu().numpy()

        # Clamp boxes to image bounds for safety.
        boxes_xyxy = clamp_boxes(boxes_xyxy, self.image_size[0], self.image_size[1])

        detections = [
            {
                "bbox": [round(float(v), 2) for v in box],
                "category": CLASS_NAMES.get(int(label), str(label)),
                "confidence": round(float(score), 4),
            }
            for box, label, score in zip(boxes_xyxy, labels, scores)
        ]

        accident_scores = [
            d["confidence"] for d in detections if d["category"] == "accident"
        ]
        accident_detection_confidence = float(max(accident_scores)) if accident_scores else 0.0

        return {
            "detector_version": "accident_detector_v1",
            "image_size": [orig_size[0], orig_size[1]],
            "detections": detections,
            "accident_detection_confidence": accident_detection_confidence,
        }

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.predict(*args, **kwargs)


def infer_probability(
    image: Image.Image | np.ndarray | torch.Tensor,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience helper to run a single-image prediction."""
    detector = AccidentDetector(checkpoint_path, **kwargs)
    return detector.predict(image)
