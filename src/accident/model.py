"""Detector model construction.

Uses torchvision Faster R-CNN with a MobileNetV3-Large-320 backbone. This is a
lightweight, CPU-friendly detector suitable for the small accident dataset and
the later safety-intelligence fusion stage.

Class layout: torchvision reserves internal label 0 for background, so the
model is built with num_classes = 3 (background + accident + non_accident).
Dataset labels (0 = accident, 1 = non_accident) are offset by +1 before
training and predictions are mapped back — see `accident.labels`.
"""

from __future__ import annotations

import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .labels import MODEL_LABEL_BACKGROUND, MODEL_LABEL_ACCIDENT, MODEL_LABEL_NON_ACCIDENT

# background (0) + accident (1) + non_accident (2), in model label space.
NUM_CLASSES = 3


def build_detector(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    """Build a Faster R-CNN detector.

    Args:
        num_classes: total model classes INCLUDING background. Must be 3 for
            the two-class accident dataset (background is class 0 internally).
        pretrained: whether to load the standard torchvision COCO-pretrained
            weights for the MobileNetV3-Large-320-FPN detector.

    Returns:
        A torchvision FasterRCNN model ready for fine-tuning.
    """
    if num_classes < 2:
        raise ValueError("num_classes must include background (>= 2).")
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        trainable_backbone_layers=3,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_parameters(model) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)