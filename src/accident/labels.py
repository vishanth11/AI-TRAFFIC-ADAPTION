"""Label-space mapping between dataset classes and torchvision model classes.

torchvision detection models reserve internal label 0 for BACKGROUND. The
dataset uses 0 = accident and 1 = non_accident, so passing dataset labels
straight into the model turns every accident box into background supervision.

This module defines the single source of truth for the offset:

    dataset accident     (0) -> model class 1
    dataset non_accident (1) -> model class 2
    background               -> model class 0

Predictions are mapped back to dataset semantics before evaluation, the
inference contract, and visualization.
"""

from __future__ import annotations

import numpy as np
import torch

DATASET_LABEL_ACCIDENT = 0
DATASET_LABEL_NON_ACCIDENT = 1

MODEL_LABEL_BACKGROUND = 0
MODEL_LABEL_ACCIDENT = 1
MODEL_LABEL_NON_ACCIDENT = 2

LABEL_OFFSET = 1  # model label = dataset label + LABEL_OFFSET

DATASET_TO_MODEL = {
    DATASET_LABEL_ACCIDENT: MODEL_LABEL_ACCIDENT,
    DATASET_LABEL_NON_ACCIDENT: MODEL_LABEL_NON_ACCIDENT,
}

MODEL_TO_DATASET = {
    MODEL_LABEL_ACCIDENT: DATASET_LABEL_ACCIDENT,
    MODEL_LABEL_NON_ACCIDENT: DATASET_LABEL_NON_ACCIDENT,
}


def dataset_to_model_labels(
    labels: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Map dataset class ids (0/1) to model class ids (1/2).

    Model label 0 is reserved for background by torchvision detection models.
    """
    return labels + LABEL_OFFSET


def model_to_dataset_labels(
    labels: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Map model class ids (1/2) back to dataset class ids (0/1)."""
    return labels - LABEL_OFFSET


def model_label_to_category_name(model_label: int) -> str | None:
    """Return the dataset category name for a model label, or None for background."""
    dataset_label = MODEL_TO_DATASET.get(int(model_label))
    if dataset_label is None:
        return None
    return "accident" if dataset_label == DATASET_LABEL_ACCIDENT else "non_accident"