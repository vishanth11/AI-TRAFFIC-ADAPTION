"""Utility helpers for the accident detection module."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reproducible paths matching the project layout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "accident"
MODEL_DIR = PROJECT_ROOT / "models" / "accident"
REPORT_DIR = PROJECT_ROOT / "reports" / "accident"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "accident"

RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available torch device (CPU only in this phase)."""
    return torch.device("cpu")


def get_env_info() -> dict[str, Any]:
    """Return a dictionary of environment and package versions."""
    import platform

    import PIL
    import pyarrow
    import torchvision

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pillow_version": PIL.__version__,
        "pyarrow_version": pyarrow.__version__,
        "numpy_version": np.__version__,
    }
