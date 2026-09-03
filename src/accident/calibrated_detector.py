"""Calibrated accident detector — Phase 6 inference interface.

Loads `models/accident/calibrated_accident_detector.pth`, which bundles the
Phase 5B detector weights with the validation-fitted calibrator, and exposes
the Phase 6 inference contract:

    {
        "detector_version": ...,
        "image_size": [w, h],
        "detections": [{bbox, category, confidence}, ...],
        "accident_detection_confidence": <raw detector signal, NOT a probability>,
        "calibrated_accident_probability": <P(accident) in [0, 1]>,
        "calibration": {"method": ..., "signal": ...},
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import SIGNAL_FUNCTIONS, SIGNAL_SCORE_FLOOR, calibrator_from_dict
from .utils import MODEL_DIR

CALIBRATED_CHECKPOINT_PATH = MODEL_DIR / "calibrated_accident_detector.pth"


def save_calibrated_checkpoint(
    detector_state_dict: dict[str, torch.Tensor],
    calibration_dict: dict[str, Any],
    metadata: dict[str, Any],
    output_path: Path = CALIBRATED_CHECKPOINT_PATH,
) -> Path:
    """Bundle detector weights + fitted calibrator into one checkpoint file.

    The payload is a plain dict of tensors/floats/strings/lists so it stays
    loadable with `torch.load(weights_only=True)`.
    """
    payload = {
        "format_version": 1,
        "model_state_dict": detector_state_dict,
        "calibration": calibration_dict,
        "metadata": {
            "checkpoint_source": metadata.get("checkpoint_source", ""),
            "fit_split": metadata.get("fit_split", "validation"),
            "signal": metadata.get("signal", "max"),
            "fit_image_count": int(metadata.get("fit_image_count", 0)),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


class CalibratedAccidentDetector:
    """Detector + calibrator implementing the Phase 6 inference contract.

    The raw signal (`accident_detection_confidence`) is aggregated exactly as
    during calibration fitting so the calibrator sees the same distribution.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = CALIBRATED_CHECKPOINT_PATH,
        image_size: tuple[int, int] = (320, 320),
        confidence_threshold: float = 0.5,
        device: torch.device | None = None,
    ):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.calibration_dict = dict(payload["calibration"])
        self.calibrator = calibrator_from_dict(self.calibration_dict)
        self.signal_name = self.calibration_dict.get("signal", "max")
        if self.signal_name not in SIGNAL_FUNCTIONS:
            raise ValueError(f"Unknown signal in checkpoint: {self.signal_name!r}")
        self.metadata = dict(payload.get("metadata", {}))

        # The bundled weights are the plain 3-class detector state dict. Only
        # Phase 5B 3-class checkpoints may be calibrated: the deployed signal
        # path applies the background-offset label mapping.
        state_dict = payload["model_state_dict"]
        inferred = state_dict["roi_heads.box_predictor.cls_score.bias"].shape[0]
        if inferred != 3:
            raise ValueError(
                f"{checkpoint_path} bundles a {inferred}-class detector head; "
                "Phase 6 calibration supports only 3-class Phase 5B checkpoints."
            )

        from .model import build_detector
        from .transforms import ResizeTransform
        from .utils import get_device

        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else get_device()
        self.resize = ResizeTransform(target_size=image_size)
        self.model = build_detector(num_classes=inferred, pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Any,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run detection + calibration on a single image (RGB or HWC array)."""
        if isinstance(image, np.ndarray):
            from PIL import Image as PILImage

            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                # Float arrays are expected in [0, 1] (e.g. dataset tensors);
                # anything above 1 is treated as an already-scaled 0-255 array.
                scale = 1.0 if arr.max() > 1.0 + 1e-6 else 255.0
                arr = np.round(arr * scale).astype(np.uint8)
            image = PILImage.fromarray(arr).convert("RGB")

        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, torch.Tensor):
            pil_image = PILImage.fromarray(
                np.round(image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)!r} "
                "(expected PIL.Image, numpy.ndarray, or torch.Tensor)"
            )

        orig_size = pil_image.size
        tensor, _ = self.resize(pil_image, np.zeros((0, 4), dtype=np.float32))
        tensor = tensor.to(self.device)

        output = self.model([tensor])[0]
        thr = score_threshold if score_threshold is not None else self.confidence_threshold
        scores_all = output["scores"].cpu().numpy()
        labels_all = output["labels"].cpu().numpy()

        # Signal: accident-class scores above the calibration floor.
        keep_signal = scores_all >= SIGNAL_SCORE_FLOOR
        from .labels import DATASET_LABEL_ACCIDENT, model_to_dataset_labels

        dataset_labels_signal = model_to_dataset_labels(labels_all[keep_signal])
        accident_scores = scores_all[keep_signal][
            dataset_labels_signal == DATASET_LABEL_ACCIDENT
        ]
        signal_fn = SIGNAL_FUNCTIONS[self.signal_name]
        signal = float(signal_fn(np.asarray(accident_scores, dtype=np.float64)))
        probability = float(self.calibrator.transform(np.asarray([signal]))[0])

        # Returned detections: filtered at the confidence threshold.
        keep_det = scores_all >= thr
        from .transforms import clamp_boxes

        boxes_xyxy = clamp_boxes(
            output["boxes"][keep_det].cpu().numpy(),
            self.image_size[0],
            self.image_size[1],
        )
        dataset_labels = model_to_dataset_labels(labels_all[keep_det])
        scores = scores_all[keep_det]

        from . import CLASS_NAMES

        detections = [
            {
                "bbox": [round(float(v), 2) for v in box],
                "category": CLASS_NAMES.get(int(lbl), str(lbl)),
                "confidence": round(float(score), 4),
            }
            for box, lbl, score in zip(boxes_xyxy, dataset_labels, scores)
        ]

        return {
            "detector_version": "accident_calibrated_detector_v1",
            "image_size": [orig_size[0], orig_size[1]],
            "detections": detections,
            "accident_detection_confidence": round(signal, 4),
            "calibrated_accident_probability": round(probability, 4),
            "calibration": {
                "method": self.calibration_dict.get("method"),
                "signal": self.signal_name,
                "fit_split": self.metadata.get("fit_split", "validation"),
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.predict(*args, **kwargs)