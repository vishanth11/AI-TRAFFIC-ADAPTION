"""Phase 7 — Safety Risk Fusion orchestrator.

Connects the two COMPLETED probability models without retraining them:

- Phase 4 near-miss/conflict model
  (``models/near_miss/random_forest_A_calibrated.joblib``) via the existing
  ``near_miss.probability_pipeline.ProbabilityPipeline`` -> calibrated
  class-1 probability.
- Phase 6 calibrated accident detector
  (``models/accident/calibrated_accident_detector.pth``) via
  ``accident.calibrated_detector.CalibratedAccidentDetector`` ->
  ``calibrated_accident_probability``.

Both probabilities are kept semantically separate and handed to
``risk_engine.assess_risk`` together with CALLER-SUPPLIED severity and event
age. The raw detector confidence (``accident_detection_confidence``) is a
ranking signal only — it is reported under a dedicated non-probability key
and never used as a probability.

This module does NOT control traffic signals, does NOT claim the risk score
is scientifically validated, and does NOT invent accident/near-miss label
semantics (the near-miss class-1 semantics remain unconfirmed per the data
audit; the value is echoed from the Phase 4 artifact as-is).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

try:  # src/ root import
    import config as _config
    from risk_engine import assess_risk
except ImportError:  # package-style import
    from . import config as _config
    from .risk_engine import assess_risk

ENGINE_VERSION = "safety_risk_fusion_v1"

#: Keys copied from the Phase 6 detector output into the fused assessment.
_DETECTOR_PROBABILITY_KEY = "calibrated_accident_probability"
_DETECTOR_RAW_SIGNAL_KEY = "accident_detection_confidence"


class SafetyIntelligence:
    """Lazy-loading fusion of the Phase 4 near-miss and Phase 6 accident models."""

    def __init__(
        self,
        near_miss_artifact: str | None = None,
        accident_checkpoint: str | None = None,
    ):
        self.near_miss_artifact = near_miss_artifact
        self.accident_checkpoint = accident_checkpoint
        self._near_miss = None  # ProbabilityPipeline, loaded on first use
        self._accident = None  # CalibratedAccidentDetector, loaded on first use

    # -- lazy model loading (no retraining; artifacts are loaded read-only) --

    def _ensure_near_miss(self):
        if self._near_miss is None:
            from near_miss.probability_pipeline import ProbabilityPipeline

            path = self.near_miss_artifact or _config.NEAR_MISS_MODEL_PATH
            self._near_miss = ProbabilityPipeline(path)
        return self._near_miss

    def _ensure_accident(self):
        if self._accident is None:
            from accident.calibrated_detector import CalibratedAccidentDetector

            path = self.accident_checkpoint or _config.ACCIDENT_DETECTOR_PATH
            self._accident = CalibratedAccidentDetector(path)
        return self._accident

    @property
    def near_miss_model_loaded(self) -> bool:
        return self._near_miss is not None

    @property
    def accident_model_loaded(self) -> bool:
        return self._accident is not None

    # -- input handling --

    def _near_miss_probability(self, features: pd.DataFrame | dict[str, Any] | None) -> float | None:
        """Calibrated near-miss class-1 probability, or None when not provided."""
        if features is None:
            return None
        pipe = self._ensure_near_miss()
        frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])
        # Pass only the model's feature columns: raw records (e.g. split rows)
        # carry passthrough columns like sample_index/event_type/split, which
        # the pipeline forbids. Missing feature columns raise a clear error.
        missing = [col for col in pipe.features if col not in frame.columns]
        if missing:
            raise ValueError(f"Input missing required columns: {missing}")
        proba = pipe.predict_proba(frame[pipe.features])
        return round(float(proba[0, 1]), 6)

    def _accident_probability(
        self,
        accident_detector_output: dict[str, Any] | None,
        accident_probability: float | None,
    ) -> tuple[float | None, dict[str, Any]]:
        """Resolve the accident probability from an explicit value or a
        Phase 6 detector output dict.

        Only the CALIBRATED probability is accepted as a probability. When a
        detector output is supplied, its raw ``accident_detection_confidence``
        is carried separately under ``detector_raw_signal`` with an explicit
        not-a-probability note.
        """
        raw_signal = None
        if accident_probability is not None:
            resolved = accident_probability
        elif accident_detector_output is not None:
            if _DETECTOR_PROBABILITY_KEY not in accident_detector_output:
                raise ValueError(
                    f"accident_detector_output is missing {_DETECTOR_PROBABILITY_KEY!r}; "
                    "the raw confidence is not a probability and is never substituted."
                )
            resolved = accident_detector_output[_DETECTOR_PROBABILITY_KEY]
            raw_signal = accident_detector_output.get(_DETECTOR_RAW_SIGNAL_KEY)
        else:
            resolved = None
        return resolved, raw_signal

    # -- public API --

    def assess_event(
        self,
        near_miss_features: pd.DataFrame | dict[str, Any] | None = None,
        accident_detector_output: dict[str, Any] | None = None,
        accident_probability: float | None = None,
        severity: float | None = None,
        event_age_hours: float | None = None,
        risk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fuse one event's evidence into a JSON-serializable risk assessment.

        Either pass a Phase 6 detector output dict (``accident_detector_output``)
        or an already-calibrated ``accident_probability``; when both are given
        the explicit probability wins.
        """
        near_miss_probability = self._near_miss_probability(near_miss_features)
        resolved_accident, raw_signal = self._accident_probability(
            accident_detector_output, accident_probability
        )

        assessment = assess_risk(
            near_miss_probability=near_miss_probability,
            accident_probability=resolved_accident,
            severity=severity,
            event_age_hours=event_age_hours,
            config=risk_config,
        )

        output: dict[str, Any] = {
            "engine_version": ENGINE_VERSION,
            "near_miss_probability": assessment["near_miss_probability"],
            "accident_probability": assessment["accident_probability"],
            "risk_score": assessment["risk_score"],
            "risk_level": assessment["risk_level"],
            "severity": assessment["severity"],
            "event_age_hours": assessment["event_age_hours"],
            "explanation": list(assessment["explanation"]),
            "present_inputs": assessment["present_inputs"],
            "missing_inputs": assessment["missing_inputs"],
            "model_provenance": {
                "near_miss_artifact": str(
                    self.near_miss_artifact or _config.NEAR_MISS_MODEL_PATH
                ),
                "near_miss_model_loaded": self.near_miss_model_loaded,
                "accident_checkpoint": str(
                    self.accident_checkpoint or _config.ACCIDENT_DETECTOR_PATH
                ),
                "accident_model_loaded": self.accident_model_loaded,
                "retrained": False,
            },
        }
        if raw_signal is not None:
            output["detector_raw_signal"] = {
                "accident_detection_confidence": raw_signal,
                "note": (
                    "Raw detector confidence (ranking signal aggregated from "
                    "accident-class box scores). This is NOT a probability; the "
                    "probability is 'accident_probability' above (calibrated)."
                ),
            }
        return output

    def assess_image(
        self,
        image: Any,
        near_miss_features: pd.DataFrame | dict[str, Any] | None = None,
        severity: float | None = None,
        event_age_hours: float | None = None,
        score_threshold: float | None = None,
        risk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the Phase 6 detector on an image, then fuse as ``assess_event``.

        ``image`` may be a PIL image, numpy array, or torch tensor (the
        Phase 6 detector's contract). The near-miss side still needs its
        DTC-FM feature vector and remains optional.
        """
        detector = self._ensure_accident()
        detector_output = detector.predict(image, score_threshold=score_threshold)
        output = self.assess_event(
            near_miss_features=near_miss_features,
            accident_detector_output=detector_output,
            severity=severity,
            event_age_hours=event_age_hours,
            risk_config=risk_config,
        )
        output["detector_output"] = {
            "detector_version": detector_output.get("detector_version"),
            "image_size": detector_output.get("image_size"),
            "n_detections": len(detector_output.get("detections", [])),
            "detections": detector_output.get("detections", []),
            "calibration": detector_output.get("calibration"),
        }
        return output


def to_json(output: dict[str, Any], indent: int | None = 2) -> str:
    """Serialize an assessment; the outputs are plain builtins by construction."""
    return json.dumps(output, indent=indent)