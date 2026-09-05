"""Phase 7 — Safety Risk Fusion orchestrator.

Connects the completed probability models without retraining them:

- Phase 4 near-miss/conflict model
  (models/near_miss/random_forest_D_calibrated.joblib)
  via the existing near_miss.probability_pipeline.ProbabilityPipeline
  -> calibrated class-1 probability.

- Phase 6 calibrated accident detector
  (models/accident/calibrated_accident_detector.pth)
  via accident.calibrated_detector.CalibratedAccidentDetector
  -> calibrated accident probability.

Model D is the deployment-realistic near-miss model. It does NOT require:
- PET
- FE_inv_PET
- FE_safety_index

The near-miss probability and accident probability remain separate
and are passed to risk_engine.assess_risk together with caller-supplied
severity and event age.

This module:
- does NOT retrain models
- does NOT control traffic signals
- does NOT treat the risk score as a probability
- does NOT substitute raw accident detector confidence for probability
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

try:
    # When src/ is directly on PYTHONPATH
    import config as _config
    from risk_engine import assess_risk
except ImportError:
    # Package-style import
    from . import config as _config
    from .risk_engine import assess_risk


# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------

ENGINE_VERSION = "safety_risk_fusion_v1"


# ---------------------------------------------------------------------
# Phase 6 detector output keys
# ---------------------------------------------------------------------

_DETECTOR_PROBABILITY_KEY = "calibrated_accident_probability"
_DETECTOR_RAW_SIGNAL_KEY = "accident_detection_confidence"


# ---------------------------------------------------------------------
# Expected Model D features
# ---------------------------------------------------------------------

MODEL_D_FEATURES = [
    "speed_object_1_kph",
    "speed_object_2_kph",
    "angle_degrees",
    "acceleration_obj1_mps2",
    "acceleration_obj2_mps2",
    "min_dist_dual_check_m",
    "object_count_at_frame1",
    "target_dist_px",
    "FE_log_mesafe",
    "FE_dist_squared",
    "class_object_1",
    "class_object_2",
]


class SafetyIntelligence:
    """Lazy-loading fusion of near-miss Model D and accident models."""

    def __init__(
        self,
        near_miss_artifact: str | None = None,
        accident_checkpoint: str | None = None,
    ):
        """
        Parameters
        ----------
        near_miss_artifact:
            Optional explicit path to the calibrated near-miss model.

        accident_checkpoint:
            Optional explicit path to the calibrated accident detector.

        If paths are not supplied, values are taken from config.py.
        """

        self.near_miss_artifact = near_miss_artifact
        self.accident_checkpoint = accident_checkpoint

        # Models are loaded lazily.
        self._near_miss = None
        self._accident = None

    # =================================================================
    # MODEL LOADING
    # =================================================================

    def _ensure_near_miss(self):
        """Load the calibrated Model D artifact once."""

        if self._near_miss is None:

            from near_miss.probability_pipeline import ProbabilityPipeline

            path = (
                self.near_miss_artifact
                or _config.NEAR_MISS_MODEL_PATH
            )

            self._near_miss = ProbabilityPipeline(path)

            # Safety check:
            # Make sure the configured artifact is actually Model D.
            actual_features = list(self._near_miss.features)

            if actual_features != MODEL_D_FEATURES:
                raise ValueError(
                    "The configured near-miss artifact does not match "
                    "the deployment-realistic Model D feature schema.\n\n"
                    f"Expected {len(MODEL_D_FEATURES)} features:\n"
                    f"{MODEL_D_FEATURES}\n\n"
                    f"Loaded {len(actual_features)} features:\n"
                    f"{actual_features}\n\n"
                    "Make sure config.py points to:\n"
                    "models/near_miss/random_forest_D_calibrated.joblib"
                )

        return self._near_miss

    def _ensure_accident(self):
        """Load the calibrated accident detector once."""

        if self._accident is None:

            from accident.calibrated_detector import (
                CalibratedAccidentDetector
            )

            path = (
                self.accident_checkpoint
                or _config.ACCIDENT_DETECTOR_PATH
            )

            self._accident = CalibratedAccidentDetector(path)

        return self._accident

    # =================================================================
    # MODEL STATUS
    # =================================================================

    @property
    def near_miss_model_loaded(self) -> bool:
        """Return True if the near-miss model has been loaded."""

        return self._near_miss is not None

    @property
    def accident_model_loaded(self) -> bool:
        """Return True if the accident model has been loaded."""

        return self._accident is not None

    # =================================================================
    # NEAR-MISS PROBABILITY
    # =================================================================

    def _near_miss_probability(
        self,
        features: pd.DataFrame | dict[str, Any] | None,
    ) -> float | None:
        """
        Calculate calibrated Model D class-1 probability.

        Parameters
        ----------
        features:
            Either:

            - pandas DataFrame containing Model D features
            - dictionary containing one Model D feature vector
            - None

        Returns
        -------
        float | None
            Calibrated class-1 probability.
        """

        if features is None:
            return None

        pipe = self._ensure_near_miss()

        # Convert dictionary into one-row DataFrame.
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            raise TypeError(
                "near_miss_features must be a pandas DataFrame, "
                "dictionary, or None."
            )

        # -------------------------------------------------------------
        # Validate required Model D features
        # -------------------------------------------------------------

        missing = [
            feature
            for feature in pipe.features
            if feature not in frame.columns
        ]

        if missing:
            raise ValueError(
                "Input missing required Model D columns: "
                f"{missing}"
            )

        # -------------------------------------------------------------
        # Select ONLY model features
        # -------------------------------------------------------------

        model_frame = frame[pipe.features]

        # -------------------------------------------------------------
        # Run calibrated probability model
        # -------------------------------------------------------------

        probabilities = pipe.predict_proba(model_frame)

        probability = float(probabilities[0, 1])

        # Probability should always be in [0, 1].
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Near-miss model returned invalid probability: "
                f"{probability}"
            )

        return round(probability, 6)

    # =================================================================
    # ACCIDENT PROBABILITY
    # =================================================================

    def _accident_probability(
        self,
        accident_detector_output: dict[str, Any] | None,
        accident_probability: float | None,
    ) -> tuple[float | None, Any]:
        """
        Resolve calibrated accident probability.

        Explicit accident_probability has priority.

        If detector output is supplied, only:
            calibrated_accident_probability

        is treated as a probability.

        The raw:
            accident_detection_confidence

        value is preserved separately and NEVER treated as a probability.
        """

        raw_signal = None

        # -------------------------------------------------------------
        # Explicit probability supplied by caller
        # -------------------------------------------------------------

        if accident_probability is not None:

            resolved = float(accident_probability)

        # -------------------------------------------------------------
        # Probability supplied through detector output
        # -------------------------------------------------------------

        elif accident_detector_output is not None:

            if _DETECTOR_PROBABILITY_KEY not in accident_detector_output:

                raise ValueError(
                    "accident_detector_output is missing "
                    f"{_DETECTOR_PROBABILITY_KEY!r}.\n"
                    "The raw detector confidence is NOT a probability "
                    "and cannot be substituted."
                )

            resolved = float(
                accident_detector_output[
                    _DETECTOR_PROBABILITY_KEY
                ]
            )

            raw_signal = accident_detector_output.get(
                _DETECTOR_RAW_SIGNAL_KEY
            )

        # -------------------------------------------------------------
        # No accident evidence supplied
        # -------------------------------------------------------------

        else:

            resolved = None

        # -------------------------------------------------------------
        # Validate probability
        # -------------------------------------------------------------

        if resolved is not None:

            if not 0.0 <= resolved <= 1.0:
                raise ValueError(
                    "Accident probability must be between 0 and 1. "
                    f"Received: {resolved}"
                )

        return resolved, raw_signal

    # =================================================================
    # PUBLIC EVENT API
    # =================================================================

    def assess_event(
        self,
        near_miss_features: (
            pd.DataFrame
            | dict[str, Any]
            | None
        ) = None,
        accident_detector_output: dict[str, Any] | None = None,
        accident_probability: float | None = None,
        severity: float | None = None,
        event_age_hours: float | None = None,
        risk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fuse near-miss and accident evidence into a safety assessment.

        Parameters
        ----------
        near_miss_features:
            Model D feature vector.

        accident_detector_output:
            Output dictionary from Phase 6 accident detector.

        accident_probability:
            Explicit calibrated accident probability.

        severity:
            Caller-supplied event severity in [0, 1].

        event_age_hours:
            Age of the event in hours.

        risk_config:
            Optional risk-engine configuration.

        Returns
        -------
        dict
            JSON-serializable safety assessment.
        """

        # -------------------------------------------------------------
        # Near-miss model
        # -------------------------------------------------------------

        near_miss_probability = self._near_miss_probability(
            near_miss_features
        )

        # -------------------------------------------------------------
        # Accident model
        # -------------------------------------------------------------

        resolved_accident, raw_signal = self._accident_probability(
            accident_detector_output,
            accident_probability,
        )

        # -------------------------------------------------------------
        # Risk fusion
        # -------------------------------------------------------------

        assessment = assess_risk(
            near_miss_probability=near_miss_probability,
            accident_probability=resolved_accident,
            severity=severity,
            event_age_hours=event_age_hours,
            config=risk_config,
        )

        # -------------------------------------------------------------
        # Build final output
        # -------------------------------------------------------------

        output: dict[str, Any] = {

            "engine_version": ENGINE_VERSION,

            "near_miss_probability": (
                assessment["near_miss_probability"]
            ),

            "accident_probability": (
                assessment["accident_probability"]
            ),

            "risk_score": assessment["risk_score"],

            "risk_level": assessment["risk_level"],

            "severity": assessment["severity"],

            "event_age_hours": assessment["event_age_hours"],

            "explanation": list(
                assessment["explanation"]
            ),

            "present_inputs": assessment["present_inputs"],

            "missing_inputs": assessment["missing_inputs"],

            "model_provenance": {

                "near_miss_artifact": str(
                    self.near_miss_artifact
                    or _config.NEAR_MISS_MODEL_PATH
                ),

                "near_miss_model_loaded": (
                    self.near_miss_model_loaded
                ),

                "near_miss_feature_configuration": (
                    "D_deployment_realistic"
                ),

                "near_miss_features": MODEL_D_FEATURES,

                "accident_checkpoint": str(
                    self.accident_checkpoint
                    or _config.ACCIDENT_DETECTOR_PATH
                ),

                "accident_model_loaded": (
                    self.accident_model_loaded
                ),

                "retrained": False,
            },
        }

        # -------------------------------------------------------------
        # Preserve raw accident detector confidence separately
        # -------------------------------------------------------------

        if raw_signal is not None:

            output["detector_raw_signal"] = {

                "accident_detection_confidence": raw_signal,

                "note": (
                    "Raw detector confidence is a ranking signal "
                    "aggregated from accident-class box scores. "
                    "It is NOT a probability. The accident probability "
                    "is the calibrated 'accident_probability' value."
                ),
            }

        return output

    # =================================================================
    # IMAGE-BASED ACCIDENT + SAFETY ASSESSMENT
    # =================================================================

    def assess_image(
        self,
        image: Any,
        near_miss_features: (
            pd.DataFrame
            | dict[str, Any]
            | None
        ) = None,
        severity: float | None = None,
        event_age_hours: float | None = None,
        score_threshold: float | None = None,
        risk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the Phase 6 accident detector on an image and fuse
        the result with near-miss evidence.
        """

        # -------------------------------------------------------------
        # Load accident detector
        # -------------------------------------------------------------

        detector = self._ensure_accident()

        # -------------------------------------------------------------
        # Run detector
        # -------------------------------------------------------------

        detector_output = detector.predict(
            image,
            score_threshold=score_threshold,
        )

        # -------------------------------------------------------------
        # Fuse safety information
        # -------------------------------------------------------------

        output = self.assess_event(
            near_miss_features=near_miss_features,
            accident_detector_output=detector_output,
            severity=severity,
            event_age_hours=event_age_hours,
            risk_config=risk_config,
        )

        # -------------------------------------------------------------
        # Add detector information
        # -------------------------------------------------------------

        output["detector_output"] = {

            "detector_version": detector_output.get(
                "detector_version"
            ),

            "image_size": detector_output.get(
                "image_size"
            ),

            "n_detections": len(
                detector_output.get(
                    "detections",
                    []
                )
            ),

            "detections": detector_output.get(
                "detections",
                []
            ),

            "calibration": detector_output.get(
                "calibration"
            ),
        }

        return output


# =====================================================================
# JSON SERIALIZATION
# =====================================================================

def to_json(
    output: dict[str, Any],
    indent: int | None = 2,
) -> str:
    """
    Convert safety assessment into JSON.
    """

    return json.dumps(
        output,
        indent=indent,
    )