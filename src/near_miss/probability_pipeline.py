"""Reusable probability pipeline for the calibrated near-miss/conflict model.

This module provides a thin wrapper around the saved calibrated pipeline so that
later phases (risk engine) can consume calibrated probabilities without needing
to know calibration internals.

Input: a DTC-FM feature vector matching the model's feature schema.
Output: raw and calibrated class probabilities plus model metadata.

No risk levels, traffic signal commands, or Member 5 integration here.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_ARTIFACT = Path("models/near_miss/random_forest_A_calibrated.joblib")


class CalibratedProbabilityPipeline:
    """Wrapper that applies raw model + univariate calibration.

    This avoids relying on `CalibratedClassifierCV(cv='prefit')`, which is not
    supported by older scikit-learn versions. The wrapper is serialisable with
    joblib and exposes the same `predict_proba` interface.
    """

    def __init__(self, raw_pipeline, calibrator, method: str):
        self.raw_pipeline = raw_pipeline
        self.calibrator = calibrator
        self.method = method
        self.feature_names_in_ = list(raw_pipeline.named_steps["preprocess"].feature_names_in_)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw_p = self.raw_pipeline.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            cal_p = self.calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        else:
            cal_p = self.calibrator.predict(raw_p)
        cal_p = np.clip(cal_p, 0.0, 1.0)
        return np.column_stack([1 - cal_p, cal_p])


class ProbabilityPipeline:
    """Load and run the calibrated near-miss/conflict probability pipeline."""

    def __init__(self, artifact_path: str | Path | None = None):
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT
        self._pipe = joblib.load(self.artifact_path)
        self._features = list(self._pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
        self._model_version = self.artifact_path.stem

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def model_version(self) -> str:
        return self._model_version

    def validate_input(self, X: pd.DataFrame) -> None:
        missing = [c for c in self._features if c not in X.columns]
        if missing:
            raise ValueError(f"Input missing required columns: {missing}")
        forbidden = [c for c in X.columns if c in ("sample_index", "event_type", "split")]
        if forbidden:
            raise ValueError(f"Input contains forbidden columns: {forbidden}")

    def predict(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """Return probability records for each input row."""
        self.validate_input(X)
        probs = self._pipe.predict_proba(X[self._features])
        return [
            {
                "class_0_probability": round(float(p[0]), 6),
                "class_1_probability": round(float(p[1]), 6),
                "model_version": self._model_version,
                "calibrated": True,
            }
            for p in probs
        ]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the calibrated probability matrix (n_samples, 2)."""
        self.validate_input(X)
        return self._pipe.predict_proba(X[self._features])


def infer_probability(feature_vector: dict[str, Any],
                      artifact_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience function for a single feature vector."""
    pipe = ProbabilityPipeline(artifact_path)
    df = pd.DataFrame([feature_vector])
    return pipe.predict(df)[0]
