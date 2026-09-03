"""Confidence signal derivation and probability calibration (Phase 6).

Two clearly separated quantities:

- **Detector confidence** (`accident_detection_confidence`): the raw
  image/event-level accident signal aggregated from accident-class box scores
  of the Faster R-CNN detector. It is a ranking score, NOT a probability.
- **Calibrated accident probability** (`calibrated_accident_probability`):
  the signal mapped through a calibrator fitted on VALIDATION data only,
  interpretable as P(the image contains an accident).

Calibration is fit exclusively on the validation split. TEST is evaluated
exactly once after the calibrator is frozen.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SIGNAL_EPS = 1e-6
NOISY_OR_TOP_K = 5
TOPK_MEAN_K = 3
ECE_BINS = 10

# Single source of truth for the detector score floor used when aggregating
# the image-level signal — both the calibration fitting pipeline and the
# deployed inference path must aggregate over the same score distribution.
SIGNAL_SCORE_FLOOR = 0.05

SignalFn = Callable[[np.ndarray], float]

# Fitted calibrators are serialized as a small plain dict so the checkpoint
# can be loaded with torch.load(weights_only=True).
CALIBRATION_METHODS = ("platt", "isotonic")


def signal_max(accident_scores: np.ndarray) -> float:
    """Maximum accident-class box score (0.0 when no accident detection)."""
    return float(accident_scores.max()) if len(accident_scores) else 0.0


def signal_topk_mean(accident_scores: np.ndarray, k: int = TOPK_MEAN_K) -> float:
    """Mean of the top-k accident scores, a softer event-level aggregate."""
    if len(accident_scores) == 0:
        return 0.0
    top = np.sort(accident_scores)[::-1][:k]
    return float(top.mean())


def signal_noisy_or(accident_scores: np.ndarray, k: int = NOISY_OR_TOP_K) -> float:
    """Noisy-OR over the top-k accident scores: 1 - prod(1 - s_i)."""
    if len(accident_scores) == 0:
        return 0.0
    top = np.sort(accident_scores)[::-1][:k]
    return float(1.0 - np.prod(1.0 - top))


SIGNAL_FUNCTIONS: dict[str, SignalFn] = {
    "max": signal_max,
    "top3_mean": signal_topk_mean,
    "noisy_or_top5": signal_noisy_or,
}


def image_level_signal(
    pred_records: list[dict[str, Any]],
    signal_name: str = "max",
    class_id: int = 0,
) -> np.ndarray:
    """Aggregate per-image accident-class box scores into event-level signals.

    Args:
        pred_records: prediction records (boxes, labels in dataset space,
            scores) as produced by `train.gather_predictions`.
        signal_name: one of `SIGNAL_FUNCTIONS`.
        class_id: dataset class id treated as the positive class (0 = accident).

    Returns:
        Array of shape (num_images,) with one signal per prediction record.
    """
    fn = SIGNAL_FUNCTIONS[signal_name]
    values = []
    for rec in pred_records:
        labels = np.asarray(rec["labels"])
        scores = np.asarray(rec["scores"], dtype=np.float64)
        accident_scores = scores[labels == class_id]
        values.append(fn(accident_scores))
    return np.asarray(values, dtype=np.float64)


def image_level_labels(gt_records: list[dict[str, Any]], class_id: int = 0) -> np.ndarray:
    """Image-level positive label: 1 if the image has >= 1 GT box of the class."""
    labels = []
    for rec in gt_records:
        gt_labels = np.asarray(rec["labels"])
        labels.append(1 if (gt_labels == class_id).any() else 0)
    return np.asarray(labels, dtype=np.int64)


def _logit(scores: np.ndarray) -> np.ndarray:
    """Logit with clipping so 0/1 scores stay finite."""
    clipped = np.clip(scores, SIGNAL_EPS, 1.0 - SIGNAL_EPS)
    return np.log(clipped / (1.0 - clipped))


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the raw signal.

    Fit on validation image-level (signal, label) pairs only.
    """

    method = "platt"

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def fit(cls, signals: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        x = _logit(np.asarray(signals, dtype=np.float64)).reshape(-1, 1)
        y = np.asarray(labels, dtype=np.int64)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        return cls(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))

    def transform(self, signals: np.ndarray) -> np.ndarray:
        x = _logit(np.asarray(signals, dtype=np.float64))
        # Numerically stable sigmoid: exp only ever sees non-positive args.
        z = self.a * x + self.b
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        out[~positive] = ez / (1.0 + ez)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlattCalibrator":
        return cls(a=d["a"], b=d["b"])


class IsotonicCalibrator:
    """Isotonic regression (PAVA) fitted on validation image-level pairs."""

    method = "isotonic"

    def __init__(self, x_thresholds: np.ndarray, y_thresholds: np.ndarray):
        self.x_thresholds = np.asarray(x_thresholds, dtype=np.float64)
        self.y_thresholds = np.asarray(y_thresholds, dtype=np.float64)

    @classmethod
    def fit(cls, signals: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(np.asarray(signals, dtype=np.float64), np.asarray(labels, dtype=np.float64))
        # sklearn >= 1.0 names the fitted knots X_thresholds_.
        x_thresholds = getattr(ir, "X_thresholds_", None)
        if x_thresholds is None:
            x_thresholds = getattr(ir, "x_thresholds_", None)
        if x_thresholds is None:
            raise AttributeError(
                "IsotonicRegression exposes no threshold attribute "
                "(expected X_thresholds_); sklearn version unsupported."
            )
        return cls(
            x_thresholds=np.asarray(x_thresholds, dtype=np.float64),
            y_thresholds=np.asarray(ir.y_thresholds_, dtype=np.float64),
        )

    def transform(self, signals: np.ndarray) -> np.ndarray:
        x = np.asarray(signals, dtype=np.float64)
        # np.interp needs an increasing x grid; isotonic thresholds are.
        return np.interp(x, self.x_thresholds, self.y_thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "x_thresholds": [float(v) for v in self.x_thresholds],
            "y_thresholds": [float(v) for v in self.y_thresholds],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IsotonicCalibrator":
        return cls(
            x_thresholds=np.asarray(d["x_thresholds"], dtype=np.float64),
            y_thresholds=np.asarray(d["y_thresholds"], dtype=np.float64),
        )


def build_calibrator(method: str) -> type:
    """Return the calibrator class for a method name."""
    if method == "platt":
        return PlattCalibrator
    if method == "isotonic":
        return IsotonicCalibrator
    raise ValueError(f"Unknown calibration method: {method!r}")


def calibrator_from_dict(d: dict[str, Any]) -> PlattCalibrator | IsotonicCalibrator:
    """Rebuild a fitted calibrator from its serialized dict."""
    cls = build_calibrator(d["method"])
    return cls.from_dict(d)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between probabilities and binary outcomes."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss(labels: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy with clipping for numerical safety."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> list[dict[str, Any]]:
    """Unrounded equal-width reliability bins over [0, 1]."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is closed on the right so p == 1.0 is included.
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(in_bin.sum())
        bins.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": n,
                "mean_predicted": float(p[in_bin].mean()) if n else 0.0,
                "observed_frequency": float(y[in_bin].mean()) if n else 0.0,
            }
        )
    return bins


def reliability_table(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = ECE_BINS
) -> list[dict[str, Any]]:
    """Equal-width reliability bins over [0, 1], rounded for display."""
    rows = []
    for b in _reliability_bins(labels, probabilities, n_bins):
        rows.append(
            {
                "bin_low": round(b["bin_low"], 2),
                "bin_high": round(b["bin_high"], 2),
                "count": b["count"],
                "mean_predicted": round(b["mean_predicted"], 4) if b["count"] else None,
                "observed_frequency": round(b["observed_frequency"], 4) if b["count"] else None,
            }
        )
    return rows


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """ECE = sum over bins of (n_b / N) * |mean_predicted_b - observed_b|.

    Computed from unrounded bin aggregates; rounding happens only at report
    time.
    """
    bins = _reliability_bins(labels, probabilities, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = sum(
        b["count"] * abs(b["mean_predicted"] - b["observed_frequency"]) for b in bins
    )
    return float(ece / total)


def cross_validated_metrics(
    signals: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """K-fold CV on the calibration split to assess method stability.

    Each fold fits the calibrator on the other folds and evaluates on the
    held-out fold. Used only for method comparison on validation.
    """
    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(signals, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    cls = build_calibrator(method)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    briers, losses = [], []
    for train_idx, test_idx in skf.split(x, y):
        cal = cls.fit(x[train_idx], y[train_idx])
        p = cal.transform(x[test_idx])
        briers.append(brier_score(y[test_idx], p))
        losses.append(log_loss(y[test_idx], p))
    return {
        "cv_brier_mean": float(np.mean(briers)),
        "cv_brier_std": float(np.std(briers)),
        "cv_log_loss_mean": float(np.mean(losses)),
        "cv_log_loss_std": float(np.std(losses)),
        "n_folds": n_splits,
    }