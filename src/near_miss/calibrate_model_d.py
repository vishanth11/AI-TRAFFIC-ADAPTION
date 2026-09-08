import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from near_miss.probability_pipeline import CalibratedProbabilityPipeline


ROOT = Path(".")
SPLIT_DIR = ROOT / "data/processed/near_miss/splits"
MODEL_DIR = ROOT / "models/near_miss"

MODEL_NAME = "random_forest_D"
TARGET = "event_type"


def load_splits():
    return {
        split: pd.read_parquet(SPLIT_DIR / f"{split}.parquet")
        for split in ["train", "validation", "test"]
    }


def feature_cols_from_pipe(pipe):
    return list(pipe.named_steps["preprocess"].feature_names_in_)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        count = int(mask.sum())

        if count == 0:
            continue

        accuracy = y_true[mask].mean()
        confidence = y_prob[mask].mean()

        error = abs(accuracy - confidence)

        ece += count * error
        mce = max(mce, error)

    ece /= len(y_true)

    return {
        "ece": float(ece),
        "mce": float(mce)
    }


def score(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)

    return (
        brier_score_loss(y, p)
        + log_loss(y, p)
    )


def main():

    print("=" * 60)
    print("Model D Probability Calibration")
    print("=" * 60)

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"

    print("\nLoading model:")
    print(model_path)

    raw_pipe = joblib.load(model_path)

    features = feature_cols_from_pipe(raw_pipe)

    print("\nFeatures:")
    for feature in features:
        print(" -", feature)

    print("\nLoading dataset splits...")

    parts = load_splits()

    X_val = parts["validation"][features]
    y_val = parts["validation"][TARGET].values

    X_test = parts["test"][features]
    y_test = parts["test"][TARGET].values

    print("\nValidation samples:", len(X_val))
    print("Test samples:", len(X_test))

    # ---------------------------------------------------------
    # Raw probabilities
    # ---------------------------------------------------------

    print("\nGenerating raw probabilities...")

    raw_val = raw_pipe.predict_proba(X_val)[:, 1]
    raw_test = raw_pipe.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # Sigmoid calibration
    # ---------------------------------------------------------

    print("\nFitting sigmoid calibration...")

    sigmoid_cal = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    )

    sigmoid_cal.fit(
        raw_val.reshape(-1, 1),
        y_val
    )

    sigmoid_val = sigmoid_cal.predict_proba(
        raw_val.reshape(-1, 1)
    )[:, 1]

    sigmoid_test = sigmoid_cal.predict_proba(
        raw_test.reshape(-1, 1)
    )[:, 1]

    # ---------------------------------------------------------
    # Isotonic calibration
    # ---------------------------------------------------------

    print("Fitting isotonic calibration...")

    isotonic_cal = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0
    )

    isotonic_cal.fit(
        raw_val,
        y_val
    )

    isotonic_val = isotonic_cal.predict(raw_val)
    isotonic_test = isotonic_cal.predict(raw_test)

    # ---------------------------------------------------------
    # Validation comparison
    # ---------------------------------------------------------

    raw_score = score(y_val, raw_val)
    sigmoid_score = score(y_val, sigmoid_val)
    isotonic_score = score(y_val, isotonic_val)

    print("\nValidation scores")
    print("-" * 40)

    print(f"Raw:      {raw_score:.6f}")
    print(f"Sigmoid:  {sigmoid_score:.6f}")
    print(f"Isotonic: {isotonic_score:.6f}")

    sigmoid_ece = expected_calibration_error(
        y_val,
        sigmoid_val
    )

    isotonic_ece = expected_calibration_error(
        y_val,
        isotonic_val
    )

    print("\nValidation calibration error")
    print("-" * 40)

    print(
        f"Sigmoid  ECE: {sigmoid_ece['ece']:.6f} "
        f"MCE: {sigmoid_ece['mce']:.6f}"
    )

    print(
        f"Isotonic ECE: {isotonic_ece['ece']:.6f} "
        f"MCE: {isotonic_ece['mce']:.6f}"
    )

    # ---------------------------------------------------------
    # Select calibration method
    # ---------------------------------------------------------

    if (
        isotonic_score < sigmoid_score
        and isotonic_ece["mce"] < 0.01
    ):
        best_method = "isotonic"
        calibrator = isotonic_cal
        selected_val = isotonic_val
        selected_test = isotonic_test

    else:
        best_method = "sigmoid"
        calibrator = sigmoid_cal
        selected_val = sigmoid_val
        selected_test = sigmoid_test

    print("\nSelected calibration method:")
    print(best_method)

    # ---------------------------------------------------------
    # Build calibrated pipeline
    # ---------------------------------------------------------

    calibrated_pipeline = CalibratedProbabilityPipeline(
        raw_pipe,
        calibrator,
        best_method
    )

    output_path = MODEL_DIR / "random_forest_D_calibrated.joblib"

    joblib.dump(
        calibrated_pipeline,
        output_path
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "original_model": MODEL_NAME,
        "original_model_path": str(model_path),
        "calibration_method": best_method,
        "feature_configuration": "D_deployment_realistic",
        "features": features,
        "calibration_training_split": "validation",
        "calibration_training_n_samples": int(len(X_val)),
        "calibration_artifact": str(output_path),
        "validation_scores": {
            "raw": float(raw_score),
            "sigmoid": float(sigmoid_score),
            "isotonic": float(isotonic_score)
        },
        "validation_ece": {
            "sigmoid": sigmoid_ece,
            "isotonic": isotonic_ece
        }
    }

    metadata_path = MODEL_DIR / "random_forest_D_calibrated_meta.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    print("\nTest evaluation")
    print("-" * 40)

    print(
        f"Raw test score:      "
        f"{score(y_test, raw_test):.6f}"
    )

    print(
        f"Selected calibrated: "
        f"{score(y_test, selected_test):.6f}"
    )

    test_ece = expected_calibration_error(
        y_test,
        selected_test
    )

    print(
        f"Selected test ECE:   "
        f"{test_ece['ece']:.6f}"
    )

    print("\nArtifacts created:")

    print(output_path)
    print(metadata_path)

    print("\nCalibration completed successfully.")

    # ---------------------------------------------------------
    # Probability sanity check
    # ---------------------------------------------------------

    print("\nProbability sanity check:")

    test_pipeline = joblib.load(output_path)

    sample = X_test.iloc[:5]

    probabilities = test_pipeline.predict_proba(sample)[:, 1]

    for i, probability in enumerate(probabilities):
        print(
            f"Sample {i}: "
            f"{probability:.6f}"
        )


if __name__ == "__main__":
    main()