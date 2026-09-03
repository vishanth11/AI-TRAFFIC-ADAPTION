"""Phase 4 tests — near-miss model calibration and probability pipeline.

Run: pytest tests/test_near_miss_phase4.py -v
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from near_miss.calibrate import (
    PRIMARY_RUN,
    compute_classification_metrics,
    load_splits,
    verify_split_integrity,
)
from near_miss.plotting import Probs
from near_miss.probability_pipeline import ProbabilityPipeline, infer_probability
from near_miss.split import FEATURE_COLS, make_duplicate_groups

DATA_NM = Path("data/processed/near_miss")
SPLIT_DIR = DATA_NM / "splits"
MODEL_DIR = Path("models/near_miss")
REPORT_DIR = Path("reports/near_miss")
TARGET = "event_type"


@pytest.fixture(scope="module")
def parts():
    return {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet")
            for s in ["train", "validation", "test"]}


@pytest.fixture(scope="module")
def calibrated_pipe():
    return joblib.load(MODEL_DIR / f"{PRIMARY_RUN}_calibrated.joblib")


# 1. Calibrated probabilities are between 0 and 1.
def test_calibrated_probabilities_in_range(parts, calibrated_pipe):
    feats = list(calibrated_pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    for split in ["validation", "test"]:
        probs = calibrated_pipe.predict_proba(parts[split][feats])
        assert probs.shape == (len(parts[split]), 2)
        assert (probs >= 0).all() and (probs <= 1).all()


# 2. Probability outputs sum approximately to 1.
def test_probability_outputs_sum_to_one(parts, calibrated_pipe):
    feats = list(calibrated_pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    for split in ["validation", "test"]:
        probs = calibrated_pipe.predict_proba(parts[split][feats])
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# 3. Calibration does not access test labels during fitting.
def test_calibration_fitted_on_validation_only():
    meta = json.loads((MODEL_DIR / f"{PRIMARY_RUN}_calibrated_meta.json").read_text())
    assert meta["calibration_training_split"] == "validation"
    assert meta["calibration_training_n_samples"] == len(pd.read_parquet(SPLIT_DIR / "validation.parquet"))


# 4. Test set remains untouched during calibration fitting.
def test_test_set_untouched_by_calibration(parts):
    meta = json.loads((MODEL_DIR / f"{PRIMARY_RUN}_calibrated_meta.json").read_text())
    assert meta["calibration_training_split"] == "validation"
    # Test split count should remain exactly 376 (Phase 3 output).
    assert len(parts["test"]) == 376


# 5. Feature schema matches training.
def test_calibrated_feature_schema_matches_training(parts, calibrated_pipe):
    train_pipe = joblib.load(MODEL_DIR / f"{PRIMARY_RUN}.joblib")
    train_feats = list(train_pipe.named_steps["preprocess"].feature_names_in_)
    cal_feats = list(calibrated_pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    assert cal_feats == train_feats


# 6. sample_index is never used as a feature.
def test_sample_index_not_feature(calibrated_pipe):
    feats = list(calibrated_pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    assert "sample_index" not in feats


# 7. target is never used as a feature.
def test_target_not_feature(calibrated_pipe):
    feats = list(calibrated_pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    assert TARGET not in feats
    assert "split" not in feats


# 8. Saved calibrated model can be loaded.
def test_calibrated_model_loadable():
    artifact = MODEL_DIR / f"{PRIMARY_RUN}_calibrated.joblib"
    assert artifact.exists()
    pipe = joblib.load(artifact)
    assert hasattr(pipe, "predict_proba")


# 9. Loaded model produces probabilities.
def test_loaded_model_produces_probabilities(parts):
    pipe = joblib.load(MODEL_DIR / f"{PRIMARY_RUN}_calibrated.joblib")
    feats = list(pipe.raw_pipeline.named_steps["preprocess"].feature_names_in_)
    probs = pipe.predict_proba(parts["test"][feats])
    assert probs.shape[0] == len(parts["test"])
    assert (probs >= 0).all() and (probs <= 1).all()


# 10. Original datasets remain untouched.
def test_original_datasets_untouched(parts):
    feats = pd.read_csv("dataset/filtered_features_DTC_FM.csv")
    labels = pd.read_csv("dataset/target_labels_DTC_FM.csv")
    assert feats.shape == (2508, 15)
    assert labels.shape == (2508, 1)
    assert list(labels.columns) == [TARGET]


# 11. Reusable probability pipeline validates input and emits correct output.
def test_probability_pipeline_interface(parts):
    pipeline = ProbabilityPipeline()
    assert pipeline.model_version == f"{PRIMARY_RUN}_calibrated"
    feats = pipeline.features
    assert "sample_index" not in feats
    assert TARGET not in feats

    sample = parts["test"].iloc[:5][feats]
    records = pipeline.predict(sample)
    assert len(records) == 5
    for r in records:
        assert 0 <= r["class_0_probability"] <= 1
        assert 0 <= r["class_1_probability"] <= 1
        assert r["calibrated"] is True
        assert r["model_version"] == pipeline.model_version
        assert np.isclose(r["class_0_probability"] + r["class_1_probability"], 1.0, atol=1e-5)


# 12. Probability pipeline rejects forbidden columns.
@pytest.mark.parametrize("forbidden_col", ["sample_index", "event_type", "split"])
def test_probability_pipeline_rejects_forbidden_columns(parts, forbidden_col):
    pipeline = ProbabilityPipeline()
    feats = pipeline.features
    bad = parts["test"].iloc[:1][feats].copy()
    bad[forbidden_col] = 0
    with pytest.raises(ValueError, match="forbidden columns"):
        pipeline.predict(bad)


# 12b. infer_probability convenience function works for a single vector.
def test_infer_probability(parts):
    pipeline = ProbabilityPipeline()
    feats = pipeline.features
    vector = parts["test"].iloc[0][feats].to_dict()
    record = infer_probability(vector)
    assert 0 <= record["class_0_probability"] <= 1
    assert 0 <= record["class_1_probability"] <= 1
    assert record["calibrated"] is True
    assert record["model_version"] == "random_forest_A_calibrated"
    assert np.isclose(record["class_0_probability"] + record["class_1_probability"], 1.0, atol=1e-5)


# 13. Split integrity still holds after calibration.
def test_split_integrity_after_calibration(parts):
    integrity = verify_split_integrity(parts)
    assert integrity["pass"], integrity
    assert integrity["duplicate_groups_crossing_splits"] == 0
    assert integrity["identical_feature_rows_crossing_splits"] == 0
    assert integrity["sample_index_overlap_total"] == 0


# 14. Calibration improves probability quality on validation (lower Brier/log-loss vs raw).
def test_calibration_improves_probability_quality(parts):
    from sklearn.metrics import brier_score_loss, log_loss
    raw_pipe = joblib.load(MODEL_DIR / f"{PRIMARY_RUN}.joblib")
    cal_pipe = joblib.load(MODEL_DIR / f"{PRIMARY_RUN}_calibrated.joblib")
    feats = list(raw_pipe.named_steps["preprocess"].feature_names_in_)
    X_val = parts["validation"][feats]
    y_val = parts["validation"][TARGET].values
    raw_p = raw_pipe.predict_proba(X_val)[:, 1]
    cal_p = cal_pipe.predict_proba(X_val)[:, 1]
    raw_brier = brier_score_loss(y_val, raw_p)
    cal_brier = brier_score_loss(y_val, cal_p)
    raw_ll = log_loss(y_val, np.clip(raw_p, 1e-15, 1 - 1e-15))
    cal_ll = log_loss(y_val, np.clip(cal_p, 1e-15, 1 - 1e-15))
    # At least one of Brier or log-loss should improve after calibration.
    assert cal_brier < raw_brier or cal_ll < raw_ll


# 15. Reports and figures produced.
def test_phase4_outputs_exist():
    required = [
        REPORT_DIR / "calibration_report.md",
        REPORT_DIR / "calibration_comparison.csv",
        REPORT_DIR / "threshold_analysis_calibrated.csv",
        REPORT_DIR / "error_analysis.md",
        REPORT_DIR / "robustness_analysis.md",
        REPORT_DIR / "probability_contract.md",
        Path("reports/figures/near_miss/calibration_comparison.png"),
        Path("reports/figures/near_miss/calibrated_precision_recall.png"),
        Path("reports/figures/near_miss/calibrated_threshold_tradeoff.png"),
        Path("reports/figures/near_miss/false_negative_analysis.png"),
        Path("reports/figures/near_miss/false_positive_analysis.png"),
    ]
    for p in required:
        assert p.exists(), f"missing {p}"
