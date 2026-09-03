"""Phase 3 integrity tests for the near-miss pipeline.

Run: pytest tests/test_near_miss_phase3.py -v
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from near_miss.split import FEATURE_COLS, build_split, make_duplicate_groups  # noqa: E402

DATA_NM = Path("data/processed/near_miss")
SPLIT_DIR = DATA_NM / "splits"
MODEL_DIR = Path("models/near_miss")
TARGET = "event_type"


@pytest.fixture(scope="module")
def parts():
    return {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet")
            for s in ["train", "validation", "test"]}


# 7. processed dataset row count remains 2,508 -------------------------------
def test_processed_row_count():
    df = pd.read_parquet(DATA_NM / "dtc_fm_combined.parquet")
    assert len(df) == 2508
    total = sum(len(pd.read_parquet(SPLIT_DIR / f"{s}.parquet"))
                for s in ["train", "validation", "test"])
    assert total == 2508


# 1 + 2. target and sample_index are not model features ----------------------
def test_target_and_index_not_in_features(parts):
    # Exclude Phase 4 calibrated artifacts; they wrap the original pipelines.
    phase3_models = [p.stem for p in MODEL_DIR.glob("*.joblib")
                     if "_calibrated" not in p.stem]
    for run in phase3_models:
        pipe = joblib.load(MODEL_DIR / f"{run}.joblib")
        feats = list(pipe.named_steps["preprocess"].feature_names_in_)
        assert TARGET not in feats, f"{run} uses target as feature"
        assert "sample_index" not in feats, f"{run} uses sample_index as feature"
        assert "split" not in feats, f"{run} uses split as feature"


# 3. no duplicate leakage across splits ---------------------------------------
def test_no_duplicate_leakage(parts):
    full = pd.concat(parts.values(), ignore_index=True)
    dup = make_duplicate_groups(full)
    grouped = full.assign(dup_group=dup).groupby("dup_group")["split"].nunique()
    assert (grouped == 1).all(), "duplicate group spans multiple splits"

    feat_hash = full[FEATURE_COLS].astype(str).agg("|".join, axis=1)
    sets = {s: set(feat_hash[full["split"] == s]) for s in parts}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])

    idx = [set(parts[s]["sample_index"]) for s in parts]
    assert not set.intersection(*idx)


# 4. preprocessing is fitted only on training data ----------------------------
def test_preprocessing_fitted_on_train_only(parts):
    pipe = joblib.load(MODEL_DIR / "random_forest_A.joblib")
    pre = pipe.named_steps["preprocess"]
    # one-hot categories must be a subset of TRAIN categories (unknown categories
    # in val/test must be ignored, not learned)
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    seen = set(ohe.categories_[0]) | set(ohe.categories_[1])
    train_cats = (set(parts["train"]["class_object_1"])
                  | set(parts["train"]["class_object_2"]))
    unseen = seen - train_cats
    assert not unseen, f"encoder learned non-train categories: {unseen}"

    # scaler statistics must match TRAIN, not the full dataset
    num_idx = [i for i, name in enumerate(pre.feature_names_in_)
               if name not in ("class_object_1", "class_object_2")]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    train_means = parts["train"][np.array(pre.feature_names_in_)[num_idx]].mean()
    assert np.allclose(scaler.mean_, train_means.values, atol=1e-6)


# 5. split is reproducible -----------------------------------------------------
def test_split_reproducible():
    df = pd.read_parquet(DATA_NM / "dtc_fm_combined.parquet")
    assignments = build_split()[1]
    saved = pd.read_csv(DATA_NM / "split_assignments.csv")
    merged = assignments.merge(saved, on="sample_index", suffixes=("_new", "_saved"))
    assert (merged["split_new"] == merged["split_saved"]).all()
    assert (merged["event_type_new"] == merged["event_type_saved"]).all()


# 6. model outputs probabilities ------------------------------------------------
def test_models_output_probabilities(parts):
    for run in ["logistic_regression_A", "random_forest_A", "gradient_boosting_A"]:
        pipe = joblib.load(MODEL_DIR / f"{run}.joblib")
        feats = list(pipe.named_steps["preprocess"].feature_names_in_)
        prob = pipe.predict_proba(parts["test"][feats])
        assert prob.shape == (len(parts["test"]), 2)
        assert np.allclose(prob.sum(axis=1), 1.0)
        assert (prob >= 0).all() and (prob <= 1).all()


# 8. original datasets remain untouched -----------------------------------------
def test_originals_untouched():
    # files still exist with the exact shapes recorded in the Phase 1 audit
    feats = pd.read_csv("dataset/filtered_features_DTC_FM.csv")
    labels = pd.read_csv("dataset/target_labels_DTC_FM.csv")
    assert feats.shape == (2508, 15)
    assert labels.shape == (2508, 1)
    assert list(labels.columns) == [TARGET]
    # and the parquet shards still exist
    for f in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet",
              "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]:
        assert Path("dataset", f).exists()