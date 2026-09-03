"""Phase 3 — near-miss model training (DTC-FM).

Deterministic (RANDOM_SEED=42). Trains Dummy/LR/RF/GB across three documented
feature configurations; preprocessing is fitted on TRAIN only inside sklearn
Pipelines; full pipelines (preprocess + model) are saved to models/near_miss/.

Configs:
  A offline_full_no_fesafety      all 15 features except FE_safety_index
  B offline_full_with_fesafety    all 15 features (A/B isolates FE_safety_index)
  D deployment_realistic          excludes PET + FE_inv_PET (post-event) and
                                  FE_safety_index (unstable composite)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
SPLIT_DIR = Path("data/processed/near_miss/splits")
MODEL_DIR = Path("models/near_miss")
REPORT_DIR = Path("reports/near_miss")

ALL_FEATURES = [
    "speed_object_1_kph", "speed_object_2_kph", "angle_degrees", "PET",
    "acceleration_obj1_mps2", "acceleration_obj2_mps2",
    "min_dist_dual_check_m", "object_count_at_frame1", "target_dist_px",
    "FE_inv_PET", "FE_safety_index", "FE_log_mesafe", "FE_dist_squared",
    "class_object_1", "class_object_2",
]
TARGET = "event_type"
CAT_FEATURES = ["class_object_1", "class_object_2"]

CONFIGS = {
    "A_offline_full_no_fesafety": [f for f in ALL_FEATURES if f != "FE_safety_index"],
    "B_offline_full_with_fesafety": list(ALL_FEATURES),
    "D_deployment_realistic": [
        f for f in ALL_FEATURES
        if f not in {"PET", "FE_inv_PET", "FE_safety_index"}
    ],
}

# (run_name, config, model factory)  — model factories close over the seed
def _lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=RANDOM_SEED)

def _rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced", random_state=RANDOM_SEED,
                                  n_jobs=-1)

def _gb():
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=3, random_state=RANDOM_SEED)

RUNS = [
    ("dummy_baseline", "A_offline_full_no_fesafety", lambda: DummyClassifier(strategy="most_frequent")),
    ("logistic_regression_A", "A_offline_full_no_fesafety", _lr),
    ("logistic_regression_B", "B_offline_full_with_fesafety", _lr),
    ("logistic_regression_D", "D_deployment_realistic", _lr),
    ("random_forest_A", "A_offline_full_no_fesafety", _rf),
    ("random_forest_B", "B_offline_full_with_fesafety", _rf),
    ("random_forest_D", "D_deployment_realistic", _rf),
    ("gradient_boosting_A", "A_offline_full_no_fesafety", _gb),
    ("gradient_boosting_B", "B_offline_full_with_fesafety", _gb),
    ("gradient_boosting_D", "D_deployment_realistic", _gb),
]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    num = [f for f in features if f not in CAT_FEATURES]
    cat = [f for f in features if f in CAT_FEATURES]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(np.unique(y_true)) < 2:  # dummy classifier can predict a single class
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def train_all() -> list[dict]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parts = {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet")
             for s in ["train", "validation", "test"]}
    results = []
    for run_name, config, factory in RUNS:
        feats = CONFIGS[config]
        X = {s: parts[s][feats] for s in parts}
        y = {s: parts[s][TARGET].values for s in parts}

        pipe = Pipeline([
            ("preprocess", make_preprocessor(feats)),
            ("model", factory()),
        ])
        # GradientBoosting has no class_weight: use balanced sample weights
        if run_name.startswith("gradient_boosting"):
            cls, counts = np.unique(y["train"], return_counts=True)
            weights = {c: len(y["train"]) / (len(cls) * n) for c, n in zip(cls, counts)}
            fit_kw = {"model__sample_weight": pd.Series(y["train"]).map(weights).values}
        else:
            fit_kw = {}
        pipe.fit(X["train"], y["train"], **fit_kw)

        entry = {"run": run_name, "config": config,
                 "model": type(pipe.named_steps["model"]).__name__,
                 "n_features": len(feats), "random_seed": RANDOM_SEED}
        for s in parts:
            prob = pipe.predict_proba(X[s])[:, 1]
            entry[f"metrics_{s}"] = compute_metrics(y[s], prob)
        joblib.dump(pipe, MODEL_DIR / f"{run_name}.joblib")
        results.append(entry)
        m = entry["metrics_validation"]
        print(f"{run_name:28s} val: acc={m['accuracy']:.3f} P={m['precision_class1']:.3f} "
              f"R={m['recall_class1']:.3f} F1={m['f1_class1']:.3f} "
              f"ROC={m['roc_auc']} PR={m['pr_auc']} FN={m['FN']}")

    return results


if __name__ == "__main__":
    results = train_all()
    comparison = []
    for r in results:
        row = {"run": r["run"], "config": r["config"], "model": r["model"],
               "n_features": r["n_features"], "seed": r["random_seed"]}
        for s in ["train", "validation", "test"]:
            m = r[f"metrics_{s}"]
            for k in ["accuracy", "precision_class1", "recall_class1", "f1_class1",
                      "roc_auc", "pr_auc", "FNR"]:
                row[f"{s}_{k}"] = m[k]
            row[f"{s}_FN"] = m["FN"]
        comparison.append(row)
    comp = pd.DataFrame(comparison)
    comp.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {len(results)} pipelines to {MODEL_DIR}/")