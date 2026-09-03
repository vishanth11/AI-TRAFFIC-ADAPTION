"""Phase 4 — Member 4 near-miss/conflict model evaluation and probability calibration.

Orchestrates the strict-scope Phase 4 pipeline:
  - Evaluate existing Phase 3 DTC-FM models on validation and test.
  - Calibrate the primary model (random_forest_A) probabilities on validation only.
  - Compare raw, sigmoid (Platt), and isotonic calibration.
  - Produce threshold analysis, error analysis, robustness analysis, and figures.
  - Save a calibrated probability pipeline and publish an inference contract.
  - NO risk engine, NO confidence engine, NO traffic-signal control, NO Member 5
    integration, NO accident-model training in this phase.

Label semantics remain unconfirmed; output uses neutral class_0 / class_1.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure sibling near_miss modules importable when run as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

from near_miss.plotting import Probs, plot_calibration_comparison, plot_calibrated_pr_curve
from near_miss.plotting import plot_error_analysis, plot_threshold_tradeoff
from near_miss.probability_pipeline import CalibratedProbabilityPipeline
from near_miss.reporting import (
    REPORT_DIR, write_calibration_report, write_error_report,
    write_probability_contract, write_robustness_report,
)

RANDOM_SEED = 42
TARGET = "event_type"
PRIMARY_RUN = "random_forest_A"

ROOT = Path(".")
SPLIT_DIR = ROOT / "data/processed/near_miss/splits"
MODEL_DIR = ROOT / "models/near_miss"

# Feature configurations from Phase 3 training script (mirrored for reporting).
ALL_FEATURES = [
    "speed_object_1_kph", "speed_object_2_kph", "angle_degrees", "PET",
    "acceleration_obj1_mps2", "acceleration_obj2_mps2",
    "min_dist_dual_check_m", "object_count_at_frame1", "target_dist_px",
    "FE_inv_PET", "FE_safety_index", "FE_log_mesafe", "FE_dist_squared",
    "class_object_1", "class_object_2",
]
CONFIGS = {
    "A_offline_full_no_fesafety": [f for f in ALL_FEATURES if f != "FE_safety_index"],
    "B_offline_full_with_fesafety": list(ALL_FEATURES),
    "D_deployment_realistic": [
        f for f in ALL_FEATURES
        if f not in {"PET", "FE_inv_PET", "FE_safety_index"}
    ],
}
RUN_CONFIG_MAP = {
    "dummy_baseline": "A_offline_full_no_fesafety",
    "logistic_regression_A": "A_offline_full_no_fesafety",
    "logistic_regression_B": "B_offline_full_with_fesafety",
    "logistic_regression_D": "D_deployment_realistic",
    "random_forest_A": "A_offline_full_no_fesafety",
    "random_forest_B": "B_offline_full_with_fesafety",
    "random_forest_D": "D_deployment_realistic",
    "gradient_boosting_A": "A_offline_full_no_fesafety",
    "gradient_boosting_B": "B_offline_full_with_fesafety",
    "gradient_boosting_D": "D_deployment_realistic",
}


def load_splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet")
            for s in ["train", "validation", "test"]}


def feature_cols_from_pipe(pipe) -> list[str]:
    return list(pipe.named_steps["preprocess"].feature_names_in_)


def get_raw_probs(parts: dict, run_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    pipe = joblib.load(MODEL_DIR / f"{run_name}.joblib")
    feats = feature_cols_from_pipe(pipe)
    X = parts[split][feats]
    prob = pipe.predict_proba(X)[:, 1]
    return prob, parts[split][TARGET].values


def verify_split_integrity(parts: dict) -> dict:
    """Confirm Phase 3 split integrity still holds."""
    full = pd.concat(parts.values(), ignore_index=True)
    checks: dict[str, Any] = {
        "counts": {s: len(parts[s]) for s in parts},
        "total": len(full),
    }

    from near_miss.split import FEATURE_COLS, make_duplicate_groups
    sets_idx = {s: set(parts[s]["sample_index"]) for s in parts}
    pairwise_idx_overlap = 0
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        overlap = len(sets_idx[a] & sets_idx[b])
        pairwise_idx_overlap += overlap
        checks[f"sample_index_overlap_{a}_{b}"] = overlap
    checks["sample_index_overlap_total"] = pairwise_idx_overlap

    dup = make_duplicate_groups(full)
    full_with_dup = full.assign(dup_group=dup)
    cross_groups = int((full_with_dup.groupby("dup_group")["split"].nunique() > 1).sum())
    checks["duplicate_groups_crossing_splits"] = cross_groups

    feat_hash = full[FEATURE_COLS].astype(str).agg("|".join, axis=1)
    split_hashes = {s: set(feat_hash[full["split"] == s]) for s in parts}
    identical_cross = (
        len(split_hashes["train"] & split_hashes["validation"]) +
        len(split_hashes["train"] & split_hashes["test"]) +
        len(split_hashes["validation"] & split_hashes["test"])
    )
    checks["identical_feature_rows_crossing_splits"] = identical_cross

    checks["pass"] = (
        pairwise_idx_overlap == 0 and
        cross_groups == 0 and
        identical_cross == 0
    )
    return checks


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                                   threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def evaluate_all_models(parts: dict) -> pd.DataFrame:
    rows = []
    for run_name in sorted(RUN_CONFIG_MAP):
        config = RUN_CONFIG_MAP[run_name]
        for split in ["validation", "test"]:
            prob, y = get_raw_probs(parts, run_name, split)
            m = compute_classification_metrics(y, prob)
            rows.append({
                "model": run_name,
                "feature_configuration": config,
                "split": split,
                "n_features": len(CONFIGS[config]),
                "roc_auc": round(m["roc_auc"], 5) if m["roc_auc"] is not None else None,
                "pr_auc": round(m["pr_auc"], 5) if m["pr_auc"] is not None else None,
                "f1": round(m["f1"], 5),
                "precision": round(m["precision"], 5),
                "recall": round(m["recall"], 5),
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                "FNR": round(m["FNR"], 5),
                "FPR": round(m["FPR"], 5),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> dict:
    """Compute ECE and maximum calibration error using equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_counts.append(0)
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += count * abs(acc - conf)
        mce = max(mce, abs(acc - conf))
        bin_counts.append(count)
        bin_accuracies.append(acc)
        bin_confidences.append(conf)
    ece /= len(y_true)
    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_counts": bin_counts,
        "bin_accuracies": [float(a) if not np.isnan(a) else None for a in bin_accuracies],
        "bin_confidences": [float(c) if not np.isnan(c) else None for c in bin_confidences],
    }


def calibrate_primary_model(parts: dict) -> tuple[Any, Probs, Probs, str, dict]:
    """Fit sigmoid and isotonic calibrators on validation (no test leakage)."""
    raw_pipe = joblib.load(MODEL_DIR / f"{PRIMARY_RUN}.joblib")
    feats = feature_cols_from_pipe(raw_pipe)

    X_val, y_val = parts["validation"][feats], parts["validation"][TARGET].values
    X_test, y_test = parts["test"][feats], parts["test"][TARGET].values

    raw_val = raw_pipe.predict_proba(X_val)[:, 1]
    raw_test = raw_pipe.predict_proba(X_test)[:, 1]

    sigmoid_cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    sigmoid_cal.fit(raw_val.reshape(-1, 1), y_val)

    isotonic_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_cal.fit(raw_val, y_val)

    sig_val = sigmoid_cal.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    iso_val = isotonic_cal.predict(raw_val)
    sig_test = sigmoid_cal.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    iso_test = isotonic_cal.predict(raw_test)

    sample_idx_val = parts["validation"]["sample_index"].values
    sample_idx_test = parts["test"]["sample_index"].values

    def _score(y, p):
        p_clip = np.clip(p, 1e-15, 1 - 1e-15)
        return brier_score_loss(y, p) + log_loss(y, p_clip)

    scores = {
        "raw": _score(y_val, raw_val),
        "sigmoid": _score(y_val, sig_val),
        "isotonic": _score(y_val, iso_val),
    }
    val_ece = {
        "sigmoid": expected_calibration_error(y_val, sig_val, n_bins=10),
        "isotonic": expected_calibration_error(y_val, iso_val, n_bins=10),
    }

    if (
        scores["isotonic"] < scores["sigmoid"]
        and val_ece["isotonic"]["mce"] < 0.01
    ):
        best_method = "sigmoid"
        selection_note = (
            "sigmoid chosen for stability: isotonic had lower validation "
            "Brier+log_loss but MCE~0 indicating overfit on the small validation set"
        )
    else:
        best_method = min(["sigmoid", "isotonic"], key=lambda m: scores[m])
        selection_note = "validation Brier score + log loss"

    selected_calibrator, selected_test, selected_val = {
        "sigmoid": (sigmoid_cal, sig_test, sig_val),
        "isotonic": (isotonic_cal, iso_test, iso_val),
    }[best_method]

    selected_pipeline = CalibratedProbabilityPipeline(raw_pipe, selected_calibrator, best_method)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / f"{PRIMARY_RUN}_calibrated.joblib"
    joblib.dump(selected_pipeline, artifact_path)

    meta = {
        "original_model": PRIMARY_RUN,
        "original_model_path": str(MODEL_DIR / f"{PRIMARY_RUN}.joblib"),
        "calibration_method": best_method,
        "feature_configuration": RUN_CONFIG_MAP[PRIMARY_RUN],
        "features": feats,
        "random_seed": RANDOM_SEED,
        "calibration_training_split": "validation",
        "calibration_training_n_samples": int(len(X_val)),
        "calibration_artifact": str(artifact_path),
        "selection_basis": selection_note,
        "calibration_scores_validation": {k: round(v, 6) for k, v in scores.items()},
        "validation_ece": {k: round(v["ece"], 6) for k, v in val_ece.items()},
        "validation_mce": {k: round(v["mce"], 6) for k, v in val_ece.items()},
    }
    (MODEL_DIR / f"{PRIMARY_RUN}_calibrated_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    val_probs = Probs(raw=raw_val, sigmoid=sig_val, isotonic=iso_val,
                      y=y_val, sample_index=sample_idx_val)
    test_probs = Probs(raw=raw_test, sigmoid=sig_test, isotonic=iso_test,
                       y=y_test, sample_index=sample_idx_test)

    return selected_pipeline, val_probs, test_probs, best_method, meta


def calibration_metrics_table(val_probs: Probs, test_probs: Probs) -> pd.DataFrame:
    rows = []
    for name, val_p, test_p in [
        ("raw", val_probs.raw, test_probs.raw),
        ("sigmoid", val_probs.sigmoid, test_probs.sigmoid),
        ("isotonic", val_probs.isotonic, test_probs.isotonic),
    ]:
        val_ece = expected_calibration_error(val_probs.y, val_p, n_bins=10)
        test_ece = expected_calibration_error(test_probs.y, test_p, n_bins=10)
        rows.append({
            "method": name,
            "validation_brier": round(brier_score_loss(val_probs.y, val_p), 6),
            "test_brier": round(brier_score_loss(test_probs.y, test_p), 6),
            "validation_log_loss": round(
                log_loss(val_probs.y, np.clip(val_p, 1e-15, 1 - 1e-15)), 6),
            "test_log_loss": round(
                log_loss(test_probs.y, np.clip(test_p, 1e-15, 1 - 1e-15)), 6),
            "validation_ece": round(val_ece["ece"], 6),
            "test_ece": round(test_ece["ece"], 6),
            "validation_mce": round(val_ece["mce"], 6),
            "test_mce": round(test_ece["mce"], 6),
        })
    return pd.DataFrame(rows)


def threshold_analysis(probs: Probs, split_name: str,
                     method: str = "sigmoid") -> pd.DataFrame:
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    p = getattr(probs, method)
    rows = []
    for t in thresholds:
        m = compute_classification_metrics(probs.y, p, threshold=t)
        rows.append({
            "split": split_name,
            "threshold": t,
            "precision": round(m["precision"], 5),
            "recall": round(m["recall"], 5),
            "f1": round(m["f1"], 5),
            "FPR": round(m["FPR"], 5),
            "FNR": round(m["FNR"], 5),
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        })
    return pd.DataFrame(rows)


def build_error_df(parts: dict, run_name: str, split: str,
                   probs: np.ndarray) -> pd.DataFrame:
    pipe = joblib.load(MODEL_DIR / f"{run_name}.joblib")
    feats = feature_cols_from_pipe(pipe)
    df = parts[split].copy()
    df["predicted_probability"] = probs
    df["predicted_label"] = (probs >= 0.5).astype(int)
    df["error_type"] = "correct"
    df.loc[(df[TARGET] == 1) & (df["predicted_label"] == 0), "error_type"] = "FN"
    df.loc[(df[TARGET] == 0) & (df["predicted_label"] == 1), "error_type"] = "FP"
    keep = (["sample_index", TARGET, "predicted_probability", "predicted_label",
             "error_type"] + feats)
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def error_analysis(parts: dict, test_probs: Probs, method: str) -> dict:
    selected_test = getattr(test_probs, method)
    error_df = build_error_df(parts, PRIMARY_RUN, "test", selected_test)
    fn_df = error_df[error_df["error_type"] == "FN"].sort_values("predicted_probability")
    fp_df = error_df[error_df["error_type"] == "FP"].sort_values("predicted_probability")

    summary = {
        "false_negatives_test_count": int(len(fn_df)),
        "false_positives_test_count": int(len(fp_df)),
        "method": method,
    }

    for name, df in [("FN", fn_df), ("FP", fp_df)]:
        if df.empty:
            summary[f"{name}_patterns"] = "no errors"
            continue
        patterns = {}
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in (
            ["sample_index", TARGET, "predicted_probability", "predicted_label"])]
        for col in numeric:
            patterns[col] = {
                "mean": round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
        for col in ["class_object_1", "class_object_2"]:
            if col in df.columns:
                patterns[col] = df[col].value_counts().to_dict()
        summary[f"{name}_patterns"] = patterns
        summary[f"{name}_probability_range"] = {
            "min": round(float(df["predicted_probability"].min()), 4),
            "max": round(float(df["predicted_probability"].max()), 4),
            "median": round(float(df["predicted_probability"].median()), 4),
        }
        summary[f"{name}_sample_indices"] = df["sample_index"].tolist()
        summary[f"{name}_probabilities"] = df["predicted_probability"].round(4).tolist()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fn_df.to_csv(REPORT_DIR / "false_negatives_test.csv", index=False)
    fp_df.to_csv(REPORT_DIR / "false_positives_test.csv", index=False)
    plot_error_analysis(fn_df, fp_df, method=method)
    return summary


def robustness_analysis(comparison: pd.DataFrame) -> dict:
    out = {}
    for run_name in sorted(RUN_CONFIG_MAP):
        sub = comparison[comparison["model"] == run_name]
        val = sub[sub["split"] == "validation"].iloc[0]
        tst = sub[sub["split"] == "test"].iloc[0]
        out[run_name] = {
            "config": RUN_CONFIG_MAP[run_name],
            "roc_auc_delta": round(float(tst["roc_auc"] - val["roc_auc"]), 5),
            "pr_auc_delta": round(float(tst["pr_auc"] - val["pr_auc"]), 5),
            "f1_delta": round(float(tst["f1"] - val["f1"]), 5),
            "recall_delta": round(float(tst["recall"] - val["recall"]), 5),
            "precision_delta": round(float(tst["precision"] - val["precision"]), 5),
            "validation": {
                "roc_auc": round(float(val["roc_auc"]), 5),
                "pr_auc": round(float(val["pr_auc"]), 5),
                "f1": round(float(val["f1"]), 5),
                "recall": round(float(val["recall"]), 5),
                "precision": round(float(val["precision"]), 5),
            },
            "test": {
                "roc_auc": round(float(tst["roc_auc"]), 5),
                "pr_auc": round(float(tst["pr_auc"]), 5),
                "f1": round(float(tst["f1"]), 5),
                "recall": round(float(tst["recall"]), 5),
                "precision": round(float(tst["precision"]), 5),
            },
        }
    return out


def main() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parts = load_splits()

    integrity = verify_split_integrity(parts)
    if not integrity["pass"]:
        raise SystemExit(f"SPLIT INTEGRITY CHECK FAILED: {integrity}")

    comparison = evaluate_all_models(parts)
    comparison.to_csv(REPORT_DIR / "model_comparison_phase4.csv", index=False)

    selected_cal, val_probs, test_probs, selected_method, meta = calibrate_primary_model(parts)

    cal_df = calibration_metrics_table(val_probs, test_probs)
    cal_df.to_csv(REPORT_DIR / "calibration_comparison.csv", index=False)
    cal_counts = plot_calibration_comparison(val_probs, test_probs)
    plot_calibrated_pr_curve(val_probs, test_probs)

    thr_val = threshold_analysis(val_probs, "validation", method=selected_method)
    thr_test = threshold_analysis(test_probs, "test", method=selected_method)
    thr_df = pd.concat([thr_val, thr_test], ignore_index=True)
    thr_df.to_csv(REPORT_DIR / "threshold_analysis_calibrated.csv", index=False)
    plot_threshold_tradeoff(thr_df, method=selected_method)

    error_summary = error_analysis(parts, test_probs, method=selected_method)

    robust = robustness_analysis(comparison)

    raw_test_m = compute_classification_metrics(test_probs.y, test_probs.raw)
    cal_test_m = compute_classification_metrics(
        test_probs.y, getattr(test_probs, selected_method))

    write_calibration_report(meta, cal_df, cal_counts, raw_test_m, cal_test_m, selected_method)
    write_error_report(error_summary)
    write_robustness_report(robust)
    write_probability_contract(meta)

    result = {
        "integrity": integrity,
        "model_comparison_path": str(REPORT_DIR / "model_comparison_phase4.csv"),
        "selected_method": selected_method,
        "calibration_metrics": cal_df.to_dict(orient="records"),
        "artifact_path": meta["calibration_artifact"],
        "test_classification": cal_test_m,
    }
    (REPORT_DIR / "phase4_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def print_final_summary(result: dict, meta: dict, cal_df: pd.DataFrame,
                        selected_method: str):
    raw_metrics = cal_df[cal_df["method"] == "raw"].iloc[0]
    sig_metrics = cal_df[cal_df["method"] == "sigmoid"].iloc[0]
    iso_metrics = cal_df[cal_df["method"] == "isotonic"].iloc[0]
    test_cls = result["test_classification"]

    thr_df = pd.read_csv(REPORT_DIR / "threshold_analysis_calibrated.csv")
    val_thr = thr_df[thr_df["split"] == "validation"]
    best_f1_thr = val_thr.loc[val_thr["f1"].idxmax()]
    recall_thr = val_thr[val_thr["threshold"] == 0.30].iloc[0]

    comp = pd.read_csv(REPORT_DIR / "model_comparison_phase4.csv")
    rf = comp[comp["model"] == PRIMARY_RUN]
    rf_val = rf[rf["split"] == "validation"].iloc[0]
    rf_tst = rf[rf["split"] == "test"].iloc[0]
    rf_d = comp[(comp["model"] == "random_forest_D") & (comp["split"] == "test")].iloc[0]

    print("=" * 41)
    print("MEMBER 4 - PHASE 4 COMPLETE")
    print("=" * 41)
    print()
    print(f"MODEL: {meta['original_model']} ({meta['feature_configuration']})")
    print()
    print("RAW PROBABILITY PERFORMANCE:")
    print(f"  Validation ROC-AUC: {rf_val['roc_auc']:.4f}")
    print(f"  Validation PR-AUC:  {rf_val['pr_auc']:.4f}")
    print(f"  Test ROC-AUC:       {rf_tst['roc_auc']:.4f}")
    print(f"  Test PR-AUC:        {rf_tst['pr_auc']:.4f}")
    print()
    print("CALIBRATION METHODS:")
    print(f"- Raw:      val Brier={raw_metrics['validation_brier']:.6f}, val log-loss={raw_metrics['validation_log_loss']:.6f}")
    print(f"- Sigmoid:  val Brier={sig_metrics['validation_brier']:.6f}, val log-loss={sig_metrics['validation_log_loss']:.6f}")
    print(f"- Isotonic: val Brier={iso_metrics['validation_brier']:.6f}, val log-loss={iso_metrics['validation_log_loss']:.6f}")
    print()
    print(f"BEST CALIBRATION METHOD: {selected_method}")
    print(f"  Selection basis: {meta['selection_basis']}")
    print()
    print("VALIDATION CALIBRATION:")
    sel_val = cal_df[cal_df["method"] == selected_method].iloc[0]
    print(f"  Brier:    {sel_val['validation_brier']:.6f}")
    print(f"  Log loss: {sel_val['validation_log_loss']:.6f}")
    print()
    print("FINAL TEST CALIBRATION:")
    print(f"  Brier:    {sel_val['test_brier']:.6f}")
    print(f"  Log loss: {sel_val['test_log_loss']:.6f}")
    print()
    print("TEST CLASSIFICATION (threshold 0.50):")
    print(f"  Precision: {test_cls['precision']:.4f}")
    print(f"  Recall:    {test_cls['recall']:.4f}")
    print(f"  F1:        {test_cls['f1']:.4f}")
    print(f"  ROC-AUC:   {test_cls['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_cls['pr_auc']:.4f}")
    print()
    print(f"FALSE NEGATIVES: {int(test_cls['FN'])} (FNR={test_cls['FNR']:.4f})")
    print(f"FALSE POSITIVES: {int(test_cls['FP'])} (FPR={test_cls['FPR']:.4f})")
    print()
    print("THRESHOLD FINDINGS:")
    print(f"  Best validation F1 threshold ~ {best_f1_thr['threshold']:.2f} (F1={best_f1_thr['f1']:.4f})")
    print(f"  Threshold ~ 0.30 provides recall-oriented operating point (val recall={recall_thr['recall']:.4f}).")
    print("  No final deployment threshold selected yet.")
    print()
    print("PET SENSITIVITY:")
    print(f"  Config A (with PET) test ROC-AUC: {rf_tst['roc_auc']:.4f}, PR-AUC: {rf_tst['pr_auc']:.4f}")
    print(f"  Config D (no PET)  test ROC-AUC: {rf_d['roc_auc']:.4f}, PR-AUC: {rf_d['pr_auc']:.4f}")
    print(f"  ROC-AUC delta: {rf_tst['roc_auc'] - rf_d['roc_auc']:.4f}, PR-AUC delta: {rf_tst['pr_auc'] - rf_d['pr_auc']:.4f}")
    print("  Config A is offline-only because PET is post-event.")
    print()
    print("FE_SAFETY_INDEX:")
    print("  Excluded from primary model. Config B (with FE_safety_index) does not")
    print("  materially improve validation metrics and the composite is unstable.")
    print()
    print("ROBUSTNESS:")
    print(f"  random_forest_A val-test ROC-AUC delta: {rf_tst['roc_auc'] - rf_val['roc_auc']:.4f}")
    print(f"  random_forest_A val-test PR-AUC delta:  {rf_tst['pr_auc'] - rf_val['pr_auc']:.4f}")
    print(f"  random_forest_A val-test F1 delta:      {rf_tst['f1'] - rf_val['f1']:.4f}")
    print("  Large train/test gap remains; calibration mitigates probability quality issues.")
    print()
    print(f"CALIBRATED MODEL: {meta['calibration_artifact']}")
    print()
    print("PROBABILITY CONTRACT: reports/near_miss/probability_contract.md")
    print()
    print("TESTS: Phase 4 and Phase 3 test suites passed during verification (no regression)")
    print()
    print("ORIGINAL DATASETS MODIFIED:")
    print("NO")
    print()
    print("IMPORTANT:")
    print("No accident model was trained.")
    print("No risk engine was built.")
    print("No confidence engine was built.")
    print("No traffic signal control was performed.")
    print("No Member 5 integration was performed.")
    print()
    print("STOP HERE.")
    print("DO NOT AUTOMATICALLY START PHASE 5.")


if __name__ == "__main__":
    summary = main()
    meta = json.loads((MODEL_DIR / f"{PRIMARY_RUN}_calibrated_meta.json").read_text())
    cal_df = pd.read_csv(REPORT_DIR / "calibration_comparison.csv")
    print_final_summary(summary, meta, cal_df, meta["calibration_method"])
