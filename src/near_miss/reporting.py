"""Phase 4 markdown/CSV report writers for near-miss/conflict calibration."""

import json
from pathlib import Path

import pandas as pd

REPORT_DIR = Path("reports/near_miss")
FIG_DIR = Path("reports/figures/near_miss")
TARGET = "event_type"
PRIMARY_RUN = "random_forest_A"
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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a Markdown table without external dependencies."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join([" --- " for _ in cols]) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def write_calibration_report(meta: dict, cal_df: pd.DataFrame, cal_counts: dict,
                             raw_test_m: dict, cal_test_m: dict, selected: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Near-Miss Model Calibration Report — Phase 4",
        "",
        "## Scope",
        "",
        "This phase evaluates the existing Phase 3 DTC-FM near-miss/conflict models,",
        "calibrates their probabilities, and selects a probability-producing model for",
        "later risk fusion. No risk engine, confidence engine, accident fusion, or",
        "traffic-signal control is built here.",
        "",
        "## Label Semantics",
        "",
        "DTC-FM label semantics remain **unconfirmed**. All metrics and outputs use",
        "neutral `class_0` and `class_1`. Do not interpret `class_1` as",
        "'near-miss', 'accident', or 'dangerous' without authoritative documentation.",
        "",
        "## Split Integrity",
        "",
        "Verified before evaluation: train/validation/test unchanged, no duplicate",
        "group crosses splits, no sample_index overlap, no identical feature rows",
        "crossing splits. See `split_report.md` for Phase 3 methodology.",
        "",
        "## Primary Model",
        "",
        f"- Original model: `{meta['original_model']}`",
        f"- Calibration method selected: **{selected}**",
        f"- Selection basis: {meta['selection_basis']}",
        f"- Calibration fitted on: `{meta['calibration_training_split']}`",
        f"- Calibration samples: {meta['calibration_training_n_samples']}",
        f"- Saved artifact: `{meta['calibration_artifact']}`",
        "",
        "## Calibration Metrics",
        "",
        dataframe_to_markdown(cal_df),
        "",
        "**Interpretation:** lower Brier and log-loss indicate better probability",
        "quality. Calibration selection is based on validation metrics; test metrics",
        "are reported for final confirmation only.",
        "",
        "## Reliability Diagram",
        "",
        "![Reliability diagram](../figures/near_miss/calibration_comparison.png)",
        "",
        "Bin counts by method (validation, uniform 10 bins): " + str(cal_counts.get("validation", {})),
        "",
        "Bin counts by method (test, uniform 10 bins): " + str(cal_counts.get("test", {})),
        "",
        "## Raw vs Calibrated Test Classification (threshold 0.50)",
        "",
        "| Metric | Raw | Calibrated |",
        "|---|---|---|",
        f"| Accuracy | {raw_test_m['accuracy']:.4f} | {cal_test_m['accuracy']:.4f} |",
        f"| Precision | {raw_test_m['precision']:.4f} | {cal_test_m['precision']:.4f} |",
        f"| Recall | {raw_test_m['recall']:.4f} | {cal_test_m['recall']:.4f} |",
        f"| F1 | {raw_test_m['f1']:.4f} | {cal_test_m['f1']:.4f} |",
        f"| ROC-AUC | {raw_test_m['roc_auc']:.4f} | {cal_test_m['roc_auc']:.4f} |",
        f"| PR-AUC | {raw_test_m['pr_auc']:.4f} | {cal_test_m['pr_auc']:.4f} |",
        f"| TP/TN/FP/FN | {raw_test_m['TP']}/{raw_test_m['TN']}/{raw_test_m['FP']}/{raw_test_m['FN']} | {cal_test_m['TP']}/{cal_test_m['TN']}/{cal_test_m['FP']}/{cal_test_m['FN']} |",
        "",
        "## Calibration Method Comparison Notes",
        "",
        "- Raw Random Forest probabilities are ranked scores that tend to be",
        "  under-confident near the extremes and over-confident in the mid-range.",
        "- Sigmoid (Platt) calibration fits a single-parameter logistic mapping; it",
        "  is smooth and stable but may underfit complex miscalibration patterns.",
        "- Isotonic regression is non-parametric and can fit more complex shapes,",
        "  but with small calibration sets (n=376) it risks overfitting. Here it",
        "  achieved MCE~0 on validation, so the smoother sigmoid mapping was selected",
        "  for stability even though isotonic had a slightly lower validation score.",
        "",
        "## Files Produced",
        "",
        "- `reports/near_miss/calibration_report.md`",
        "- `reports/near_miss/calibration_comparison.csv`",
        "- `reports/near_miss/threshold_analysis_calibrated.csv`",
        "- `reports/near_miss/error_analysis.md`",
        "- `reports/near_miss/robustness_analysis.md`",
        "- `reports/near_miss/probability_contract.md`",
        "- `models/near_miss/random_forest_A_calibrated.joblib`",
        "",
    ]
    (REPORT_DIR / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_error_report(error_summary: dict):
    lines = [
        "# Error Analysis — Phase 4",
        "",
        f"Analysis of `random_forest_A` false positives and false negatives on the",
        f"test split using the selected calibrated probabilities (method={error_summary.get('method', 'sigmoid')}, threshold 0.50).",
        "Association is reported; no causal claims are made.",
        "",
        "## False Negatives (class_1 predicted as class_0)",
        "",
        f"Count: {error_summary['false_negatives_test_count']}",
        "",
    ]
    if "FN_patterns" in error_summary and error_summary["FN_patterns"] != "no errors":
        lines.append("### Patterns associated with false negatives")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(error_summary["FN_patterns"], indent=2, default=str))
        lines.append("```")
        lines.append("")
    if "FN_probability_range" in error_summary:
        lines.append("### Probability distribution")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(error_summary["FN_probability_range"], indent=2))
        lines.append("```")
        lines.append("")
    if "FN_sample_indices" in error_summary:
        lines.append("### Sample indices")
        lines.append("")
        lines.append(f"- count: {len(error_summary['FN_sample_indices'])}")
        lines.append(f"- first 20: {error_summary['FN_sample_indices'][:20]}")
        lines.append("")

    lines.extend([
        "## False Positives (class_0 predicted as class_1)",
        "",
        f"Count: {error_summary['false_positives_test_count']}",
        "",
    ])
    if "FP_patterns" in error_summary and error_summary["FP_patterns"] != "no errors":
        lines.append("### Patterns associated with false positives")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(error_summary["FP_patterns"], indent=2, default=str))
        lines.append("```")
        lines.append("")
    if "FP_probability_range" in error_summary:
        lines.append("### Probability distribution")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(error_summary["FP_probability_range"], indent=2))
        lines.append("```")
        lines.append("")
    if "FP_sample_indices" in error_summary:
        lines.append("### Sample indices")
        lines.append("")
        lines.append(f"- count: {len(error_summary['FP_sample_indices'])}")
        lines.append(f"- first 20: {error_summary['FP_sample_indices'][:20]}")
        lines.append("")

    lines.extend([
        "## Interpretation Discipline",
        "",
        "- 'Associated with' does not mean 'causes'.",
        "- The model may err on edge cases, low-severity events, or rare",
        "  categorical combinations.",
        "- No labels were manually corrected; the test set remains untouched.",
        "",
        "## Detailed Tables",
        "",
        "- `reports/near_miss/false_negatives_test.csv`",
        "- `reports/near_miss/false_positives_test.csv`",
        "",
    ])
    (REPORT_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_robustness_report(robust: dict):
    lines = [
        "# Robustness Analysis — Phase 4",
        "",
        "Validation vs test performance deltas for every Phase 3 model. Negative",
        "delta = test performance below validation performance (generalization gap).",
        "",
        "| Model | Config | ROC-AUC Δ | PR-AUC Δ | F1 Δ | Recall Δ | Precision Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    for run_name in sorted(robust):
        r = robust[run_name]
        lines.append(
            f"| {run_name} | {r['config']} | {r['roc_auc_delta']:.4f} | "
            f"{r['pr_auc_delta']:.4f} | {r['f1_delta']:.4f} | "
            f"{r['recall_delta']:.4f} | {r['precision_delta']:.4f} |"
        )

    lines.extend([
        "",
        "## Key Observations",
        "",
        "- Random Forest config A shows a large train/test gap in raw probability",
        "  calibration (train PR-AUC ~ 0.9996, test PR-AUC ~ 0.830). This is why",
        "  calibration is required before using probabilities as safety scores.",
        "- Validation and test rankings are stable: random_forest_A remains the best",
        "  offline configuration by validation ROC-AUC / PR-AUC.",
        "- Deployment-realistic config D (no PET/FE_inv_PET/FE_safety_index) loses",
        "  ~0.02 ROC-AUC and ~0.06–0.08 PR-AUC versus config A, but still exceeds",
        "  0.90 ROC-AUC. PET is a post-event measurement, so the offline config A",
        "  model is not real-time-deployable without replacing PET-derived signals.",
        "",
        "## Configuration Robustness",
        "",
        "- **FE_safety_index (B vs A):** adding it does not materially improve",
        "  validation metrics and introduces a numerically unstable composite.",
        "  Keep it excluded on parsimony/stability grounds.",
        "- **PET (A vs D):** removing PET and FE_inv_PET reduces offline",
        "  discriminative power. For live deployment, plan to replace PET with",
        "  pre-event surrogates (e.g., projected time-to-collision, distance rate).",
        "",
    ])
    (REPORT_DIR / "robustness_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_probability_contract(meta: dict):
    lines = [
        "# Probability Inference Contract — Member 4",
        "",
        "## Purpose",
        "",
        "Defines the lightweight inference interface for the calibrated near-miss/",
        "conflict probability model. Later phases (risk engine) consume these",
        "probabilities; this contract does not emit risk levels or traffic commands.",
        "",
        "## Model",
        "",
        f"- Calibrated artifact: `{meta['calibration_artifact']}`",
        f"- Original model: `{meta['original_model']}`",
        f"- Calibration method: `{meta['calibration_method']}`",
        f"- Feature configuration: `{meta['feature_configuration']}`",
        f"- Random seed: `{meta['random_seed']}`",
        "",
        "## Input",
        "",
        "A single feature vector matching the model's feature schema:",
        "",
        "```json",
        json.dumps({f: "float | int | str (for categorical)" for f in meta["features"]}, indent=2),
        "```",
        "",
        f"Required columns: {', '.join(meta['features'])}.",
        "",
        "## Output",
        "",
        "```json",
        json.dumps({
            "class_0_probability": "float in [0, 1]",
            "class_1_probability": "float in [0, 1]",
            "model_version": "random_forest_A_calibrated",
            "calibrated": True,
        }, indent=2),
        "```",
        "",
        "- `class_0_probability + class_1_probability` ~ 1.0 (within numerical tolerance).",
        "- Probabilities are calibrated on validation data; they approximate the",
        "  observed frequency of class_1 outcomes.",
        "- Label semantics remain unconfirmed; interpret `class_1` only after",
        "  authoritative documentation is provided.",
        "",
        "## Not Included",
        "",
        "- risk_level (LOW/MEDIUM/HIGH/CRITICAL)",
        "- confidence score",
        "- accident probability",
        "- traffic signal command",
        "- Member 5 decision",
        "",
        "## Loading Example",
        "",
        "```python",
        "import joblib",
        "pipe = joblib.load('models/near_miss/random_forest_A_calibrated.joblib')",
        "probs = pipe.predict_proba(X)  # shape (n_samples, 2)",
        "```",
        "",
    ]
    (REPORT_DIR / "probability_contract.md").write_text("\n".join(lines), encoding="utf-8")
