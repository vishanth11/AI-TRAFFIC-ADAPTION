"""Phase 3 — near-miss model evaluation artifacts.

Threshold analysis (validation), ROC/PR curves, confusion-matrix figures,
feature importance, and the evaluation + leakage-review markdown reports.

Charts follow the validated reference dataviz palette (light mode):
series slots blue/orange/aqua, hairline grid, ink-toned text.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, average_precision_score,
)

RANDOM_SEED = 42
SPLIT_DIR = Path("data/processed/near_miss/splits")
MODEL_DIR = Path("models/near_miss")
REPORT_DIR = Path("reports/near_miss")
FIG_DIR = Path("reports/figures/near_miss")

# reference palette (light mode) + chart chrome
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

TARGET = "event_type"
PRIMARY_A = "random_forest_A"    # best val ROC/PR among config A
SECONDARY_A = "logistic_regression_A"  # highest val recall among config A


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_all():
    parts = {s: pd.read_parquet(SPLIT_DIR / f"{s}.parquet")
             for s in ["train", "validation", "test"]}
    return parts


def get_probs(parts, run_name, split):
    pipe = joblib.load(MODEL_DIR / f"{run_name}.joblib")
    feats = [c for c in parts[split].columns
             if c not in ("sample_index", TARGET, "split")]
    prob = pipe.predict_proba(parts[split][feats])[:, 1]
    return prob, parts[split][TARGET].values


def threshold_analysis(parts) -> pd.DataFrame:
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for run in [PRIMARY_A, SECONDARY_A]:
        prob, y = get_probs(parts, run, "validation")
        for t in thresholds:
            pred = (prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append({
                "model": run, "threshold": t,
                "precision": round(precision_score(y, pred, zero_division=0), 4),
                "recall": round(recall_score(y, pred, zero_division=0), 4),
                "f1": round(f1_score(y, pred, zero_division=0), 4),
                "false_positives": int(fp), "false_negatives": int(fn),
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    return df


def plot_threshold(df):
    sub = df[df["model"] == PRIMARY_A]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.plot(sub["threshold"], sub["precision"], color=C1, lw=2, marker="o",
            ms=5, label="precision")
    ax.plot(sub["threshold"], sub["recall"], color=C2, lw=2, marker="o",
            ms=5, label="recall (class_1)")
    ax.plot(sub["threshold"], sub["f1"], color=C3, lw=2, marker="o",
            ms=5, label="F1 (class_1)")
    ax.set_xlabel("decision threshold", color=INK2)
    ax.set_ylabel("score", color=INK2)
    ax.set_title(f"Threshold trade-off — {PRIMARY_A} (validation)",
                 color=INK, fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_analysis.png", facecolor=SURFACE)
    plt.close(fig)


def plot_curves(parts, run_name):
    prob, y = get_probs(parts, run_name, "validation")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    style_ax(ax1)
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax1.plot(fpr, tpr, color=C1, lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax1.set_xlabel("false positive rate", color=INK2)
    ax1.set_ylabel("true positive rate", color=INK2)
    ax1.set_title(f"ROC — {run_name} (validation)", color=INK, fontsize=11)
    ax1.legend(frameon=False, labelcolor=INK2, fontsize=9)

    pr, rc, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    prev = y.mean()
    style_ax(ax2)
    ax2.axhline(prev, color=MUTED, lw=1, ls="--", label=f"prevalence = {prev:.2f}")
    ax2.plot(rc, pr, color=C1, lw=2, label=f"PR-AUC = {ap:.3f}")
    ax2.set_xlabel("recall (class_1)", color=INK2)
    ax2.set_ylabel("precision (class_1)", color=INK2)
    ax2.set_title(f"Precision–Recall — {run_name} (validation)", color=INK, fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, labelcolor=INK2, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_pr_curve_{run_name}.png", facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(parts, run_name, split="test"):
    prob, y = get_probs(parts, run_name, split)
    cm = confusion_matrix(y, (prob >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    # sequential blue ramp for magnitude (counts); text ink swaps for contrast
    norm = cm / cm.max()
    for i in range(2):
        for j in range(2):
            step = SEQ[min(int(norm[i, j] * (len(SEQ) - 1)), len(SEQ) - 1)]
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.96, 0.96, color=step))
            txt_color = "white" if norm[i, j] > 0.45 else INK
            ax.text(j + 0.48, 1 - i + 0.42, str(cm[i, j]), ha="center",
                    va="center", fontsize=16, color=txt_color, fontweight="bold")
            labels = [["TN", "FP"], ["FN", "TP"]]
            ax.text(j + 0.48, 1 - i + 0.72, labels[i][j], ha="center", va="center",
                    fontsize=8, color=txt_color)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["pred class_0", "pred class_1"], color=INK2, fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["actual class_1", "actual class_0"], color=INK2, fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"Confusion matrix — {run_name} ({split}, t=0.50)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_matrix_{run_name}_{split}.png", facecolor=SURFACE)
    plt.close(fig)
    return cm


def feature_importance(parts) -> pd.DataFrame:
    rows = []
    # RF impurity importances (config A)
    pipe = joblib.load(MODEL_DIR / f"{PRIMARY_A}.joblib")
    pre = pipe.named_steps["preprocess"]
    feats = [c for c in parts["train"].columns
             if c not in ("sample_index", TARGET, "split")]
    names = pre.get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    for n, v in sorted(zip(names, imp), key=lambda x: -x[1]):
        rows.append({"model": "random_forest_A", "feature": n,
                     "importance": round(float(v), 5)})
    # LR coefficients (config A)
    pipe_lr = joblib.load(MODEL_DIR / f"{SECONDARY_A}.joblib")
    coefs = pipe_lr.named_steps["model"].coef_[0]
    names_lr = pipe_lr.named_steps["preprocess"].get_feature_names_out()
    for n, v in sorted(zip(names_lr, coefs), key=lambda x: -abs(x[1])):
        rows.append({"model": "logistic_regression_A", "feature": n,
                     "importance": round(float(v), 5)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    return df


def plot_importance(df):
    top = df[df["model"] == "random_forest_A"].head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(range(len(top)), top["importance"], color=C1, height=0.62)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], color=INK2, fontsize=9)
    ax.set_xlabel("RF impurity importance (associated with predictions, "
                  "not causation)", color=INK2, fontsize=9)
    ax.set_title("Feature importance — random_forest_A (train fit)",
                 color=INK, fontsize=11)
    for i, v in enumerate(top["importance"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_random_forest_A.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_all()

    thr = threshold_analysis(parts)
    plot_threshold(thr)
    plot_curves(parts, PRIMARY_A)
    plot_curves(parts, SECONDARY_A)
    cm = plot_confusion(parts, PRIMARY_A, "test")
    imp = feature_importance(parts)
    plot_importance(imp)
    print(f"threshold rows: {len(thr)}; primary test confusion: {cm.tolist()}")
    print("figures:", sorted(p.name for p in FIG_DIR.iterdir()))


if __name__ == "__main__":
    main()