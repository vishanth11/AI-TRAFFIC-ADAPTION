"""Phase 4 plotting utilities for near-miss/conflict model calibration.

Contains figure generation for reliability diagrams, precision-recall curves,
threshold trade-offs, and error analysis. Shared styling and the Probs dataclass
live here so that calibrate.py can import them without circular dependencies.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

FIG_DIR = Path("reports/figures/near_miss")

# Reference light-mode palette used in Phase 3.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


@dataclass(frozen=True)
class Probs:
    raw: np.ndarray
    sigmoid: np.ndarray
    isotonic: np.ndarray
    y: np.ndarray
    sample_index: np.ndarray


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_calibration_comparison(val_probs: Probs, test_probs: Probs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    counts = {}
    for ax, probs, title in [
        (ax_val, val_probs, "Validation"),
        (ax_test, test_probs, "Test"),
    ]:
        style_ax(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect")
        colors = {"raw": C1, "sigmoid": C2, "isotonic": C3}
        markers = {"raw": "o", "sigmoid": "s", "isotonic": "^"}
        method_counts = {}
        for label, p in [("raw", probs.raw), ("sigmoid", probs.sigmoid),
                         ("isotonic", probs.isotonic)]:
            prob_true, prob_pred = calibration_curve(probs.y, p, n_bins=10, strategy="uniform")
            ax.plot(prob_pred, prob_true, color=colors[label], marker=markers[label],
                    ms=5, lw=2, label=label)
            bin_edges = np.linspace(0, 1, 11)
            title_counts = []
            for i in range(10):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (p >= lo) & (p < hi) if i < 9 else (p >= lo) & (p <= hi)
                title_counts.append(int(mask.sum()))
            method_counts[label] = title_counts
        ax.set_title(f"Reliability — {title}", color=INK, fontsize=11)
        ax.set_xlabel("predicted probability", color=INK2)
        ax.set_ylabel("observed frequency", color=INK2)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
        counts[title.lower()] = method_counts

    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_comparison.png", facecolor=SURFACE)
    plt.close(fig)
    return counts


def plot_calibrated_pr_curve(val_probs: Probs, test_probs: Probs):
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, probs, title in [
        (ax_val, val_probs, "Validation"),
        (ax_test, test_probs, "Test"),
    ]:
        style_ax(ax)
        prevalence = probs.y.mean()
        ax.axhline(prevalence, color=MUTED, lw=1, ls="--",
                   label=f"prevalence = {prevalence:.2f}")
        colors = {"raw": C1, "sigmoid": C2, "isotonic": C3}
        for label, p in [("raw", probs.raw), ("sigmoid", probs.sigmoid),
                         ("isotonic", probs.isotonic)]:
            pr, rc, _ = precision_recall_curve(probs.y, p)
            ap = average_precision_score(probs.y, p)
            ax.plot(rc, pr, color=colors[label], lw=2,
                    label=f"{label} PR-AUC={ap:.3f}")
        ax.set_title(f"Precision–Recall — {title}", color=INK, fontsize=11)
        ax.set_xlabel("recall (class_1)", color=INK2)
        ax.set_ylabel("precision (class_1)", color=INK2)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibrated_precision_recall.png", facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_tradeoff(threshold_df: pd.DataFrame, method: str = "sigmoid"):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    sub = threshold_df[threshold_df["split"] == "validation"]
    ax.plot(sub["threshold"], sub["precision"], color=C1, lw=2, marker="o",
            ms=5, label="precision")
    ax.plot(sub["threshold"], sub["recall"], color=C2, lw=2, marker="o",
            ms=5, label="recall")
    ax.plot(sub["threshold"], sub["f1"], color=C3, lw=2, marker="o",
            ms=5, label="F1")
    ax.set_xlabel(f"decision threshold ({method} calibrated)", color=INK2)
    ax.set_ylabel("score", color=INK2)
    ax.set_title(f"Calibrated threshold trade-off — random_forest_A (validation, {method})",
                 color=INK, fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibrated_threshold_tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


def plot_error_analysis(fn_df: pd.DataFrame, fp_df: pd.DataFrame, method: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # FN figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        style_ax(ax)

    if not fn_df.empty:
        axes[0, 0].hist(fn_df["predicted_probability"], bins=10, range=(0, 1),
                        color=C2, edgecolor="white")
    axes[0, 0].set_title("False negative probability distribution", color=INK, fontsize=11)
    axes[0, 0].set_xlabel(f"{method} calibrated probability", color=INK2)
    axes[0, 0].set_ylabel("count", color=INK2)

    if "PET" in fn_df.columns:
        axes[0, 1].hist(fn_df["PET"], bins=15, color=C2, edgecolor="white")
    axes[0, 1].set_title("False negatives — PET distribution", color=INK, fontsize=11)
    axes[0, 1].set_xlabel("PET", color=INK2)
    axes[0, 1].set_ylabel("count", color=INK2)

    for ax, df, title, color in [
        (axes[1, 0], fn_df, "False negatives", C2),
    ]:
        if df.empty or "PET" not in df.columns or "min_dist_dual_check_m" not in df.columns:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
            continue
        ax.scatter(df["PET"], df["min_dist_dual_check_m"], color=color, alpha=0.7, s=40)
        ax.set_title(f"{title} — PET vs min distance", color=INK, fontsize=11)
        ax.set_xlabel("PET", color=INK2)
        ax.set_ylabel("min_dist_dual_check_m", color=INK2)

    for ax, df, title, color in [
        (axes[1, 1], fn_df, "False negatives", C2),
    ]:
        if df.empty or "speed_object_1_kph" not in df.columns or "target_dist_px" not in df.columns:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
            continue
        ax.scatter(df["speed_object_1_kph"], df["target_dist_px"], color=color, alpha=0.7, s=40)
        ax.set_title(f"{title} — speed vs target_dist_px", color=INK, fontsize=11)
        ax.set_xlabel("speed_object_1_kph", color=INK2)
        ax.set_ylabel("target_dist_px", color=INK2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "false_negative_analysis.png", facecolor=SURFACE)
    plt.close(fig)

    # FP figure
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    fig2.patch.set_facecolor(SURFACE)
    for ax in axes2.flat:
        style_ax(ax)

    if not fp_df.empty:
        axes2[0, 0].hist(fp_df["predicted_probability"], bins=10, range=(0, 1),
                         color=C1, edgecolor="white")
    axes2[0, 0].set_title("False positive probability distribution", color=INK, fontsize=11)
    axes2[0, 0].set_xlabel(f"{method} calibrated probability", color=INK2)
    axes2[0, 0].set_ylabel("count", color=INK2)

    if "class_object_1" in fp_df.columns:
        vc = fp_df["class_object_1"].value_counts().head(6)
        axes2[0, 1].barh(range(len(vc)), vc.values, color=C1, height=0.62)
        axes2[0, 1].set_yticks(range(len(vc)))
        axes2[0, 1].set_yticklabels(vc.index.astype(str), color=INK2, fontsize=9)
        axes2[0, 1].set_title("False positives — class_object_1", color=INK, fontsize=11)
        axes2[0, 1].set_xlabel("count", color=INK2)

    for ax, df, title, color in [
        (axes2[1, 0], fp_df, "False positives", C1),
    ]:
        if df.empty or "PET" not in df.columns or "min_dist_dual_check_m" not in df.columns:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
            continue
        ax.scatter(df["PET"], df["min_dist_dual_check_m"], color=color, alpha=0.7, s=40)
        ax.set_title(f"{title} — PET vs min distance", color=INK, fontsize=11)
        ax.set_xlabel("PET", color=INK2)
        ax.set_ylabel("min_dist_dual_check_m", color=INK2)

    for ax, df, title, color in [
        (axes2[1, 1], fp_df, "False positives", C1),
    ]:
        if df.empty or "speed_object_1_kph" not in df.columns or "min_dist_dual_check_m" not in df.columns:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
            continue
        ax.scatter(df["speed_object_1_kph"], df["min_dist_dual_check_m"],
                   color=color, alpha=0.7, s=40)
        ax.set_title(f"{title} — speed vs distance", color=INK, fontsize=11)
        ax.set_xlabel("speed_object_1_kph", color=INK2)
        ax.set_ylabel("min_dist_dual_check_m", color=INK2)

    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "false_positive_analysis.png", facecolor=SURFACE)
    plt.close(fig2)
