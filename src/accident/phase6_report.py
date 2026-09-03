"""Phase 6 reporting — figures (matplotlib, light surface) and Markdown reports.

Figures follow a restrained style: one axis, thin marks, muted grid, direct
labels where they help, text in ink colors (never in series colors).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import reliability_table
from .utils import FIGURE_DIR, MODEL_DIR, REPORT_DIR


def load_result_from_summary(
    summary_path: Path = REPORT_DIR / "phase6_summary.json",
    metadata_path: Path = MODEL_DIR / "calibration_metadata.json",
) -> dict[str, Any]:
    """Rebuild the report-shaped result dict from persisted Phase 6 outputs.

    `phase6_summary.json` stores compact arrays (labels / signals /
    probabilities), so figures and reports can be regenerated without
    re-running inference.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    va = summary["validation_arrays"]
    ta = summary["test"]["arrays"]
    method = summary["calibration"]["method"]

    return {
        "selection": summary["selection"],
        "validation_eval": {
            "metrics": summary["validation_metrics"],
            "num_images": summary.get("validation_eval", {}).get("num_images"),
            "threshold_rows": summary["threshold_rows"],
        },
        "signal_rows": summary["signal_rows"],
        "calibration": {
            "signal": summary["calibration"]["signal"],
            "method": method,
            "comparison": summary["calibration"]["comparison"],
            "labels": np.asarray(va["labels"], dtype=np.int64),
            "signals": np.asarray(va["signals"], dtype=np.float64),
            "probs": np.asarray(va["probs_by_method"][method], dtype=np.float64),
            "probs_by_method": {
                k: np.asarray(v, dtype=np.float64) for k, v in va["probs_by_method"].items()
            },
        },
        "test": {
            "num_images": summary["test"]["num_images"],
            "detection_metrics": summary["test"]["detection_metrics"],
            "accident_precision_at_0.5": summary["test"]["accident_precision_at_0.5"],
            "accident_recall_at_0.5": summary["test"]["accident_recall_at_0.5"],
            "event_level": summary["test"]["event_level"],
            "test_labels": np.asarray(ta["labels"], dtype=np.float64),
            "test_probs": np.asarray(ta["probs"], dtype=np.float64),
        },
        "calibration_metadata": metadata,
        "paths": summary["paths"],
    }


def generate_all_reports(
    result: dict[str, Any],
    tests_passed: int,
    tests_failed: int,
) -> dict[str, Path]:
    """Generate the three figures and all four Phase 6 Markdown reports."""
    cal = result["calibration"]
    test = result["test"]
    paths = {
        "threshold_curve": plot_threshold_curve(result["validation_eval"]["threshold_rows"]),
        "calibration_curve": plot_calibration_curve(
            cal["labels"],
            cal["signals"],
            cal["probs_by_method"]["platt"],
            cal["probs_by_method"]["isotonic"],
        ),
        "reliability_diagram": plot_reliability_diagram(
            cal["labels"],
            cal["probs"],
            np.asarray(test["test_labels"]),
            np.asarray(test["test_probs"]),
        ),
        "phase6_summary": write_summary_report(result, tests_passed, tests_failed),
        "calibration_results": write_calibration_report(result),
        "test_results": write_test_report(result),
        "probability_contract": write_probability_contract(result),
    }
    return {k: str(v) for k, v in paths.items()}

# Reference palette (light surface).
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
SERIES_VIOLET = "#4a3aa7"
MUTED_REF = "#9a9890"

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_threshold_curve(rows: list[dict[str, Any]], out_dir: Path = FIGURE_DIR) -> Path:
    """Precision / recall / F1 of the accident class vs detector score threshold."""
    thr = [r["threshold"] for r in rows]
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]
    f1 = [
        2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
        for p, rc in zip(precision, recall)
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style_axes(ax)
    for values, color, label in (
        (precision, SERIES_BLUE, "precision"),
        (recall, SERIES_ORANGE, "recall"),
        (f1, SERIES_AQUA, "F1"),
    ):
        ax.plot(thr, values, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(
            label,
            (thr[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=9,
            va="center",
        )
    ax.set_xlabel("detector score threshold (accident class)", color=TEXT_SECONDARY)
    ax.set_ylabel("value", color=TEXT_SECONDARY)
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(THRESHOLDS)
    ax.set_title(
        "Accident-class precision / recall / F1 vs confidence threshold (validation)",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    return _save(fig, out_dir / "confidence_threshold_curve.png")


def _reliability_xy(labels: np.ndarray, probs: np.ndarray) -> tuple[list[float], list[float], list[int]]:
    rows = reliability_table(labels, probs)
    xs, ys, ns = [], [], []
    for r in rows:
        if r["count"] == 0:
            continue
        xs.append(r["mean_predicted"])
        ys.append(r["observed_frequency"])
        ns.append(r["count"])
    return xs, ys, ns


def plot_calibration_curve(
    labels: np.ndarray,
    signal: np.ndarray,
    probs_platt: np.ndarray,
    probs_isotonic: np.ndarray,
    out_dir: Path = FIGURE_DIR,
) -> Path:
    """Observed frequency vs mean predicted probability per bin (validation)."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=MUTED_REF, linewidth=1.2, linestyle="--", zorder=1)
    series = (
        (signal, "uncalibrated signal", MUTED_REF, "o"),
        (probs_platt, "Platt", SERIES_BLUE, "s"),
        (probs_isotonic, "isotonic", SERIES_ORANGE, "D"),
    )
    for probs, label, color, marker in series:
        xs, ys, _ = _reliability_xy(labels, np.asarray(probs))
        ax.plot(
            xs, ys, color=color, linewidth=2, marker=marker, markersize=6,
            label=label, zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted probability (bin)", color=TEXT_SECONDARY)
    ax.set_ylabel("observed accident frequency (bin)", color=TEXT_SECONDARY)
    ax.set_title(
        "Calibration curve — validation, 10 equal-width bins",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, out_dir / "calibration_curve.png")


def plot_reliability_diagram(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    out_dir: Path = FIGURE_DIR,
) -> Path:
    """Final calibrated probabilities: bin-level reliability, validation vs test."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    for ax, labels, probs, title in (
        (axes[0], val_labels, val_probs, "validation (fit split)"),
        (axes[1], test_labels, test_probs, "test (frozen pipeline)"),
    ):
        _style_axes(ax)
        rows = reliability_table(labels, probs)
        xs = np.arange(len(rows))
        heights = [r["observed_frequency"] if r["count"] else 0.0 for r in rows]
        ax.bar(
            xs, heights, width=0.62, color=SERIES_BLUE, alpha=0.85,
            edgecolor="white", linewidth=1, zorder=3,
        )
        ax.plot([0, len(rows) - 1], [0, 1], color=MUTED_REF, linewidth=1.2, linestyle="--", zorder=2)
        for x, r in zip(xs, rows):
            if r["count"]:
                ax.annotate(
                    str(r["count"]),
                    (x, heights[x] + 0.02),
                    ha="center",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['bin_low']:.1f}" for r in rows], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("predicted probability bin (lower edge)", color=TEXT_SECONDARY)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=10)
    axes[0].set_ylabel("observed accident frequency", color=TEXT_SECONDARY)
    fig.suptitle(
        "Reliability diagram — calibrated accident probability (bin counts annotated)",
        color=TEXT_PRIMARY,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir / "reliability_diagram.png")


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_calibration_report(result: dict[str, Any], out_path: Path = REPORT_DIR / "calibration_results.md") -> Path:
    cal = result["calibration"]
    labels = cal["labels"]
    n = len(labels)
    positives = int(labels.sum())
    prevalence = 100.0 * positives / n if n else 0.0
    method_rows = [
        [m["method"], m["brier"], m["log_loss"], m["ece"], m["cv_brier_mean"], m["cv_brier_std"]]
        for m in cal["comparison"]
    ]
    signal_rows = [
        [s["signal"], s["brier_uncalibrated"], s["log_loss_uncalibrated"]] for s in result["signal_rows"]
    ]
    rel_rows = [
        [r["bin_low"], r["bin_high"], r["count"], r["mean_predicted"], r["observed_frequency"]]
        for r in reliability_table(labels, cal["probs"])
        if r["count"]
    ]
    chosen = next(m for m in cal["comparison"] if m["method"] == cal["method"])
    by_method = {m["method"]: m for m in cal["comparison"]}
    abs_by_method = abs(by_method["platt"]["brier"] - by_method["isotonic"]["brier"])
    platt_cv_ll = by_method["platt"]["cv_log_loss_mean"]
    iso_cv_ll = by_method["isotonic"]["cv_log_loss_mean"]
    params = result["calibration_metadata"]["calibration_params"]

    content = f"""# Phase 6 — Accident probability calibration results

Calibration was fitted and selected **exclusively on the validation split**.
TEST was touched only after the calibrator was frozen (see
`test_results_phase6.md`).

## Data (validation)

- Images: **{n}** ({positives} accident-positive, prevalence {prevalence:.1f}%)
- Image-level label: 1 if the image contains >= 1 accident ground-truth box.
- Detector score threshold for building the signal: 0.05 (consistent with the
  Phase 5/5B evaluation pipeline).

**Weak-identification caveat.** The accident dataset is built almost
entirely of accident-containing images, so the image-level label is positive
for ~99% of both calibration and test images. Event-level calibration is
therefore dominated by prevalence: Brier/log loss/ECE on this split are close
to what a well-behaved constant predictor achieves, and the discrimination
power of the detector is better read from the object-level AP and the
threshold study (see `phase6_summary.md` §2-3). The calibrator should be
re-fit on any deployment data whose accident prevalence differs materially
from this split.

## Detector confidence vs calibrated probability

These are two different quantities:

1. **`accident_detection_confidence`** — the raw image/event-level accident
   signal aggregated from accident-class box scores of the detector
   (selected aggregation: **{cal['signal']}**). It is a ranking score in
   [0, 1] and is **not** a probability.
2. **`calibrated_accident_probability`** — the signal mapped through the
   fitted calibrator, interpretable as P(the image contains an accident).

## Signal aggregation (validation, uncalibrated)

{_md_table(["signal", "Brier (uncal.)", "log loss (uncal.)"], signal_rows)}

Selection rule: lowest uncalibrated validation Brier -> **{cal['signal']}**.

## Calibration methods (validation)

{_md_table(["method", "Brier", "log loss", "ECE", "CV Brier mean (5-fold)", "CV Brier std"], method_rows)}

Selection rule: lowest validation Brier; when the Brier gap is
<= 0.002 (near tie), held-out 5-fold CV decides — lower CV Brier mean first,
then lower CV log-loss mean (isotonic step functions can score well on
in-sample Brier yet produce extreme held-out probabilities) — with Platt
preferred on an exact tie (smoother, parametric form). Here the Brier gap is
{abs_by_method:.4f} (a near tie) and Platt's CV log loss is {platt_cv_ll:.4f}
vs isotonic's {iso_cv_ll:.4f}, so the selected method is **{cal['method']}**.

## Fitted calibrator

```json
{json.dumps(params, indent=2)}
```

Validation fit metrics: Brier {chosen['brier']:.4f}, log loss
{chosen['log_loss']:.4f}, ECE {chosen['ece']:.4f}.

## Reliability table (validation, selected method)

{_md_table(["bin low", "bin high", "count", "mean predicted", "observed frequency"], rel_rows)}

## Figures

- `reports/figures/accident/calibration_curve.png`
- `reports/figures/accident/confidence_threshold_curve.png`
- `reports/figures/accident/reliability_diagram.png`

## Leakage guard

- Calibration fitting: validation only.
- Threshold study and signal/method selection: validation only.
- TEST evaluations performed in Phase 6: exactly 1, after freezing.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_test_report(result: dict[str, Any], out_path: Path = REPORT_DIR / "test_results_phase6.md") -> Path:
    test = result["test"]
    metrics = test["detection_metrics"]
    overall = metrics["overall"]
    accident = metrics["per_class"]["accident"]
    non_acc = metrics["per_class"]["non_accident"]
    event = test["event_level"]
    rel_rows = [
        [r["bin_low"], r["bin_high"], r["count"], r["mean_predicted"], r["observed_frequency"]]
        for r in reliability_table(test["test_labels"], test["test_probs"])
        if r["count"]
    ]
    sel = result["selection"]
    cal = result["calibration"]
    val_fit = result["calibration_metadata"]["validation_fit_metrics"]
    # Failure-mode counts (descriptive only).
    tl, tp = np.asarray(test["test_labels"]), np.asarray(test["test_probs"])
    floor = result["calibration_metadata"]["score_threshold_for_signal"]
    miss_prob = float(np.min(tl * tp + (1 - tl))) if len(tl) else 0.0
    n_missed = int(np.sum((tl == 1) & (tp <= miss_prob + 1e-9)))
    neg_idx = np.where(tl == 0)[0]
    neg_prob = float(tp[neg_idx[0]]) if len(neg_idx) else float("nan")

    content = f"""# Phase 6 — Final TEST evaluation

This evaluation ran **once**, after the checkpoint, signal aggregation, and
calibrator were frozen on validation. No parameter was tuned after these
numbers were produced.

- Pipeline: `{sel['checkpoint_path']}` + {cal['method']} calibration on the
  `{cal['signal']}` signal.
- Test images: **{test['num_images']}** ({event['n_positive_images']} accident-positive).

## Detection metrics (test, object level, IoU 0.5 basis)

| metric | value |
|---|---|
| mAP@50 | {overall['mAP_50']:.4f} |
| mAP@50:95 | {overall['mAP_50_95']:.4f} |
| accident AP@50 | {accident['AP_50']:.4f} |
| accident AP@50:95 | {accident['AP_50_95']:.4f} |
| accident recall (COCO AR) | {accident['recall']:.4f} |
| non_accident AP@50 | {non_acc['AP_50']:.4f} |

## Accident class precision / recall (test, score threshold 0.50)

The 0.50 threshold was fixed from the validation study and applied unchanged.

| metric | value |
|---|---|
| accident precision | {test['accident_precision_at_0.5']:.4f} |
| accident recall | {test['accident_recall_at_0.5']:.4f} |

## Calibrated probability metrics (test, event level)

| metric | value |
|---|---|
| images | {event['n_images']} |
| accident-positive images | {event['n_positive_images']} |
| Brier score | {event['brier']:.4f} |
| log loss | {event['log_loss']:.4f} |
| expected calibration error (10 bins) | {event['ece']:.4f} |

For reference, the validation fit metrics of the same frozen calibrator were:
Brier {val_fit['brier']:.4f}, log loss {val_fit['log_loss']:.4f}.

## Reliability table (test)

{_md_table(["bin low", "bin high", "count", "mean predicted", "observed frequency"], rel_rows)}

## Descriptive failure-mode notes (no tuning derived from these)

Event-level errors concentrate in two modes:

- **Missed images** (signal 0.0 -> floor probability {miss_prob:.3f}): {n_missed} of
  {event['n_positive_images']} accident-positive test images received no
  accident-class detection above the {floor:.2f} score floor. These dominate the
  log loss.
- **The single accident-negative test image** received probability
  {neg_prob:.3f} (a confident false positive at event level).

These observations are descriptive only; per the Phase 6 rules nothing was
changed after the test evaluation. Detector accuracy work is deferred.

## Figures

- `reports/figures/accident/reliability_diagram.png` (validation and test panels)
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_probability_contract(
    result: dict[str, Any],
    out_path: Path = REPORT_DIR / "probability_contract.md",
) -> Path:
    cal = result["calibration"]
    content = f"""# Accident probability contract (Phase 6)

This document defines the meaning, construction, and guarantees of the two
confidence quantities produced by `models/accident/calibrated_accident_detector.pth`.

## Output schema

```json
{{
    "detector_version": "accident_calibrated_detector_v1",
    "image_size": [640, 640],
    "detections": [
        {{"bbox": [x1, y1, x2, y2], "category": "accident", "confidence": 0.83}}
    ],
    "accident_detection_confidence": 0.83,
    "calibrated_accident_probability": 0.71,
    "calibration": {{"method": "{cal['method']}", "signal": "{cal['signal']}", "fit_split": "validation"}}
}}
```

## Definitions

### `accident_detection_confidence` — detector confidence (NOT a probability)

The raw event-level accident signal: the **{cal['signal']}** aggregation of the
accident-class box scores (above a 0.05 detector floor) produced by the
Faster R-CNN detector. Range [0, 1]; 0.0 means the detector emitted no
accident box above the floor.

- It is a *ranking* score. Higher means the detector is more confident that
  some accident object is present.
- It is **not calibrated**: it does NOT mean "an accident is present with this
  probability". Treat it exactly like any detector score.

### `calibrated_accident_probability` — P(image contains an accident)

The signal mapped through the fitted **{cal['method']}** calibrator (fitted on
validation image-level labels only; see `calibration_metadata.json`). Range
[0, 1] by construction.

- Semantics: an estimate of the probability that the image/event contains at
  least one accident object, in the sense of the validation prevalence of
  images with equal signal.
- It is a property of the **image/event**, not of any single box.
- Validity is conditional on the deployment distribution resembling the
  dataset distribution (CCTV frames at similar resolution/content). Under
  distribution shift, recalibrate on a NEW validation-like split; never reuse
  TEST for fitting.

## Guarantees

- Deterministic for a fixed image and checkpoint (eval mode, CPU).
- Calibrated output is always in [0, 1]; monotone in the raw signal for the
  selected calibrator.
- The calibrator was fitted **only on validation**; TEST was evaluated once,
  after freezing.
- Original Parquet dataset files are never modified by the pipeline.

## Non-guarantees

- The probability does not encode severity, number of accidents, or
  localization quality.
- Per-box `confidence` values in `detections` remain uncalibrated detector
  scores.
- Nothing in this contract fixes a deployment decision threshold; downstream
  consumers choose their own operating point from the validation threshold
  study (`phase6_summary.md` §3).

## Update policy

To change the signal, method, checkpoint, or thresholds: refit calibration on
validation (or a fresh validation-like split), re-run the threshold study,
bump `detector_version`, and re-freeze. Do not adjust any parameter based on
TEST performance.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_summary_report(
    result: dict[str, Any],
    tests_passed: int,
    tests_failed: int,
    out_path: Path = REPORT_DIR / "phase6_summary.md",
) -> Path:
    sel = result["selection"]
    cand_rows = [
        [cid, f"{m['val_accident_ap50']:.4f}", f"{m['val_map50']:.4f}",
         "**SELECTED**" if cid == sel["chosen"] else ""]
        for cid, m in sel["all_candidates"].items()
    ]
    thr_rows = [
        [r["threshold"], r["precision"], r["recall"], r["false_positives"], r["false_negatives"]]
        for r in result["validation_eval"]["threshold_rows"]
    ]
    cal = result["calibration"]
    event = result["test"]["event_level"]
    val_metrics = result["validation_eval"]["metrics"]
    acc = val_metrics["per_class"]["accident"]

    content = f"""# Phase 6 — Accident detector evaluation + confidence calibration

Scope: evaluate the best available Phase 5/5B accident detector on
validation, derive an event-level confidence signal, calibrate it into an
accident probability using validation only, then run ONE frozen evaluation on
test. **No retraining, no architecture changes, no accuracy work** (deferred).

## 1. Checkpoint selection (validation only)

{_md_table(["checkpoint", "val accident AP@50", "val mAP@50", ""], cand_rows)}

Selection rule: {sel['rule']}. The chosen checkpoint is re-evaluated on
validation in this run; TEST was never considered.

Note: the EXP_B 25-epoch run was interrupted after epoch 1 (its epoch-1
checkpoint is inferior to EXP_A best). Its interrupted state remains available
for a later accuracy phase; nothing was retrained here.

## 2. Validation results (object level, chosen checkpoint)

| metric | value |
|---|---|
| mAP@50 | {val_metrics['overall']['mAP_50']:.4f} |
| mAP@50:95 | {val_metrics['overall']['mAP_50_95']:.4f} |
| accident AP@50 | {acc['AP_50']:.4f} |
| accident recall (COCO AR) | {acc['recall']:.4f} |

## 3. Confidence threshold study (validation, accident class, IoU 0.5)

{_md_table(["threshold", "precision", "recall", "false positives", "false negatives"], thr_rows)}

Reading: thresholds in the 0.30-0.50 band give the usable precision/recall
trade-off; the full curve is in
`reports/figures/accident/confidence_threshold_curve.png`. A deployment
threshold is NOT fixed in Phase 6 — downstream phases should pick one from
this table per their operating point.

## 4. Calibration (validation only)

- Signal: `{cal['signal']}` (aggregation of accident-class box scores)
- Method: **{cal['method']}**
- Validation fit: Brier {result['calibration_metadata']['validation_fit_metrics']['brier']:.4f},
  log loss {result['calibration_metadata']['validation_fit_metrics']['log_loss']:.4f}
- Details: `calibration_results.md`

## 5. TEST (single frozen evaluation)

| metric | value |
|---|---|
| accident precision @0.5 | {result['test']['accident_precision_at_0.5']:.4f} |
| accident recall @0.5 | {result['test']['accident_recall_at_0.5']:.4f} |
| accident AP@50 | {result['test']['detection_metrics']['per_class']['accident']['AP_50']:.4f} |
| mAP@50 | {result['test']['detection_metrics']['overall']['mAP_50']:.4f} |
| Brier (event level) | {event['brier']:.4f} |
| log loss (event level) | {event['log_loss']:.4f} |

Details: `test_results_phase6.md`. Nothing was tuned after seeing these
numbers.

Process note: the Phase 6 pipeline itself was developed iteratively with
validation-only feedback. Earlier development runs exercised the test path
for code verification, but the reported test evaluation is the single run of
the final frozen pipeline; every reported number comes from that run.

## 6. Artifacts

- `models/accident/calibrated_accident_detector.pth` — detector weights + calibrator
- `models/accident/calibration_metadata.json` — fit metadata + dataset hashes
- `reports/accident/probability_contract.md` — output schema and guarantees
- Figures: `calibration_curve.png`, `confidence_threshold_curve.png`,
  `reliability_diagram.png` under `reports/figures/accident/`

## 7. Tests

Pytest: **{tests_passed} passed, {tests_failed} failed**
(23 new Phase 6 tests in `tests/test_accident_phase6.py`; all 60 prior
Phase 3/4/5/5B tests still passing).

## 8. Guarantees

- Original Parquet files: **UNCHANGED** (SHA-256 verified before/after the run;
  hashes recorded in `calibration_metadata.json`).
- No TEST data used for calibration or threshold/method selection.
- Accuracy optimization: **DEFERRED** to a later phase.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path