"""Phase 6 orchestration — accident detector evaluation + confidence calibration.

Pipeline (in this fixed order):
1. Select the best available Phase 5/5B checkpoint on VALIDATION
   (accident AP@50 primary, mAP@50 tie-break).
2. Evaluate the chosen checkpoint on validation (object-level).
3. Derive image/event-level accident signals from accident-class box scores.
4. Study confidence thresholds 0.10..0.90 on validation (accident class).
5. Fit Platt + isotonic calibration on validation image-level labels only;
   compare by Brier, log loss, ECE, and 5-fold CV stability.
6. Freeze calibration, record dataset checksums, and run the ONE test
   evaluation. Nothing is tuned after test results are seen.

No training and no architecture changes happen here.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import (
    SIGNAL_FUNCTIONS,
    SIGNAL_SCORE_FLOOR,
    brier_score,
    build_calibrator,
    cross_validated_metrics,
    expected_calibration_error,
    image_level_labels,
    image_level_signal,
    log_loss,
)
from .dataset import AccidentParquetDataset, collate_detection
from .labels import DATASET_LABEL_ACCIDENT
from .metrics import evaluate_coco, threshold_analysis
from .model import build_detector
from .utils import DATASET_DIR, MODEL_DIR, REPORT_DIR, get_device, seed_everything
from .calibrated_detector import save_calibrated_checkpoint

DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SIGNAL_EPS_TIE = 0.002  # Brier gap below which stability decides the method

# Real counter for the leakage guard: incremented at the single test
# evaluation call site in run_phase6.
TEST_EVALUATIONS = 0

# Checkpoint candidates with their validation history files. Phase 5 baseline
# metrics come from phase5_summary.json; Phase 5B from the per-experiment
# histories. Selection is re-verified by evaluating each candidate on
# validation in this run.
CANDIDATE_CHECKPOINTS = {
    "phase5_baseline": {
        "path": MODEL_DIR / "best_detector.pth",
        "history": REPORT_DIR / "phase5_summary.json",
    },
    "phase5b_expA": {
        "path": MODEL_DIR / "phase5b_expA_best.pth",
        "history": REPORT_DIR / "phase5b_exp_a_history.json",
    },
    "phase5b_expB": {
        "path": MODEL_DIR / "phase5b_expB_best.pth",
        "history": REPORT_DIR / "phase5b_exp_b_history.json",
    },
}

DATASET_FILES = [
    "train-00000-of-00002.parquet",
    "train-00001-of-00002.parquet",
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's SHA-256 so multi-hundred-MB Parquets stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprints(dataset_dir: Path = DATASET_DIR) -> dict[str, str]:
    """SHA-256 of the original Parquet files (integrity guard)."""
    return {
        name: sha256_file(dataset_dir / name)
        for name in DATASET_FILES
        if (dataset_dir / name).exists()
    }


def logged_candidate_metrics() -> dict[str, dict[str, float]]:
    """Best recorded validation accident AP@50 per candidate checkpoint.

    Phase 5B histories use "history" with "val_accident_ap50"; the Phase 5
    summary uses "training_history" with "val_accident_AP".
    """
    out = {}
    for cand_id, cand in CANDIDATE_CHECKPOINTS.items():
        with open(cand["history"], "r", encoding="utf-8") as f:
            history = json.load(f)
        entries = history.get("history") or history.get("training_history") or []
        if not entries:
            raise ValueError(f"No history entries found in {cand['history']}")
        key = "val_accident_ap50" if "val_accident_ap50" in entries[0] else "val_accident_AP"
        best = max(entries, key=lambda e: e[key])
        map_key = "val_map50" if "val_map50" in best else "val_mAP_50"
        if map_key not in best:
            raise KeyError(
                f"No validation mAP key (val_map50/val_mAP_50) in {cand['history']}"
            )
        out[cand_id] = {
            "val_accident_ap50": float(best[key]),
            "val_map50": float(best[map_key]),
        }
    return out


def select_checkpoint(
    candidates: dict[str, dict[str, Any]],
    verified: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Pick the candidate with the best verified validation accident AP@50."""
    chosen_id = max(
        verified, key=lambda cid: (verified[cid]["val_accident_ap50"], verified[cid]["val_map50"])
    )
    return {
        "chosen": chosen_id,
        "checkpoint_path": str(candidates[chosen_id]["path"]),
        "verified_metrics": verified[chosen_id],
        "all_candidates": verified,
        "rule": "max validation accident AP@50; mAP@50 tie-break (Phase 5B spec)",
    }


@torch.no_grad()
def _gather_predictions_mapped(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    shift_labels: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """gather_predictions with a controllable label mapping.

    Phase 5B checkpoints (3-class head) were trained in model label space and
    need the -1 shift back to dataset labels. The Phase 5 baseline (2-class
    head) was trained with UN-shifted dataset labels, so its model labels are
    already dataset labels and must not be shifted.
    """
    from .labels import model_to_dataset_labels

    model.eval()
    gt_records: list[dict[str, Any]] = []
    pred_records: list[dict[str, Any]] = []
    for images, targets in loader:
        images_device = [img.to(device) for img in images]
        outputs = model(images_device)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            gt_records.append(
                {
                    "image_id": image_id,
                    "boxes": target["boxes"].cpu().numpy(),
                    "labels": target["labels"].cpu().numpy(),
                }
            )
            keep = output["scores"].cpu() >= score_threshold
            labels_out = output["labels"].cpu()[keep].numpy()
            if shift_labels:
                labels_out = model_to_dataset_labels(labels_out)
            pred_records.append(
                {
                    "image_id": image_id,
                    "boxes": output["boxes"].cpu()[keep].numpy(),
                    "labels": labels_out,
                    "scores": output["scores"].cpu()[keep].numpy(),
                }
            )
    return gt_records, pred_records


def evaluate_checkpoint_records(
    split: str,
    checkpoint_path: Path,
    image_size: tuple[int, int] = (320, 320),
    batch_size: int = 4,
    score_threshold: float = SIGNAL_SCORE_FLOOR,
    dataset_dir: Path = DATASET_DIR,
) -> dict[str, Any]:
    """Load a checkpoint, run inference on a split, return records + metrics.

    The label mapping is chosen from the head width: 3-class Phase 5B
    checkpoints get the background offset removed; 2-class Phase 5 baseline
    checkpoints keep their raw labels (see `_gather_predictions_mapped`).
    """
    device = get_device()
    ds = AccidentParquetDataset(split=split, target_size=image_size, dataset_dir=dataset_dir)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_detection
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    num_classes = state_dict["roi_heads.box_predictor.cls_score.bias"].shape[0]
    model = build_detector(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    gt_records, pred_records = _gather_predictions_mapped(
        model, loader, device, score_threshold, shift_labels=(num_classes == 3)
    )
    metrics = evaluate_coco(gt_records, pred_records, image_size)
    return {
        "split": split,
        "gt_records": gt_records,
        "pred_records": pred_records,
        "metrics": metrics,
        "num_images": len(ds),
    }


def compare_signals(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Uncalibrated event-level quality of each signal aggregation (validation)."""
    labels = image_level_labels(gt_records, DATASET_LABEL_ACCIDENT)
    rows = []
    for name in SIGNAL_FUNCTIONS:
        signals = image_level_signal(pred_records, signal_name=name)
        rows.append(
            {
                "signal": name,
                "brier_uncalibrated": round(brier_score(labels, signals), 4),
                "log_loss_uncalibrated": round(log_loss(labels, np.clip(signals, 1e-6, 1 - 1e-6)), 4),
            }
        )
    return rows


def select_signal(rows: list[dict[str, Any]]) -> str:
    """Choose the signal aggregation with the lowest uncalibrated Brier score."""
    return min(rows, key=lambda r: r["brier_uncalibrated"])["signal"]


def fit_and_compare_calibration(
    signals: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit Platt + isotonic on validation; compare fit quality and stability."""
    comparison = []
    fitted: dict[str, Any] = {}
    for method in ("platt", "isotonic"):
        cls = build_calibrator(method)
        cal = cls.fit(signals, labels)
        probs = cal.transform(signals)
        cv = cross_validated_metrics(signals, labels, method, n_splits=5, seed=seed)
        fitted[method] = {
            "calibrator": cal,
            "brier": brier_score(labels, probs),
            "log_loss": log_loss(labels, probs),
            "ece": expected_calibration_error(labels, probs),
            "cv": cv,
            "probs": probs,
        }
        comparison.append(
            {
                "method": method,
                "brier": round(fitted[method]["brier"], 4),
                "log_loss": round(fitted[method]["log_loss"], 4),
                "ece": round(fitted[method]["ece"], 4),
                "cv_brier_mean": round(cv["cv_brier_mean"], 4),
                "cv_brier_std": round(cv["cv_brier_std"], 4),
                "cv_log_loss_mean": round(cv["cv_log_loss_mean"], 4),
            }
        )
    # Selection uses the unrounded fitted metrics, not the display rows.
    selected = _select_method(fitted)
    return {"comparison": comparison, "selected_method": selected, "fitted": fitted}


def _select_method(fitted: dict[str, dict[str, Any]]) -> str:
    """Deterministic selection rule: fit Brier first; near-tie -> held-out stability.

    Compares UNROUNDED metrics (rounding is display-only). The tie-break uses
    5-fold CV on the calibration split: lower CV Brier mean first, then lower
    CV log-loss mean (isotonic step functions can score well on in-sample
    Brier yet produce extreme probabilities on held-out folds). Exact
    stability ties go to Platt (smoother, parametric).
    """
    platt, iso = fitted["platt"], fitted["isotonic"]
    gap = platt["brier"] - iso["brier"]
    if abs(gap) > SIGNAL_EPS_TIE:
        return "isotonic" if gap > 0 else "platt"
    for key in ("cv_brier_mean", "cv_log_loss_mean"):
        if iso["cv"][key] < platt["cv"][key] - 1e-6:
            return "isotonic"
        if platt["cv"][key] < iso["cv"][key] - 1e-6:
            return "platt"
    return "platt"


def run_phase6(
    dataset_dir: Path | None = None,
    output_model_dir: Path = MODEL_DIR,
    output_report_dir: Path = REPORT_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full Phase 6 pipeline. See module docstring for the order."""
    global TEST_EVALUATIONS
    dataset_dir = DATASET_DIR if dataset_dir is None else Path(dataset_dir)
    seed_everything(seed)
    started = time.time()

    fingerprints_before = dataset_fingerprints(dataset_dir)

    # 1. Checkpoint selection — logged metrics, then re-verified on validation.
    logged = logged_candidate_metrics()
    evals_by_id = {}
    verified = {}
    for cand_id, cand in CANDIDATE_CHECKPOINTS.items():
        res = evaluate_checkpoint_records("validation", cand["path"])
        evals_by_id[cand_id] = res
        verified[cand_id] = {
            "val_accident_ap50": res["metrics"]["per_class"]["accident"]["AP_50"],
            "val_map50": res["metrics"]["overall"]["mAP_50"],
            "val_accident_recall": res["metrics"]["per_class"]["accident"]["recall"],
        }
    selection = select_checkpoint(CANDIDATE_CHECKPOINTS, verified)
    chosen_id = selection["chosen"]
    val_eval = evals_by_id[chosen_id]

    # 2. Object-level threshold study on validation (accident class).
    threshold_rows = threshold_analysis(
        val_eval["gt_records"],
        val_eval["pred_records"],
        thresholds=DEFAULT_THRESHOLDS,
        class_filter=DATASET_LABEL_ACCIDENT,
    )

    # 3. Event-level signal + labels from the validation predictions.
    labels = image_level_labels(val_eval["gt_records"], DATASET_LABEL_ACCIDENT)
    signal_rows = compare_signals(val_eval["gt_records"], val_eval["pred_records"])
    signal_name = select_signal(signal_rows)
    signals = image_level_signal(val_eval["pred_records"], signal_name=signal_name)

    # 4. Fit + compare calibration methods (validation only).
    cal = fit_and_compare_calibration(signals, labels, seed=seed)
    method = cal["selected_method"]
    chosen_calibrator = cal["fitted"][method]["calibrator"]

    # 5. Freeze: save the calibrated checkpoint + metadata BEFORE touching test.
    # Only 3-class Phase 5B checkpoints may be calibrated: the deployed signal
    # path applies the background-offset mapping, which is only valid for
    # checkpoints trained in model label space.
    head_width = torch.load(
        selection["checkpoint_path"], map_location="cpu", weights_only=True
    )["roi_heads.box_predictor.cls_score.bias"].shape[0]
    if head_width != 3:
        raise ValueError(
            f"Selected checkpoint {selection['checkpoint_path']} has a "
            f"{head_width}-class head; Phase 6 calibration requires the "
            "3-class Phase 5B checkpoints."
        )
    calibration_dict = chosen_calibrator.to_dict()
    calibration_dict["signal"] = signal_name
    calibration_metadata = {
        "phase": 6,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_source": selection["checkpoint_path"],
        "checkpoint_source_id": chosen_id,
        "selection": selection,
        "logged_candidate_metrics": logged,
        "signal": signal_name,
        "calibration_method": method,
        "calibration_params": {
            k: v for k, v in calibration_dict.items() if k != "signal"
        },
        "fit_split": "validation",
        "selection_split": "validation",
        "fit_image_count": int(len(signals)),
        "fit_positive_image_count": int(labels.sum()),
        "validation_fit_metrics": {
            "brier": round(cal["fitted"][method]["brier"], 6),
            "log_loss": round(cal["fitted"][method]["log_loss"], 6),
            "ece": round(cal["fitted"][method]["ece"], 6),
        },
        "calibration_comparison": cal["comparison"],
        "leakage_guard": {
            "test_used_for_calibration": False,
            "test_used_for_threshold_selection": False,
            "test_evaluations_performed": TEST_EVALUATIONS + 1,
        },
        "dataset_sha256": fingerprints_before,
        "score_threshold_for_signal": SIGNAL_SCORE_FLOOR,
        "iou_threshold_for_eval": 0.5,
        "artifact_paths": {
            "calibrated_checkpoint": str(output_model_dir / "calibrated_accident_detector.pth"),
            "test_results": str(output_report_dir / "test_results_phase6.json"),
        },
    }
    ckpt_path = save_calibrated_checkpoint(
        detector_state_dict=torch.load(
            selection["checkpoint_path"], map_location="cpu", weights_only=True
        ),
        calibration_dict=calibration_dict,
        metadata={
            "checkpoint_source": selection["checkpoint_path"],
            "fit_split": "validation",
            "signal": signal_name,
            "fit_image_count": int(len(signals)),
        },
        output_path=output_model_dir / "calibrated_accident_detector.pth",
    )
    meta_path = output_model_dir / "calibration_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(calibration_metadata, f, indent=2)

    print(f"[Phase 6] Calibrated checkpoint saved: {ckpt_path}")

    # 6. ONE final test evaluation with everything frozen.
    TEST_EVALUATIONS += 1
    test_eval = evaluate_checkpoint_records(
        "test", selection["checkpoint_path"], dataset_dir=dataset_dir
    )
    test_labels = image_level_labels(test_eval["gt_records"], DATASET_LABEL_ACCIDENT)
    test_signals = image_level_signal(
        test_eval["pred_records"], signal_name=signal_name
    )
    test_probs = chosen_calibrator.transform(test_signals)
    test_row = threshold_analysis(
        test_eval["gt_records"],
        test_eval["pred_records"],
        thresholds=[0.5],
        class_filter=DATASET_LABEL_ACCIDENT,
    )[0]

    test_summary = {
        "detection_metrics": test_eval["metrics"],
        "num_images": test_eval["num_images"],
        "accident_precision_at_0.5": test_row["precision"],
        "accident_recall_at_0.5": test_row["recall"],
        "event_level": {
            "n_images": int(len(test_labels)),
            "n_positive_images": int(test_labels.sum()),
            "brier": round(brier_score(test_labels, test_probs), 4),
            "log_loss": round(log_loss(test_labels, test_probs), 4),
            "ece": round(expected_calibration_error(test_labels, test_probs), 4),
        },
        "test_signals": test_signals,
        "test_probs": test_probs,
        "test_labels": test_labels,
        "pred_records": test_eval["pred_records"],
        "gt_records": test_eval["gt_records"],
    }

    fingerprints_after = dataset_fingerprints(dataset_dir)
    if fingerprints_after != fingerprints_before:
        raise RuntimeError("Original Parquet files changed during the Phase 6 run.")

    result = {
        "selection": selection,
        "logged_candidate_metrics": logged,
        "validation_eval": {
            "metrics": val_eval["metrics"],
            "num_images": val_eval["num_images"],
            "threshold_rows": threshold_rows,
        },
        "signal_rows": signal_rows,
        "calibration": {
            "signal": signal_name,
            "method": method,
            "comparison": cal["comparison"],
            "labels": labels,
            "signals": signals,
            "probs": cal["fitted"][method]["probs"],
            "probs_by_method": {
                name: entry["probs"] for name, entry in cal["fitted"].items()
            },
            "calibrator": chosen_calibrator,
        },
        "test": test_summary,
        "calibration_metadata": calibration_metadata,
        "paths": {
            "calibrated_checkpoint": str(ckpt_path),
            "calibration_metadata": str(meta_path),
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "dataset_unchanged": True,
    }

    # Persist a machine-readable summary for later phases.
    summary_path = output_report_dir / "phase6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection": selection,
                "logged_candidate_metrics": logged,
                "validation_eval": {"num_images": val_eval["num_images"]},
                "validation_metrics": val_eval["metrics"],
                "threshold_rows": threshold_rows,
                "signal_rows": signal_rows,
                "calibration": {
                    "signal": signal_name,
                    "method": method,
                    "comparison": cal["comparison"],
                    "fit_metrics": calibration_metadata["validation_fit_metrics"],
                },
                "test": {
                    "num_images": test_summary["num_images"],
                    "detection_metrics": test_summary["detection_metrics"],
                    "accident_precision_at_0.5": test_summary["accident_precision_at_0.5"],
                    "accident_recall_at_0.5": test_summary["accident_recall_at_0.5"],
                    "event_level": test_summary["event_level"],
                    # Compact arrays so reports/figures can be regenerated
                    # without re-running inference.
                    "arrays": {
                        "labels": [float(v) for v in test_labels],
                        "signals": [round(float(v), 6) for v in test_signals],
                        "probs": [round(float(v), 6) for v in test_probs],
                    },
                },
                "validation_arrays": {
                    "labels": [int(v) for v in labels],
                    "signals": [round(float(v), 6) for v in signals],
                    "probs_by_method": {
                        name: [round(float(v), 6) for v in entry["probs"]]
                        for name, entry in cal["fitted"].items()
                    },
                },
                "paths": result["paths"],
                "leakage_guard": calibration_metadata["leakage_guard"],
                "elapsed_seconds": result["elapsed_seconds"],
                "dataset_unchanged": True,
            },
            f,
            indent=2,
        )
    print(f"[Phase 6] complete in {result['elapsed_seconds']:.0f}s")
    return result


if __name__ == "__main__":
    result = run_phase6()
    print(json.dumps(result["selection"], indent=2))