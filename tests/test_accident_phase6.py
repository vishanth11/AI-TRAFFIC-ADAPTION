"""Phase 6 tests — calibration pipeline, inference contract, integrity guards.

Heavy tests (real checkpoint inference) are marked `slow`; the full phase 6
artifact set must exist for them to run (run the phase 6 pipeline first).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from accident.calibration import (
    PlattCalibrator,
    IsotonicCalibrator,
    brier_score,
    build_calibrator,
    calibrator_from_dict,
    expected_calibration_error,
    image_level_labels,
    image_level_signal,
    log_loss,
)
from accident.calibrated_detector import (
    CALIBRATED_CHECKPOINT_PATH,
    CalibratedAccidentDetector,
    save_calibrated_checkpoint,
)
from accident.metrics import threshold_analysis
from accident.model import build_detector
from accident.phase6 import _select_method, select_signal, sha256_file
from accident.utils import DATASET_DIR, MODEL_DIR, REPORT_DIR

METADATA_PATH = MODEL_DIR / "calibration_metadata.json"

REQUIRED_OUTPUT_KEYS = {
    "detector_version",
    "image_size",
    "detections",
    "accident_detection_confidence",
    "calibrated_accident_probability",
}

DATASET_FILES = [
    "train-00000-of-00002.parquet",
    "train-00001-of-00002.parquet",
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
]


@pytest.fixture(scope="module")
def calibration_metadata() -> dict:
    if not METADATA_PATH.exists():
        pytest.skip("Phase 6 artifacts not built yet — run the Phase 6 pipeline.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def detector() -> CalibratedAccidentDetector:
    if not CALIBRATED_CHECKPOINT_PATH.exists():
        pytest.skip("Phase 6 artifacts not built yet — run the Phase 6 pipeline.")
    return CalibratedAccidentDetector()


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.rand(320, 320, 3) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Unit tests for calibration math (no artifacts required)
# ---------------------------------------------------------------------------

def test_platt_fit_outputs_probability_range():
    rng = np.random.RandomState(0)
    signals = rng.rand(200)
    labels = (signals > 0.6).astype(int)
    cal = PlattCalibrator.fit(signals, labels)
    probs = cal.transform(np.linspace(0.0, 1.0, 101))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_isotonic_fit_outputs_probability_range_and_is_monotone():
    rng = np.random.RandomState(1)
    signals = rng.rand(200)
    labels = (signals > 0.5).astype(int)
    cal = IsotonicCalibrator.fit(signals, labels)
    grid = np.linspace(0.0, 1.0, 101)
    probs = cal.transform(grid)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.all(np.diff(probs) >= -1e-9)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_serialization_round_trip(method):
    rng = np.random.RandomState(2)
    signals = rng.rand(150)
    labels = (signals > 0.4).astype(int)
    original = build_calibrator(method).fit(signals, labels)
    rebuilt = calibrator_from_dict(original.to_dict())
    grid = np.linspace(0, 1, 21)
    np.testing.assert_allclose(original.transform(grid), rebuilt.transform(grid), atol=1e-9)


def test_brier_and_log_loss_are_correct_on_known_case():
    labels = np.array([1.0, 0.0])
    probs = np.array([0.8, 0.2])
    assert brier_score(labels, probs) == pytest.approx(0.04)
    expected = -(0.5 * np.log(0.8) + 0.5 * np.log(0.8))
    assert log_loss(labels, probs) == pytest.approx(expected, rel=1e-6)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    labels = np.array([1.0] * 50 + [0.0] * 50)
    probs = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(labels, probs) == pytest.approx(0.0)


def test_image_level_signal_max_and_labels():
    gt = [{"labels": np.array([0, 1])}, {"labels": np.array([1])}]
    pred = [
        {"labels": np.array([0, 0]), "scores": np.array([0.2, 0.9])},
        {"labels": np.array([1]), "scores": np.array([0.7])},
    ]
    signals = image_level_signal(pred, signal_name="max")
    np.testing.assert_allclose(signals, [0.9, 0.0])
    np.testing.assert_array_equal(image_level_labels(gt), [1, 0])


def test_unknown_signal_raises():
    with pytest.raises(KeyError):
        image_level_signal([{"labels": np.array([0]), "scores": np.array([0.5])}], "bogus")


# ---------------------------------------------------------------------------
# Artifact-based tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_calibrated_checkpoint_loading(calibration_metadata):
    assert CALIBRATED_CHECKPOINT_PATH.exists()
    payload = torch.load(CALIBRATED_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert "model_state_dict" in payload and "calibration" in payload
    # The bundled weights must fit the standard 3-class detector.
    model = build_detector(num_classes=3, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    head = payload["model_state_dict"]["roi_heads.box_predictor.cls_score.bias"]
    assert head.shape[0] == 3
    assert payload["calibration"]["method"] == calibration_metadata["calibration_method"]


def test_calibration_metadata_loading(calibration_metadata):
    assert calibration_metadata["fit_split"] == "validation"
    assert calibration_metadata["selection_split"] == "validation"
    assert calibration_metadata["calibration_method"] in ("platt", "isotonic")
    assert calibration_metadata["signal"] in ("max", "top3_mean", "noisy_or_top5")
    params = calibration_metadata["calibration_params"]
    assert params["method"] == calibration_metadata["calibration_method"]
    if params["method"] == "platt":
        assert {"a", "b"} <= set(params)
    else:
        assert {"x_thresholds", "y_thresholds"} <= set(params)
    # Rebuilding the calibrator from metadata must work and output [0, 1].
    calibrator = calibrator_from_dict(calibration_metadata["calibration_params"])
    probs = calibrator.transform(np.linspace(0, 1, 101))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_calibrated_probability_always_in_unit_interval(detector):
    grid = np.linspace(0.0, 1.0, 1001)
    probs = detector.calibrator.transform(grid)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert detector.calibrator.transform(np.array([0.0]))[0] <= 1.0
    assert detector.calibrator.transform(np.array([1.0]))[0] >= 0.0


def test_calibration_is_monotone_in_signal(detector):
    probs = detector.calibrator.transform(np.linspace(0, 1, 201))
    assert np.all(np.diff(probs) >= -1e-9)


@pytest.mark.slow
def test_inference_output_schema(detector):
    out = detector.predict(Image.fromarray(_sample_image()).convert("RGB"))
    assert REQUIRED_OUTPUT_KEYS <= set(out)
    assert isinstance(out["detections"], list)
    for det in out["detections"]:
        assert set(det) == {"bbox", "category", "confidence"}
        assert len(det["bbox"]) == 4
        assert det["category"] in ("accident", "non_accident")
        assert 0.0 <= det["confidence"] <= 1.0
    assert 0.0 <= out["accident_detection_confidence"] <= 1.0
    assert 0.0 <= out["calibrated_accident_probability"] <= 1.0
    assert out["detector_version"] == "accident_calibrated_detector_v1"


@pytest.mark.slow
def test_inference_accepts_numpy_and_tensor(detector):
    arr = _sample_image(seed=3)
    from_arr = detector.predict(arr)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    from_tensor = detector.predict(tensor)
    assert from_arr == from_tensor
    # Float [0, 1] HWC arrays (the dataset's natural format) must give the
    # same result as the equivalent uint8 array.
    from_float = detector.predict(arr.astype(np.float32) / 255.0)
    assert from_float == from_arr


@pytest.mark.slow
def test_deterministic_inference(detector):
    image = _sample_image(seed=7)
    first = detector.predict(image)
    second = detector.predict(image)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_no_test_data_used_for_calibration(calibration_metadata):
    guard = calibration_metadata["leakage_guard"]
    assert guard["test_used_for_calibration"] is False
    assert guard["test_used_for_threshold_selection"] is False
    assert calibration_metadata["fit_split"] == "validation"
    assert calibration_metadata["selection_split"] == "validation"
    summary_path = REPORT_DIR / "phase6_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["leakage_guard"]["test_evaluations_performed"] == 1
    # The calibrator must reproduce from the stored validation-only arrays:
    # a behavioral check that the fit inputs come from the validation split.
    va = summary["validation_arrays"]
    signals = np.asarray(va["signals"], dtype=np.float64)
    labels = np.asarray(va["labels"], dtype=np.int64)
    refit = build_calibrator(calibration_metadata["calibration_method"]).fit(signals, labels)
    stored = calibration_metadata["calibration_params"]
    if stored["method"] == "platt":
        # Stored signals are rounded to 6 decimals in the summary, so the
        # refit matches only to a small tolerance.
        assert refit.a == pytest.approx(stored["a"], abs=1e-3)
        assert refit.b == pytest.approx(stored["b"], abs=1e-3)
    else:
        np.testing.assert_allclose(refit.x_thresholds, stored["x_thresholds"], atol=1e-3)
        np.testing.assert_allclose(refit.y_thresholds, stored["y_thresholds"], atol=1e-3)


def test_original_dataset_unchanged(calibration_metadata):
    recorded = calibration_metadata["dataset_sha256"]
    assert set(recorded) == set(DATASET_FILES)
    for name, expected_hash in recorded.items():
        digest = sha256_file(DATASET_DIR / name)
        assert digest == expected_hash, f"{name} changed after the Phase 6 run"


# ---------------------------------------------------------------------------
# Selection-rule and class-filter coverage
# ---------------------------------------------------------------------------

def test_threshold_analysis_class_filter_counts_only_that_class():
    gt = [
        {"image_id": 0, "boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
         "labels": np.array([0, 1])},
    ]
    # Accident pred matches GT[0]; non_accident pred is a FP for its class.
    pred = [
        {"image_id": 0,
         "boxes": np.array([[1, 1, 11, 11], [50, 50, 60, 60]], dtype=float),
         "labels": np.array([0, 1]),
         "scores": np.array([0.8, 0.9])},
    ]
    rows = threshold_analysis(gt, pred, thresholds=[0.5], class_filter=0)
    assert rows[0]["true_positives"] == 1
    assert rows[0]["false_positives"] == 0
    assert rows[0]["false_negatives"] == 0
    assert rows[0]["recall"] == 1.0
    rows_all = threshold_analysis(gt, pred, thresholds=[0.5])
    # Without the filter, the non_accident box with no GT adds an FP.
    assert rows_all[0]["false_positives"] == 1


def test_select_signal_picks_lowest_brier():
    rows = [
        {"signal": "max", "brier_uncalibrated": 0.07},
        {"signal": "top3_mean", "brier_uncalibrated": 0.20},
        {"signal": "noisy_or_top5", "brier_uncalibrated": 0.04},
    ]
    assert select_signal(rows) == "noisy_or_top5"


def _fitted(method: str, brier: float, cv_brier: float, cv_ll: float) -> dict:
    return {method: {"brier": brier, "cv": {"cv_brier_mean": cv_brier, "cv_log_loss_mean": cv_ll}}}


def test_select_method_brier_gap_decides_outside_tie_band():
    fitted = _fitted("platt", 0.010, 0.010, 0.05)
    fitted.update(_fitted("isotonic", 0.020, 0.020, 0.10))
    assert _select_method(fitted) == "platt"
    fitted = _fitted("platt", 0.020, 0.010, 0.05)
    fitted.update(_fitted("isotonic", 0.010, 0.020, 0.10))
    assert _select_method(fitted) == "isotonic"


def test_select_method_tie_break_prefers_heldout_stability():
    # In-sample Brier near tie (gap 0.0001 < 0.002): held-out CV decides.
    fitted = _fitted("platt", 0.0063, 0.0063, 0.038)
    fitted.update(_fitted("isotonic", 0.0061, 0.0064, 0.106))
    assert _select_method(fitted) == "platt"
    # And the mirror case, where isotonic is genuinely more stable.
    fitted = _fitted("platt", 0.0061, 0.0064, 0.106)
    fitted.update(_fitted("isotonic", 0.0063, 0.0063, 0.038))
    assert _select_method(fitted) == "isotonic"


def test_select_method_uses_unrounded_gap():
    # True gap 0.00204 (> 0.002) must decide by in-sample Brier; rounding both
    # metrics to 4 decimals first would show a 0.0020 gap and wrongly fall
    # through to the tie-break, which favors Platt here.
    fitted = _fitted("platt", 0.00650, 0.0063, 0.038)
    fitted.update(_fitted("isotonic", 0.00446, 0.0064, 0.106))
    assert _select_method(fitted) == "isotonic"


def test_original_dataset_unchanged(calibration_metadata):
    recorded = calibration_metadata["dataset_sha256"]
    assert set(recorded) == set(DATASET_FILES)
    for name, expected_hash in recorded.items():
        digest = hashlib.sha256((DATASET_DIR / name).read_bytes()).hexdigest()
        assert digest == expected_hash, f"{name} changed after the Phase 6 run"


def test_save_calibrated_checkpoint_round_trip(tmp_path, calibration_metadata):
    payload = torch.load(CALIBRATED_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    out = save_calibrated_checkpoint(
        detector_state_dict=payload["model_state_dict"],
        calibration_dict=payload["calibration"],
        metadata={
            "checkpoint_source": payload["metadata"]["checkpoint_source"],
            "fit_split": "validation",
            "signal": payload["calibration"]["signal"],
            "fit_image_count": payload["metadata"]["fit_image_count"],
        },
        output_path=tmp_path / "round_trip.pth",
    )
    reloaded = torch.load(out, map_location="cpu", weights_only=True)
    assert reloaded["calibration"] == payload["calibration"]
    assert set(reloaded["model_state_dict"]) == set(payload["model_state_dict"])