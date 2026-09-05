"""Phase 5B tests: label mapping, checkpointing, experiment tracking, thresholds."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("pycocotools")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from accident import CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT
from accident.dataset import AccidentParquetDataset, collate_detection
from accident.inference import AccidentDetector
from accident.labels import (
    DATASET_LABEL_ACCIDENT,
    DATASET_LABEL_NON_ACCIDENT,
    LABEL_OFFSET,
    MODEL_LABEL_ACCIDENT,
    MODEL_LABEL_BACKGROUND,
    MODEL_LABEL_NON_ACCIDENT,
    dataset_to_model_labels,
    model_label_to_category_name,
    model_to_dataset_labels,
)
from accident.metrics import threshold_analysis
from accident.model import NUM_CLASSES, build_detector
from accident.utils import DATASET_DIR, seed_everything


# ---------------------------------------------------------------------------
# Label / background mapping
# ---------------------------------------------------------------------------


def test_label_offset_mapping_is_correct():
    assert dataset_to_model_labels(torch.tensor([0, 1])).tolist() == [
        MODEL_LABEL_ACCIDENT,
        MODEL_LABEL_NON_ACCIDENT,
    ]
    assert dataset_to_model_labels(np.array([0, 1])).tolist() == [1, 2]


def test_model_to_dataset_mapping_round_trips():
    model_labels = torch.tensor([1, 2])
    round_trip = dataset_to_model_labels(model_to_dataset_labels(model_labels))
    assert torch.equal(round_trip, model_labels)
    assert model_to_dataset_labels(torch.tensor([1, 2])).tolist() == [
        DATASET_LABEL_ACCIDENT,
        DATASET_LABEL_NON_ACCIDENT,
    ]


def test_background_label_is_zero_and_dataset_labels_are_nonzero_in_model_space():
    assert MODEL_LABEL_BACKGROUND == 0
    model_labels = dataset_to_model_labels(torch.tensor([CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT]))
    assert (model_labels > 0).all(), "no dataset class may map onto model background"


def test_built_model_has_three_class_predictor():
    seed_everything(42)
    model = build_detector(num_classes=NUM_CLASSES, pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES == 3


def test_training_targets_never_contain_background_label():
    """Labels passed through the training offset must all be >= 1."""
    ds = AccidentParquetDataset(split="validation", target_size=(320, 320))
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_detection)
    images, targets = next(iter(loader))
    offset = dataset_to_model_labels(targets[0]["labels"])
    assert (offset >= 1).all()


def test_model_label_to_category_name():
    assert model_label_to_category_name(1) == "accident"
    assert model_label_to_category_name(2) == "non_accident"
    assert model_label_to_category_name(0) is None


# ---------------------------------------------------------------------------
# Bounding-box handling across the full dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_all_boxes_within_image_bounds(split):
    """Every box in every split: x2 > x1, y2 > y1 and inside [0, 640]."""
    ds = AccidentParquetDataset(split=split, target_size=(640, 640))
    for item in ds:
        h, w = item["image"].shape[1], item["image"].shape[2]
        for box in item["boxes"].numpy():
            x1, y1, x2, y2 = box
            assert 0 <= x1 < w and 0 <= y1 < h, f"bad x1/y1 in {split} id={item['image_id']}"
            assert 0 < x2 <= w and 0 < y2 <= h, f"bad x2/y2 in {split} id={item['image_id']}"
            assert x2 > x1 and y2 > y1, f"degenerate box in {split} id={item['image_id']}"


# ---------------------------------------------------------------------------
# Prediction class mapping (save/load + inference output)
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_round_trip(tmp_path):
    seed_everything(42)
    model = build_detector(num_classes=NUM_CLASSES, pretrained=False)
    path = tmp_path / "phase5b.pth"
    torch.save(model.state_dict(), path)

    loaded = build_detector(num_classes=NUM_CLASSES, pretrained=False)
    loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    loaded.eval()

    ds = AccidentParquetDataset(split="validation", target_size=(320, 320))
    item = ds[0]
    with torch.no_grad():
        out = loaded([item["image"]])[0]
    # Model label space only: background + the two classes.
    assert set(out["labels"].unique().tolist()).issubset({0, 1, 2})


def test_inference_maps_model_labels_to_dataset_categories(tmp_path):
    seed_everything(42)
    model = build_detector(num_classes=NUM_CLASSES, pretrained=False)
    path = tmp_path / "detector.pth"
    torch.save(model.state_dict(), path)

    ds = AccidentParquetDataset(split="validation", target_size=(320, 320))
    result = AccidentDetector(path, image_size=(320, 320)).predict(ds[0]["image"], score_threshold=0.0)

    categories = {d["category"] for d in result["detections"]}
    assert categories.issubset({"accident", "non_accident"})
    assert "background" not in categories


def test_baseline_phase5_checkpoint_loads_with_two_class_head():
    """The immutable Phase 5 baseline (2-class head) still loads explicitly."""
    checkpoint = PROJECT_ROOT / "models" / "accident" / "best_detector.pth"
    if not checkpoint.exists():
        pytest.skip("Phase 5 baseline checkpoint not present")
    model = build_detector(num_classes=2, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )


# ---------------------------------------------------------------------------
# Best-checkpoint selection
# ---------------------------------------------------------------------------


def test_checkpoint_selection_prefers_accident_ap_then_map50():
    from accident.phase5b import checkpoint_score

    better = {"accident_ap50": 0.40, "map50": 0.30}
    worse_lower_ap = {"accident_ap50": 0.20, "map50": 0.90}
    worse_tie_ap = {"accident_ap50": 0.40, "map50": 0.25}
    assert checkpoint_score(better) > checkpoint_score(worse_lower_ap)
    assert checkpoint_score(better) > checkpoint_score(worse_tie_ap)


def test_checkpoint_selection_update_logic(tmp_path):
    from accident.phase5b import is_improvement

    assert is_improvement(0.5, -1.0)
    assert is_improvement(0.51, 0.5)
    assert not is_improvement(0.5, 0.5)
    assert not is_improvement(0.49, 0.5)


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


def test_append_experiment_row_writes_csv_with_header(tmp_path):
    from accident.phase5b import EXPERIMENT_COLUMNS, append_experiment_row

    csv_path = tmp_path / "experiments.csv"
    row = {col: "" for col in EXPERIMENT_COLUMNS}
    row.update(
        {
            "experiment_id": "EXP_TEST",
            "model": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "epochs": 2,
            "val_accident_ap50": 0.1234,
        }
    )
    append_experiment_row(csv_path, row)
    append_experiment_row(csv_path, {**row, "experiment_id": "EXP_TEST2"})

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "EXP_TEST"
    assert rows[1]["experiment_id"] == "EXP_TEST2"
    assert "val_map50" in rows[0]


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def test_threshold_analysis_covers_required_thresholds():
    rng = np.random.default_rng(42)
    gt_records = [
        {
            "image_id": 0,
            "boxes": np.array([[10.0, 10.0, 60.0, 60.0], [200.0, 200.0, 300.0, 300.0]]),
            "labels": np.array([CLASS_ID_ACCIDENT, CLASS_ID_NON_ACCIDENT]),
        }
    ]
    pred_records = [
        {
            "image_id": 0,
            "boxes": np.array([[12.0, 11.0, 62.0, 61.0], [210.0, 205.0, 310.0, 305.0]]),
            "labels": np.array([CLASS_ID_ACCIDENT, CLASS_ID_ACCIDENT]),
            "scores": np.array([0.8, 0.3]),
        }
    ]
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = threshold_analysis(gt_records, pred_records, thresholds=thresholds)
    assert [r["threshold"] for r in rows] == thresholds
    # The 0.3-score accident prediction must be dropped above 0.30.
    by_thr = {r["threshold"]: r for r in rows}
    assert by_thr[0.05]["true_positives"] >= 1
    assert by_thr[0.90]["false_positives"] == 0
    for r in rows:
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0