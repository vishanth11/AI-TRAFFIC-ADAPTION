"""Phase 7 tests — risk fusion engine, safety intelligence, boundary cases.

Unit tests cover the fusion math, threshold boundaries, and missing/invalid
input handling without touching any model artifact. Integration tests load
the real Phase 4 near-miss pipeline and Phase 6 calibrated detector (marked
`slow`) and are skipped when the artifacts are absent.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
from risk_engine import (
    RiskInputError,
    assess_risk,
    compute_risk_level,
    recency_factor,
)
from safety_intelligence import SafetyIntelligence, to_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEAR_MISS_ARTIFACT = PROJECT_ROOT / "models" / "near_miss" / "random_forest_A_calibrated.joblib"
ACCIDENT_CHECKPOINT = PROJECT_ROOT / "models" / "accident" / "calibrated_accident_detector.pth"
VALIDATION_SPLIT = PROJECT_ROOT / "data" / "processed" / "near_miss" / "splits" / "validation.parquet"


@pytest.fixture(scope="module")
def near_miss_row() -> dict:
    if not NEAR_MISS_ARTIFACT.exists() or not VALIDATION_SPLIT.exists():
        pytest.skip("Phase 4 artifacts not built yet.")
    return pd.read_parquet(VALIDATION_SPLIT).iloc[0].to_dict()


@pytest.fixture(scope="module")
def safety() -> SafetyIntelligence:
    if not NEAR_MISS_ARTIFACT.exists() or not ACCIDENT_CHECKPOINT.exists():
        pytest.skip("Phase 4/6 artifacts not built yet.")
    return SafetyIntelligence()


# ---------------------------------------------------------------------------
# Fusion math
# ---------------------------------------------------------------------------

def test_full_assessment_contributions_sum_to_score():
    out = assess_risk(
        near_miss_probability=0.8,
        accident_probability=0.6,
        severity=0.5,
        event_age_hours=24.0,
    )
    assert abs(sum(out["contributions"].values()) - out["risk_score"]) < 1e-3
    # Weights are renormalized over the 4 present inputs: w_i / total.
    assert abs(sum(out["weights_applied"].values()) - 1.0) < 1e-6
    expected = 100.0 * (0.35 * 0.8 + 0.40 * 0.6 + 0.15 * 0.5 + 0.10 * 0.5)
    assert out["risk_score"] == pytest.approx(expected, abs=1e-3)


def test_single_input_score_is_scaled_value():
    assert assess_risk(accident_probability=1.0)["risk_score"] == pytest.approx(100.0)
    assert assess_risk(accident_probability=0.3)["risk_score"] == pytest.approx(30.0)
    assert assess_risk(severity=0.7)["risk_score"] == pytest.approx(70.0)


def test_each_input_is_monotone_in_its_own_value():
    base = {"near_miss_probability": 0.2, "accident_probability": 0.2,
            "severity": 0.2, "event_age_hours": 1.0}
    baseline = assess_risk(**base)
    for bump in (
        {"near_miss_probability": 0.9},
        {"accident_probability": 0.9},
        {"severity": 0.9},
        {"event_age_hours": 0.0},  # more recent -> higher recency factor
    ):
        out = assess_risk(**{**base, **bump})
        assert out["risk_score"] > baseline["risk_score"]


def test_recency_factor_decay():
    assert recency_factor(0.0, 24.0) == pytest.approx(1.0)
    assert recency_factor(24.0, 24.0) == pytest.approx(0.5)
    assert recency_factor(48.0, 24.0) == pytest.approx(0.25)
    # Far enough past the half-life that the exponent underflows to 0.0.
    assert recency_factor(100_000.0, 24.0) == 0.0
    assert 0.0 < recency_factor(10_000.0, 24.0) < 1e-100


# ---------------------------------------------------------------------------
# Risk levels and threshold boundaries (thresholds are inclusive lower bounds)
# ---------------------------------------------------------------------------

def test_level_boundaries_default_config():
    defaults = config.RISK_LEVEL_THRESHOLDS
    assert compute_risk_level(defaults["medium"] - 1e-9, defaults) == "LOW"
    assert compute_risk_level(defaults["medium"], defaults) == "MEDIUM"
    assert compute_risk_level(defaults["high"], defaults) == "HIGH"
    assert compute_risk_level(defaults["critical"], defaults) == "CRITICAL"


def test_score_exactly_at_thresholds_goes_up_a_level():
    # With default weights, a lone accident probability p maps to score 100*p.
    assert assess_risk(accident_probability=0.2499)["risk_level"] == "LOW"
    assert assess_risk(accident_probability=0.25)["risk_level"] == "MEDIUM"
    assert assess_risk(accident_probability=0.4999)["risk_level"] == "MEDIUM"
    assert assess_risk(accident_probability=0.50)["risk_level"] == "HIGH"
    assert assess_risk(accident_probability=0.7499)["risk_level"] == "HIGH"
    assert assess_risk(accident_probability=0.75)["risk_level"] == "CRITICAL"


def test_config_override_changes_levels():
    cfg = {"risk_level_thresholds": {"medium": 10.0, "high": 20.0, "critical": 30.0}}
    assert assess_risk(accident_probability=0.29, config=cfg)["risk_level"] == "HIGH"
    assert assess_risk(accident_probability=0.31, config=cfg)["risk_level"] == "CRITICAL"


def test_config_override_weights_change_fusion():
    cfg = {"risk_weights": {"near_miss": 1.0, "accident": 1.0, "severity": 1.0, "recency": 1.0}}
    out = assess_risk(near_miss_probability=0.0, accident_probability=1.0,
                      severity=0.0, event_age_hours=0.0, config=cfg)
    assert out["risk_score"] == pytest.approx(50.0)


def test_invalid_config_overrides_raise():
    with pytest.raises(RiskInputError):
        assess_risk(accident_probability=0.5, config={"risk_weights": {"near_miss": 0.0}})
    with pytest.raises(RiskInputError):
        assess_risk(accident_probability=0.5,
                    config={"risk_level_thresholds": {"medium": 60.0, "high": 30.0, "critical": 90.0}})
    with pytest.raises(RiskInputError):
        assess_risk(accident_probability=0.5, config={"risk_score_scale": 0.0})
    with pytest.raises(RiskInputError):
        assess_risk(accident_probability=0.5, config={"not_a_key": 1})


# ---------------------------------------------------------------------------
# Missing-input handling
# ---------------------------------------------------------------------------

def test_all_inputs_missing_raises():
    with pytest.raises(RiskInputError, match="No risk inputs"):
        assess_risk()


def test_each_single_input_alone_produces_a_level():
    assert assess_risk(near_miss_probability=0.5)["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert assess_risk(accident_probability=0.5)["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert assess_risk(severity=0.5)["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert assess_risk(event_age_hours=12.0)["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_missing_inputs_are_reported_and_excluded():
    out = assess_risk(accident_probability=0.5, severity=0.5)
    assert out["missing_inputs"] == ["near_miss", "recency"]
    assert sorted(out["present_inputs"]) == ["accident", "severity"]
    # Renormalization: weights_applied covers only present inputs.
    assert set(out["weights_applied"]) == {"accident", "severity"}
    assert any("renormalized" in line for line in out["explanation"])


def test_missing_accident_and_near_miss_are_null_not_zero():
    out = assess_risk(severity=0.5)
    assert out["near_miss_probability"] is None
    assert out["accident_probability"] is None


# ---------------------------------------------------------------------------
# Invalid-input handling (fail fast, never guess)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"accident_probability": 1.5},
    {"accident_probability": -0.1},
    {"near_miss_probability": 2.0},
    {"severity": -1.0},
    {"accident_probability": float("nan")},
    {"accident_probability": float("inf")},
    {"event_age_hours": -1.0},
    {"event_age_hours": float("nan")},
    {"accident_probability": True},  # bool is not an accepted numeric type
    {"severity": "high"},  # no invented severity semantics: numbers only
    {"accident_probability": None if False else [0.5]},
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(RiskInputError):
        assess_risk(**kwargs)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_is_json_serializable():
    out = assess_risk(near_miss_probability=0.8, accident_probability=0.6,
                      severity=0.5, event_age_hours=24.0)
    parsed = json.loads(json.dumps(out))
    assert parsed["risk_level"] == out["risk_level"]
    assert math.isfinite(parsed["risk_score"])


def test_example_schema_keys_present():
    out = assess_risk(near_miss_probability=0.8, accident_probability=0.6,
                      severity=0.5, event_age_hours=24.0)
    for key in ("near_miss_probability", "accident_probability", "risk_score",
                "risk_level", "severity", "explanation"):
        assert key in out


def test_semantic_separation_is_enforced_and_documented():
    out = assess_risk(near_miss_probability=0.9, accident_probability=0.1, severity=0.0)
    # The two probabilities are echoed separately and unchanged.
    assert out["near_miss_probability"] == 0.9
    assert out["accident_probability"] == 0.1
    # The score is explicitly not a probability.
    assert out["is_probability"] is False
    explanation = " ".join(out["explanation"])
    assert "NOT a probability" in explanation
    assert "kept semantically separate" in explanation
    assert "raw detector confidence is never treated as a probability" in explanation
    assert "unconfirmed" in explanation  # near-miss label semantics caveat


def test_explanation_discloses_missing_inputs():
    out = assess_risk(accident_probability=0.5)
    explanation = " ".join(out["explanation"])
    assert "near_miss_probability is missing" in explanation
    assert "recency factor" in explanation


# ---------------------------------------------------------------------------
# Integration — real Phase 4 near-miss model (no retraining)
# ---------------------------------------------------------------------------

def test_near_miss_probability_from_real_model(near_miss_row):
    si = SafetyIntelligence()
    out = si.assess_event(near_miss_features=near_miss_row, severity=0.4, event_age_hours=6.0)
    assert 0.0 <= out["near_miss_probability"] <= 1.0
    assert out["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "near_miss" in out["present_inputs"]
    assert out["model_provenance"]["retrained"] is False
    json.loads(to_json(out))


def test_near_miss_missing_feature_column_raises_clear_error(near_miss_row):
    si = SafetyIntelligence()
    broken = dict(near_miss_row)
    broken.pop("PET")
    with pytest.raises(ValueError, match="missing required columns"):
        si.assess_event(near_miss_features=broken)


def test_near_miss_lazy_load_and_model_version(near_miss_row):
    si = SafetyIntelligence()
    assert not si.near_miss_model_loaded
    si.assess_event(near_miss_features=near_miss_row)
    assert si.near_miss_model_loaded


# ---------------------------------------------------------------------------
# Integration — Phase 6 accident detector output handling (no retraining)
# ---------------------------------------------------------------------------

def test_detector_output_dict_supplies_calibrated_probability():
    si = SafetyIntelligence()
    out = si.assess_event(
        accident_detector_output={
            "calibrated_accident_probability": 0.9,
            "accident_detection_confidence": 0.99,
        },
        severity=0.5,
        event_age_hours=0.0,
    )
    assert out["accident_probability"] == 0.9
    # The raw confidence is carried separately, never as the probability.
    assert out["detector_raw_signal"]["accident_detection_confidence"] == 0.99
    assert "NOT a probability" in out["detector_raw_signal"]["note"]
    json.loads(to_json(out))


def test_detector_output_without_calibrated_probability_is_refused():
    si = SafetyIntelligence()
    with pytest.raises(ValueError, match="calibrated_accident_probability"):
        si.assess_event(accident_detector_output={"accident_detection_confidence": 0.99})


def test_explicit_probability_wins_over_detector_output():
    si = SafetyIntelligence()
    out = si.assess_event(
        accident_detector_output={"calibrated_accident_probability": 0.9},
        accident_probability=0.2,
    )
    assert out["accident_probability"] == 0.2


@pytest.mark.slow
def test_full_image_assessment_with_real_detector(safety):
    rng = np.random.RandomState(0)
    image = (rng.rand(320, 320, 3) * 255).astype(np.uint8)
    out = safety.assess_image(image, severity=0.5, event_age_hours=0.0)
    assert 0.0 <= out["accident_probability"] <= 1.0
    assert 0.0 <= out["risk_score"] <= 100.0
    assert out["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert out["detector_output"]["detector_version"] == "accident_calibrated_detector_v1"
    assert "accident" in out["present_inputs"]
    json.loads(to_json(out))


@pytest.mark.slow
def test_image_assessment_with_both_real_models(safety, near_miss_row):
    rng = np.random.RandomState(1)
    image = (rng.rand(320, 320, 3) * 255).astype(np.uint8)
    out = safety.assess_image(image, near_miss_features=near_miss_row, severity=0.2)
    assert 0.0 <= out["near_miss_probability"] <= 1.0
    assert sorted(out["present_inputs"]) == ["accident", "near_miss", "severity"]
    parsed = json.loads(to_json(out))
    assert parsed["risk_level"] == out["risk_level"]