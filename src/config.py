"""Phase 7 — Safety Risk Fusion configuration.

Every risk-fusion parameter lives here; no threshold or weight is hardcoded
in the risk engine. These are heuristic engineering choices, NOT scientifically
validated values, and none of them was tuned on any dataset split (let alone
the test set).

The fusion inputs are four, and they are deliberately kept semantically
separate:

- ``near_miss_probability``  — calibrated P(class_1) from the Phase 4
  near-miss/conflict model. NOTE: the class-1 label semantics are unconfirmed
  (see reports/data_dictionary.md); this pipeline only echoes the Phase 4
  artifact's own output.
- ``accident_probability``   — calibrated accident probability from the
  Phase 6 detector. Raw detector confidence is never used as a probability.
- ``severity``               — CALLER-SUPPLIED relative severity in [0, 1].
  Neither source dataset carries severity labels; the meaning of this value
  is owned entirely by the caller. This pipeline does not measure severity.
- ``recency``                — derived from the CALLER-SUPPLIED event age in
  hours via an exponential decay (see RECENCY_HALF_LIFE_HOURS).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Artifact paths (Phase 4 near-miss + Phase 6 accident, loaded read-only)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEAR_MISS_MODEL_PATH = PROJECT_ROOT / "models" / "near_miss" / "random_forest_D_calibrated.joblib"
ACCIDENT_DETECTOR_PATH = PROJECT_ROOT / "models" / "accident" / "calibrated_accident_detector.pth"

# ---------------------------------------------------------------------------
# Fusion weights
# ---------------------------------------------------------------------------

# Weight of each input in the linear fusion. Inputs present in an assessment
# contribute weight * value; the weights of the PRESENT inputs are then
# renormalized (see src/risk_engine.py), so the raw values below only define
# relative importance. All weights must be > 0.
RISK_WEIGHTS: dict[str, float] = {
    "near_miss": 0.35,
    "accident": 0.40,
    "severity": 0.15,
    "recency": 0.10,
}

# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

# risk_score is on a 0-100 scale (RISK_SCORE_SCALE). Boundary semantics:
#   score <  medium_threshold              -> LOW
#   medium_threshold <= score < high       -> MEDIUM
#   high_threshold <= score < critical     -> HIGH
#   score >= critical_threshold            -> CRITICAL
# So each value here is the LOWER bound of its level (inclusive), and LOW
# implicitly starts at 0.
RISK_SCORE_SCALE: float = 100.0
RISK_LEVEL_THRESHOLDS: dict[str, float] = {
    "medium": 25.0,
    "high": 50.0,
    "critical": 75.0,
}

# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

# recency_factor = 0.5 ** (event_age_hours / RECENCY_HALF_LIFE_HOURS)
# An event exactly this many hours old contributes 0.5; twice as old 0.25;
# age 0 contributes 1.0. Must be > 0.
RECENCY_HALF_LIFE_HOURS: float = 24.0
