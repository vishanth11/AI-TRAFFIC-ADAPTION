"""Phase 7 — transparent risk fusion engine.

Fuses four inputs into a heuristic risk score and a LOW/MEDIUM/HIGH/CRITICAL
level:

- ``near_miss_probability`` — calibrated P(class_1) from the Phase 4
  near-miss/conflict model (kept semantically separate from the accident
  probability: different model, different event, never merged into one
  probability).
- ``accident_probability`` — calibrated accident probability from the Phase 6
  detector. Raw detector confidence is NEVER accepted as a probability here.
- ``severity``             — caller-supplied relative severity in [0, 1].
  Neither source dataset carries severity labels; the meaning of this number
  is owned by the caller. This engine does not measure or interpret severity.
- ``event_age_hours``      — caller-supplied event age; converted to a
  recency factor via an exponential half-life decay from config.

Missing inputs (``None``) are excluded and the weights of the PRESENT inputs
are renormalized, so an assessment is always computable from whatever
evidence exists — but an assessment with no inputs at all raises ValueError
instead of silently returning a confident-looking LOW.

The score is a HEURISTIC engineering fusion, not a probability and not a
scientifically validated quantity. All weights/thresholds come from
``src/config.py``; nothing is hardcoded here.
"""

from __future__ import annotations

import math
from typing import Any

try:  # src/ root import (risk_engine.py sits next to config.py)
    import config as _config
except ImportError:  # package-style import
    from . import config as _config

__all__ = ["assess_risk", "compute_risk_level", "recency_factor", "RiskInputError"]

#: Config keys this module reads, exposed so callers can build overrides.
CONFIG_KEYS = ("risk_weights", "risk_level_thresholds", "risk_score_scale", "recency_half_life_hours")

#: Fusion input keys, in canonical output order.
INPUT_KEYS = ("near_miss", "accident", "severity", "recency")


class RiskInputError(ValueError):
    """Raised for missing-evidence or invalid risk-fusion inputs."""


# ---------------------------------------------------------------------------
# Validation helpers — validate at the boundary, fail fast, never guess
# ---------------------------------------------------------------------------

def _validate_unit_interval(name: str, value: Any) -> float | None:
    """Validate a probability-like input; ``None`` means genuinely missing."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskInputError(f"{name} must be a number or None, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RiskInputError(f"{name} must be finite, got {numeric}")
    if not 0.0 <= numeric <= 1.0:
        raise RiskInputError(f"{name} must be within [0, 1], got {numeric}")
    return numeric


def _validate_age_hours(value: Any) -> float | None:
    """Validate a caller-supplied event age in hours; ``None`` = missing."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskInputError(f"event_age_hours must be a number or None, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RiskInputError(f"event_age_hours must be finite, got {numeric}")
    if numeric < 0.0:
        raise RiskInputError(f"event_age_hours must be >= 0, got {numeric}")
    return numeric


def _validate_weights(weights: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in INPUT_KEYS if key not in weights]
    if missing:
        raise RiskInputError(f"risk_weights is missing keys: {missing}")
    validated: dict[str, float] = {}
    for key in INPUT_KEYS:
        raw = weights[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise RiskInputError(f"risk_weights[{key!r}] must be a finite number, got {raw!r}")
        if float(raw) <= 0.0:
            raise RiskInputError(f"risk_weights[{key!r}] must be > 0, got {raw}")
        validated[key] = float(raw)
    return validated


def _validate_thresholds(thresholds: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in ("medium", "high", "critical") if name not in thresholds]
    if missing:
        raise RiskInputError(f"risk_level_thresholds is missing keys: {missing}")
    validated = {name: float(thresholds[name]) for name in ("medium", "high", "critical")}
    ordered = [validated["medium"], validated["high"], validated["critical"]]
    if any(not math.isfinite(v) for v in ordered):
        raise RiskInputError("risk_level_thresholds must be finite")
    if not 0 < ordered[0] < ordered[1] < ordered[2]:
        raise RiskInputError(
            "risk_level_thresholds must satisfy 0 < medium < high < critical, "
            f"got medium={ordered[0]}, high={ordered[1]}, critical={ordered[2]}"
        )
    return validated


def _validate_scale(scale: Any) -> float:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise RiskInputError(f"risk_score_scale must be a finite number, got {scale!r}")
    if float(scale) <= 0.0:
        raise RiskInputError(f"risk_score_scale must be > 0, got {scale}")
    return float(scale)


def _resolve_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller overrides onto the src/config.py defaults."""
    resolved = {
        "risk_weights": dict(_config.RISK_WEIGHTS),
        "risk_level_thresholds": dict(_config.RISK_LEVEL_THRESHOLDS),
        "risk_score_scale": _config.RISK_SCORE_SCALE,
        "recency_half_life_hours": _config.RECENCY_HALF_LIFE_HOURS,
    }
    if overrides:
        unknown = [k for k in overrides if k not in CONFIG_KEYS]
        if unknown:
            raise RiskInputError(f"Unknown config override keys: {unknown}; expected {CONFIG_KEYS}")
        for key, value in overrides.items():
            if key in ("risk_weights", "risk_level_thresholds"):
                if not isinstance(value, dict):
                    raise RiskInputError(f"config override {key!r} must be a dict, got {type(value).__name__}")
                resolved[key].update(value)
            else:
                resolved[key] = value
    resolved["risk_weights"] = _validate_weights(resolved["risk_weights"])
    resolved["risk_level_thresholds"] = _validate_thresholds(resolved["risk_level_thresholds"])
    resolved["risk_score_scale"] = _validate_scale(resolved["risk_score_scale"])
    half_life = resolved["recency_half_life_hours"]
    if (
        isinstance(half_life, bool)
        or not isinstance(half_life, (int, float))
        or not math.isfinite(float(half_life))
        or float(half_life) <= 0.0
    ):
        raise RiskInputError(f"recency_half_life_hours must be a finite number > 0, got {half_life!r}")
    resolved["recency_half_life_hours"] = float(half_life)
    return resolved


# ---------------------------------------------------------------------------
# Fusion math
# ---------------------------------------------------------------------------

def recency_factor(event_age_hours: float | None, half_life_hours: float) -> float | None:
    """Exponential recency decay: ``0.5 ** (age / half_life)``.

    Age 0 -> 1.0, one half-life -> 0.5, and sufficiently large ages decay to
    0.0 (float underflow, which is the mathematically correct limit).
    """
    if event_age_hours is None:
        return None
    return 0.5 ** (float(event_age_hours) / float(half_life_hours))


def compute_risk_level(score: float, thresholds: dict[str, float]) -> str:
    """Map a fused score to its level (thresholds are inclusive lower bounds)."""
    if score >= thresholds["critical"]:
        return "CRITICAL"
    if score >= thresholds["high"]:
        return "HIGH"
    if score >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"


def assess_risk(
    near_miss_probability: float | None = None,
    accident_probability: float | None = None,
    severity: float | None = None,
    event_age_hours: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse the four inputs into a transparent heuristic risk assessment.

    All inputs are optional; ``None`` means the input is genuinely missing and
    it is excluded, with the weights of the present inputs renormalized. At
    least one input must be present.

    Returns a JSON-serializable dict (see the module docstring for the input
    semantics and the caveats).
    """
    cfg = _resolve_config(config)
    weights = cfg["risk_weights"]
    thresholds = cfg["risk_level_thresholds"]
    scale = cfg["risk_score_scale"]

    values: dict[str, float | None] = {
        "near_miss": _validate_unit_interval("near_miss_probability", near_miss_probability),
        "accident": _validate_unit_interval("accident_probability", accident_probability),
        "severity": _validate_unit_interval("severity", severity),
        "recency": recency_factor(
            _validate_age_hours(event_age_hours), cfg["recency_half_life_hours"]
        ),
    }
    present = {key: value for key, value in values.items() if value is not None}
    if not present:
        raise RiskInputError(
            "No risk inputs provided — refused to invent a risk level. "
            f"Provide at least one of: {', '.join(INPUT_KEYS)}."
        )

    weight_total = sum(weights[key] for key in present)
    contributions = {key: weights[key] * value for key, value in present.items()}
    fused = sum(contributions.values())
    risk_score = scale * fused / weight_total
    risk_level = compute_risk_level(risk_score, thresholds)

    echo = {
        "near_miss": near_miss_probability,
        "accident": accident_probability,
        "severity": severity,
        "event_age_hours": event_age_hours,
    }
    missing = [key for key in INPUT_KEYS if key not in present]
    explanation = _build_explanation(
        echo=echo,
        present=present,
        missing=missing,
        weights=weights,
        contributions=contributions,
        weight_total=weight_total,
        risk_score=risk_score,
        thresholds=thresholds,
        half_life=cfg["recency_half_life_hours"],
    )

    return {
        "near_miss_probability": near_miss_probability,
        "accident_probability": accident_probability,
        "severity": severity,
        "event_age_hours": event_age_hours,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "present_inputs": sorted(present),
        "missing_inputs": missing,
        "weights_applied": {key: round(weights[key] / weight_total, 6) for key in present},
        "contributions": {key: round(scale * contributions[key] / weight_total, 4) for key in present},
        "recency_factor": None if values["recency"] is None else round(values["recency"], 6),
        "is_probability": False,
        "explanation": explanation,
    }


def _fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value}"


def _build_explanation(
    echo: dict[str, float | None],
    present: dict[str, float],
    missing: list[str],
    weights: dict[str, float],
    contributions: dict[str, float],
    weight_total: float,
    risk_score: float,
    thresholds: dict[str, float],
    half_life: float,
) -> list[str]:
    """Human-readable, formula-transparent rationale for the assessment."""
    lines = [
        (
            "risk_score is a heuristic fusion score, NOT a probability and NOT "
            "scientifically validated; it must not be used to control traffic signals."
        ),
        (
            "Formula: risk_score = scale * sum(weight_i * value_i for present inputs) "
            "/ sum(weight_i for present inputs)."
        ),
        (
            "near_miss_probability and accident_probability are kept semantically "
            "separate: they come from different models measuring different events "
            "and are never merged into a single probability."
        ),
        (
            "accident_probability is the calibrated accident probability from the "
            "Phase 6 detector; raw detector confidence is never treated as a probability."
        ),
        (
            "near-miss label semantics are unconfirmed in the source data (class_1 is "
            "only statistically consistent with a conflict/near-miss); the value is "
            "echoed from the Phase 4 artifact without added interpretation."
        ),
        (
            "severity is a caller-supplied relative value in [0, 1]; this pipeline "
            "does not measure severity and does not define its meaning."
        ),
    ]
    for key in INPUT_KEYS:
        label = {
            "near_miss": "near_miss_probability",
            "accident": "accident_probability",
            "severity": "severity",
            "recency": f"recency factor from event_age_hours (half-life {half_life} h)",
        }[key]
        if key in present:
            display = echo["event_age_hours"] if key == "recency" else echo[key]
            lines.append(
                f"Input {label} = {_fmt(display)}"
                f" -> contribution {contributions[key]:.4f} (weight {weights[key]})."
            )
        else:
            lines.append(f"Input {label} is missing -> excluded from the fusion.")
    if missing:
        lines.append(
            "Missing inputs are excluded and weights are renormalized over the "
            f"present inputs (weight total used: {weight_total:.4f}); absence of an "
            "input is treated as absence of evidence, not as evidence of absence."
        )
    lines.append(
        f"Fused score {risk_score:.4f} vs thresholds (inclusive lower bounds) "
        f"medium={thresholds['medium']}, high={thresholds['high']}, "
        f"critical={thresholds['critical']}."
    )
    return lines