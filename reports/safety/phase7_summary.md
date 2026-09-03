# Phase 7 — Safety Risk Fusion (Member 4 layer)

Status: **complete**. Phase 7 consumed the two finished probability models
as-is (no retraining, no architecture change, no artifact modification) and
fused their outputs with caller-supplied severity and event recency into a
transparent, configurable risk score.

## What was created

| File | Purpose |
|---|---|
| `src/config.py` | All fusion weights, risk-level thresholds, score scale, recency half-life, artifact paths — nothing hardcoded elsewhere |
| `src/risk_engine.py` | Pure fusion function `assess_risk(...)`, level mapping, recency decay, input validation |
| `src/safety_intelligence.py` | Orchestrator `SafetyIntelligence`: loads the Phase 4 near-miss pipeline and the Phase 6 calibrated detector lazily, runs them, fuses via `assess_risk` |
| `tests/test_risk_engine.py` | 36 tests: fusion math, threshold boundaries, missing/invalid inputs, config overrides, JSON contract, integration with both real artifacts |
| `reports/safety/phase7_summary.md` | This report |

## Inputs used (both loaded read-only, never retrained)

1. `models/near_miss/random_forest_A_calibrated.joblib` — Phase 4 sigmoid-calibrated
   random forest (config `A_offline_full_no_fesafety`, 14 features), consumed through the
   existing `near_miss.probability_pipeline.ProbabilityPipeline`. Output:
   calibrated class-1 probability → `near_miss_probability`.
2. `models/accident/calibrated_accident_detector.pth` — Phase 6 calibrated accident
   detector (Phase 5B EXP_A Faster R-CNN + Platt calibration on the `noisy_or_top5`
   signal). Output: `calibrated_accident_probability` → `accident_probability`.
   The raw detector confidence (`accident_detection_confidence`) is **never** used as a
   probability; when a detector output is fused it is carried separately under
   `detector_raw_signal` with an explicit not-a-probability note, and a detector
   output lacking the calibrated field is refused rather than substituted.

## Fusion formula

```
recency_factor = 0.5 ** (event_age_hours / RECENCY_HALF_LIFE_HOURS)

risk_score = SCALE * Σ( weight_i × value_i  for present inputs i )
                    / Σ( weight_i          for present inputs i )

risk_level  = LOW      if risk_score < 25
              MEDIUM   if 25 ≤ risk_score < 50
              HIGH     if 50 ≤ risk_score < 75
              CRITICAL if risk_score ≥ 75
```

- Inputs: `near_miss_probability` (w 0.35), `accident_probability` (w 0.40),
  `severity` (w 0.15), recency from `event_age_hours` (w 0.10) — all from
  `src/config.py` (`RISK_WEIGHTS`, `RISK_LEVEL_THRESHOLDS`, `RISK_SCORE_SCALE`,
  `RECENCY_HALF_LIFE_HOURS`).
- Thresholds are **inclusive lower bounds** of their level; a score of exactly
  25 is MEDIUM, exactly 50 HIGH, exactly 75 CRITICAL (unit-tested).
- **Missing inputs** (`None`) are excluded and the weights of the *present*
  inputs are renormalized — absence of an input is treated as absence of
  evidence, not as evidence of absence. With no inputs at all the engine
  raises instead of returning a confident-looking LOW.
- Invalid inputs (probabilities outside [0, 1], negative event age, booleans,
  non-numeric severity, NaN/inf) raise `RiskInputError` — the engine never
  guesses. Severity is accepted only as a number in [0, 1]; no named-severity
  vocabulary is invented.
- Every assessment returns a JSON-serializable dict containing the required
  schema keys (`near_miss_probability`, `accident_probability`, `risk_score`,
  `risk_level`, `severity`, `explanation`) plus `present_inputs`,
  `missing_inputs`, `weights_applied`, `contributions`, `event_age_hours`,
  `recency_factor`, `model_provenance` and a formula-transparent
  `explanation` list.

## Model provenance

| Input | Source | Calibrated | Fit split |
|---|---|---|---|
| `near_miss_probability` | Phase 4 `random_forest_A_calibrated` (sigmoid) | yes | validation (Phase 4) |
| `accident_probability` | Phase 6 `calibrated_accident_detector` (Platt) | yes | validation (Phase 6) |
| `severity` | caller-supplied | n/a | n/a |
| `event_age_hours` | caller-supplied | n/a | n/a |

## Thresholds (defaults, configurable in `src/config.py`)

| Parameter | Default |
|---|---|
| Weights (near_miss / accident / severity / recency) | 0.35 / 0.40 / 0.15 / 0.10 |
| Level boundaries (inclusive lower bounds, 0–100 scale) | MEDIUM 25, HIGH 50, CRITICAL 75 |
| Recency half-life | 24 hours |

No threshold or weight was tuned on any dataset split. The test set was not
used to select or tune anything in Phase 7 (the fusion is an engineering
heuristic defined by `config.py`, fitted to nothing).

## Tests

- `tests/test_risk_engine.py`: **36 tests, all passing**.
- Full project suite after Phase 7: **119 passed, 0 failed** (Phase 3–6 suites
  unchanged and still green).
- Coverage includes boundary cases (score exactly on each threshold, inputs at
  0/1, event age 0 / half-life / underflow), missing inputs (all-missing raise;
  each partial combination renormalizes and is reported), invalid inputs
  (range, sign, type, bool, NaN/inf), config overrides (weights, thresholds,
  and rejected invalid configs), JSON serializability, semantic-separation
  assertions, and integration with the real Phase 4 and Phase 6 artifacts
  (including refusal to treat raw detector confidence as a probability and a
  clear error for incomplete near-miss feature vectors).
- The Phase 6 suite's dataset sha256 checks still pass, confirming the
  original Parquet files and all prior artifacts are untouched.

## Example output

```json
{
  "engine_version": "safety_risk_fusion_v1",
  "near_miss_probability": 0.711654,
  "accident_probability": null,
  "risk_score": 65.5281,
  "risk_level": "HIGH",
  "severity": 0.4,
  "event_age_hours": 6.0,
  "explanation": ["risk_score is a heuristic fusion score, NOT a probability ..."],
  "present_inputs": ["near_miss", "recency", "severity"],
  "missing_inputs": ["accident"]
}
```

## Limitations (explicit, per Phase 7 rules)

1. **Not scientifically validated.** The risk score is a heuristic engineering
   fusion. No ground truth exists for "risk", no calibration or validation of
   the score itself was performed, and no claim of predictive validity is made.
2. **Not for traffic-signal control.** This layer produces assessments only.
   No signal-control, actuation, or intervention path exists.
3. **Severity and recency are caller-supplied.** Neither source dataset
   carries severity labels or timestamps; the pipeline does not measure
   severity and does not define its meaning — the caller owns the [0, 1]
   contract. The recency half-life is a configurable heuristic.
4. **Near-miss label semantics remain unconfirmed.** Per the Phase 1/2 audit,
   class_1 is only *statistically consistent* with a conflict/near-miss; the
   value is echoed from the Phase 4 artifact without added interpretation.
5. **Inherited model caveats.** The near-miss probability carries the Phase 4
   caveats (PET is post-event, pixel distances are camera-dependent). The
   accident probability carries the Phase 6 caveats (event-level calibration
   is weakly identified at ~99% accident prevalence; the calibrator should be
   re-fit on deployment data with different prevalence).
6. **Linear fusion is a design choice**, not an empirically derived model:
   additive weights (rather than, e.g., multiplicative risk) were chosen for
   transparency and testability. Changing them requires no code changes, only
   `config.py`.
7. **Missing evidence is not absence of danger.** Renormalizing weights over
   present inputs can understate risk when the strongest signals (e.g.
   accident probability) are unavailable; `missing_inputs` is always reported
   so downstream consumers can gate on evidence completeness.

STOP after Phase 7.