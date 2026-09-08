# Member 4 Component Verification Audit

Date: 2026-09-08

Scope: read-only inspection of the existing accident, near-miss, and risk
components. No code, model, dataset, or existing report was modified. The
only new file is this requested audit.

## A. Verdict

| Component | Verdict | Status |
|---|---|---|
| Accident detection model | YES | CONFIRMED |
| Near-miss trained ML model | YES | CONFIRMED |
| Risk engine | YES | CONFIRMED |

The near-miss label semantics are not confirmed by the source data. The
implementation predicts the saved model's `class_1`; the project reports call
this statistically consistent with near-miss/conflict behavior but do not
claim authoritative label semantics.

## 1. Accident

### Status

**CONFIRMED**: a trained PyTorch object-detection checkpoint exists, it is
loaded by inference code, its output is calibrated into an event-level
accident probability, and it is connected to the Member 4 safety pipeline.

### Exact files found

- `models/accident/calibrated_accident_detector.pth`
- `models/accident/calibration_metadata.json`
- `models/accident/model_metadata.json`
- `models/accident/phase5b_expA_best.pth`
- `models/accident/phase5b_checkpoint_metadata.json`
- `src/accident/model.py`
- `src/accident/calibrated_detector.py`
- `src/accident/calibration.py`
- `src/accident/phase6.py`
- `reports/accident/phase6_summary.md`
- `reports/accident/phase6_summary.json`
- `reports/accident/calibration_results.md`
- `reports/accident/test_results_phase6.md`
- `reports/accident/probability_contract.md`
- `tests/test_accident_phase6.py`
- `src/config.py`
- `src/safety_intelligence.py`

### Model/framework/type

- Framework: PyTorch plus torchvision.
- Detector: `fasterrcnn_mobilenet_v3_large_320_fpn`, with a 3-class model
  head: background, accident, and non-accident.
- Training source: Phase 5B EXP_A checkpoint
  `models/accident/phase5b_expA_best.pth`.
- Calibration: Platt calibration of the `noisy_or_top5` accident-box signal.
- Final artifact: detector state dictionary plus calibration parameters in
  `calibrated_accident_detector.pth`.

### Direct artifact evidence

The existing checkpoint was loaded with `torch.load(..., weights_only=True)`.
Observed payload:

- file size: 76,042,385 bytes;
- keys: `format_version`, `model_state_dict`, `calibration`, `metadata`;
- `format_version`: `1`;
- state-dict entries: `284`;
- classifier bias shape: `(3,)`;
- calibration: method `platt`, signal `noisy_or_top5`, with stored `a` and `b`
  parameters;
- embedded metadata: source `phase5b_expA_best.pth`, fit split `validation`,
  fit image count `316`.

This is not an empty placeholder or a rules-only file. The state dictionary
contains the learned detector parameters and the calibrated artifact contains
the fitted probability mapping.

### What the component actually does

`src/accident/calibrated_detector.py`:

1. Loads the checkpoint with PyTorch.
2. Reconstructs the 3-class Faster R-CNN detector with `pretrained=False`.
3. Loads the saved state dictionary and switches the detector to evaluation
   mode.
4. Converts an image to the expected tensor representation and runs object
   detection.
5. Maps model labels back to dataset labels.
6. Aggregates accident-class scores using `noisy_or_top5` above a 0.05 floor.
7. Applies the saved Platt calibrator.
8. Returns detections, raw detector confidence, and
   `calibrated_accident_probability`.

The code explicitly keeps raw detector confidence separate from the
calibrated probability.

### Training and evaluation evidence

`reports/accident/calibration_metadata.json` records:

- selected source checkpoint: `phase5b_expA_best.pth`;
- validation selection metrics: accident AP@50 `0.7624`, mAP@50 `0.4686`,
  accident recall `0.5083`;
- calibration method `platt` and signal `noisy_or_top5`;
- calibration fit split `validation`;
- test data not used for calibration or threshold selection.

`reports/accident/test_results_phase6.md` records the frozen test evaluation:

- 325 test images, 324 accident-positive;
- accident AP@50 `0.6706`;
- accident precision `0.6949` and recall `0.6292` at threshold 0.50;
- event-level Brier score `0.0169` and log loss `0.0673`.

### Connection to the accident and safety pipelines

- `src/config.py` points `ACCIDENT_DETECTOR_PATH` to
  `models/accident/calibrated_accident_detector.pth`.
- `SafetyIntelligence._ensure_accident()` constructs
  `CalibratedAccidentDetector` from that path.
- `SafetyIntelligence.assess_image()` calls the detector, takes its
  `calibrated_accident_probability`, and passes it to `assess_event()`.
- `assess_event()` refuses a detector output that has only raw confidence and
  no calibrated probability.
- The risk engine receives the calibrated accident probability, never the raw
  detector confidence.

### Tests

`tests/test_accident_phase6.py` covers checkpoint loading, 3-class state-dict
compatibility, calibration metadata, probability bounds and monotonicity,
inference output schema, deterministic inference, leakage guards, and dataset
hash integrity. The focused command passed with the existing artifacts.

### Limitations

- The event-level calibration is weakly identified because approximately 99%
  of the accident test images are positive.
- The test report documents five missed positive images and one confident
  negative-image false positive at probability `0.994`.
- The probability estimates are conditional on a deployment distribution
  similar to the CCTV data and do not encode severity, accident count, or
  localization quality.
- `models/accident/model_metadata.json` describes the older Phase 5 baseline
  (`best_detector.pth`), including a 2-class metadata declaration and zero
  accident AP on its baseline evaluation. It is not the authoritative
  metadata for the calibrated Phase 6 artifact. The calibrated checkpoint's
  embedded metadata and `calibration_metadata.json` identify the actual
  Phase 5B 3-class source. This is a metadata naming/provenance caveat, not
  evidence that the calibrated artifact is absent.

## 2. Near-miss

### Status

**CONFIRMED**: the project contains a genuinely trained sklearn model and a
calibrated wrapper around it. The probability pipeline is model inference, not
only a probability formula or hand-written rule system.

### Exact files found

- `models/near_miss/random_forest_A.joblib`
- `models/near_miss/random_forest_A_calibrated.joblib`
- `models/near_miss/random_forest_A_calibrated_meta.json`
- Other trained artifacts: `random_forest_B.joblib`, `random_forest_D.joblib`,
  logistic-regression artifacts, gradient-boosting artifacts, and the dummy
  baseline.
- `src/near_miss/train.py`
- `src/near_miss/calibrate.py`
- `src/near_miss/evaluate.py`
- `src/near_miss/probability_pipeline.py`
- `reports/near_miss/model_evaluation.md`
- `reports/near_miss/model_results.json`
- `reports/near_miss/calibration_report.md`
- `reports/near_miss/probability_contract.md`
- `reports/near_miss/robustness_analysis.md`
- `tests/test_near_miss_phase3.py`
- `tests/test_near_miss_phase4.py`

### Model/framework/type

- Framework: scikit-learn serialized with joblib.
- Raw model: preprocessing `Pipeline` plus
  `RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
  class_weight="balanced", random_state=42)`.
- Selected configuration: `A_offline_full_no_fesafety`, 14 features.
- Calibrated model: `CalibratedProbabilityPipeline` containing the raw random
  forest pipeline and a `LogisticRegression` sigmoid/Platt calibrator.

Direct loading of the existing files observed:

- `random_forest_A.joblib`: sklearn `Pipeline`, steps `preprocess` and
  `model`, model type `RandomForestClassifier`, 300 estimators.
- `random_forest_A_calibrated.joblib`: local
  `CalibratedProbabilityPipeline`, method `sigmoid`, raw model
  `RandomForestClassifier`, calibrator `LogisticRegression`, 14 input
  features.

### What the component actually does

`src/near_miss/train.py` fits preprocessing and the classifiers on the train
split, evaluates train/validation/test probabilities, and serializes full
pipelines. It defines ten model runs, including the selected random forest.

`src/near_miss/calibrate.py` loads the selected raw random forest, fits the
calibrator on the validation split, compares raw/sigmoid/isotonic options, and
saves the calibrated wrapper.

`src/near_miss/probability_pipeline.py`:

1. Loads `random_forest_A_calibrated.joblib` with joblib.
2. Validates the required feature schema and rejects forbidden passthrough
   columns such as `sample_index`, `event_type`, and `split`.
3. Calls the serialized wrapper's `predict_proba()`.
4. Returns class-0 and class-1 probabilities that sum to approximately one,
   with `calibrated: true`.

There is no rule-based threshold or hand-written near-miss formula in this
inference path. The probability is generated by the trained random forest and
then transformed by the saved calibrator.

### Training and evaluation evidence

`reports/near_miss/model_evaluation.md` records the selected raw model's test
metrics at threshold 0.50: accuracy `0.867`, class-1 precision `0.713`,
class-1 recall `0.802`, ROC-AUC `0.929`, and PR-AUC `0.830`. It also records
that the dummy majority baseline has ROC-AUC `0.500` and zero class-1 recall.

`reports/near_miss/calibration_report.md` records sigmoid calibration fitted
on the validation split with 376 samples. Test Brier score is `0.089807` for
the calibrated output versus `0.099755` raw, and test ECE is `0.043901` versus
`0.086362` raw.

### Connection to the safety pipeline

- `src/config.py` points `NEAR_MISS_MODEL_PATH` to
  `models/near_miss/random_forest_A_calibrated.joblib`.
- `SafetyIntelligence._ensure_near_miss()` constructs the reusable
  `ProbabilityPipeline` from that path.
- `SafetyIntelligence._near_miss_probability()` selects the wrapper's class-1
  probability and passes it to `risk_engine.assess_risk()` as
  `near_miss_probability`.

### Tests

`tests/test_near_miss_phase3.py` and `tests/test_near_miss_phase4.py` cover
split integrity, model contracts, artifact loading, feature exclusion,
probability bounds, calibration behavior, and inference output. The focused
Phase 3/4 command passed with the existing artifacts.

### Limitations

- The source data does not authoritatively establish that class 1 means
  near-miss; the implementation intentionally exposes neutral class-1
  semantics.
- The selected A configuration includes `PET` and `FE_inv_PET`, which the
  reports identify as post-event/offline features. It is therefore not a
  real-time deployment model as-is.
- The raw random forest nearly memorizes its training data (reported train PR
  approximately `0.999`), which is why Phase 4 calibration was performed.
- Calibration uses a relatively small validation set of 376 samples.

## 3. Risk engine

### Status

**CONFIRMED**: an actual fusion and decision component is implemented and
tested. It consumes near-miss and accident outputs when those inputs are
provided, then produces `risk_score` and `risk_level`.

### Exact files found

- `src/risk_engine.py`
- `src/safety_intelligence.py`
- `src/config.py`
- `reports/safety/phase7_summary.md`
- `tests/test_risk_engine.py`

### Model/framework/type

This is not a trained ML risk model. It is a deterministic, configurable,
heuristic fusion/decision component implemented in Python.

Default fusion weights from `src/config.py` are:

- near-miss: `0.35`;
- accident: `0.40`;
- caller-supplied severity: `0.15`;
- recency: `0.10`.

Recency is derived as `0.5 ** (event_age_hours / 24.0)`. The score is scaled
to 0-100 and mapped to LOW, MEDIUM, HIGH, or CRITICAL at thresholds 25, 50,
and 75.

### What the component actually does

`src/risk_engine.py`:

- validates probabilities and caller-supplied values at the boundary;
- converts event age to a recency factor;
- excludes missing inputs and renormalizes weights over present inputs;
- refuses to produce a result when every input is missing;
- computes a transparent weighted score;
- maps the score to a risk level;
- returns contributions, applied weights, missing/present inputs, provenance
  fields, and an explanation;
- explicitly marks `is_probability` as false.

`src/safety_intelligence.py` is the orchestrator. Its `assess_event()` path
computes the near-miss probability from the trained sklearn artifact, accepts
the calibrated accident probability from a detector output or explicit
argument, and calls `assess_risk()`. Its `assess_image()` path runs the actual
accident detector first and then feeds that output through the same fusion
path. The returned object includes both model provenance and the final
`risk_score`/`risk_level`.

### Does it consume both model outputs?

**Yes, when supplied.** The two values remain semantically separate:

```text
near_miss model -> near_miss_probability
accident model  -> calibrated_accident_probability -> accident_probability
                         |
                         v
                  risk_engine.assess_risk()
                         |
                         v
                  risk_score / risk_level
                         |
                         v
                  SafetyIntelligence output
```

The orchestrator does not substitute raw accident detector confidence for
the calibrated probability. A detector output missing
`calibrated_accident_probability` is rejected.

### Tests

`tests/test_risk_engine.py` covers fusion math, thresholds, missing and
invalid inputs, semantic separation, near-miss artifact inference, accident
output handling, real detector image assessment, and combined real-model
assessment. The module passed with **36 tests**.

The complete existing project suite also passed with **119 tests, 0 failures**.

### Limitations

- The risk score is explicitly a heuristic engineering score, not a
  probability and not a scientifically validated risk model.
- No risk ground-truth labels, risk calibration, or predictive validation of
  the fusion score are present.
- Severity and event age are caller-supplied; the engine does not measure
  severity or infer it from either model.
- Missing inputs cause weight renormalization. This makes the calculation
  available with partial evidence but can understate risk when the strongest
  signal is unavailable; missing inputs are reported in the output.
- There is no traffic-signal control or actuation path.

## 4. Connection audit

The three components are connected in the implemented path:

1. `src/config.py` supplies both artifact paths.
2. `SafetyIntelligence` lazily loads the calibrated near-miss pipeline and
   calibrated accident detector.
3. `assess_event()` obtains `near_miss_probability` from the sklearn wrapper
   and `accident_probability` from the calibrated detector output.
4. `risk_engine.assess_risk()` fuses the present inputs and returns the score,
   level, contributions, and explanation.
5. `SafetyIntelligence` returns that assessment plus model provenance and, for
   image assessments, detector details.

The tests exercised this connection with real artifacts. The implementation
also supports partial assessments when one model input is unavailable; that
is intentional and is reported through `present_inputs` and `missing_inputs`.

## B. Rebuttal to the allegation

The allegation is not supported by the repository evidence. Accident is a
real trained Faster R-CNN detector with a saved 3-class state dictionary,
validation-fitted Platt calibration, inference code, evaluation results, and
an active connection to the safety pipeline. Near-miss is a real trained
scikit-learn random forest serialized in joblib, wrapped by a saved sigmoid
calibrator, and used by the probability pipeline; it is not only a
rule-based probability calculation. Risk is a real Python fusion and decision
component that consumes the two model-derived probabilities and emits a
heuristic score and level.

That rebuttal does not claim the models are production-ready or scientifically
validated. The repository itself documents the accident prevalence issue,
near-miss label ambiguity and post-event features, and the heuristic nature
of the risk score.

## C. Independent verification commands

Run these commands from `C:\Users\Varun\Desktop\acci` in PowerShell. They
only load existing artifacts or run tests; they do not train or modify them.

```powershell
$env:PYTHONPATH = "$PWD\src"

# Accident checkpoint, calibration, leakage guards, and inference contract
python -m pytest tests/test_accident_phase6.py -q

# Near-miss trained and calibrated artifacts
python -m pytest tests/test_near_miss_phase3.py tests/test_near_miss_phase4.py -q

# Risk fusion and real-artifact integration
python -m pytest tests/test_risk_engine.py -q

# Complete regression check
python -m pytest -q
```

To inspect the serialized model types directly:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -c "import joblib, torch; from pathlib import Path; p=Path('models/accident/calibrated_accident_detector.pth'); a=torch.load(p,map_location='cpu',weights_only=True); print(sorted(a)); print(a['calibration']); print(a['metadata']); r=joblib.load('models/near_miss/random_forest_A.joblib'); c=joblib.load('models/near_miss/random_forest_A_calibrated.joblib'); print(type(r).__name__, type(r.named_steps['model']).__name__, r.named_steps['model'].n_estimators); print(type(c).__name__, c.method, type(c.raw_pipeline.named_steps['model']).__name__, type(c.calibrator).__name__)"
```

Expected focused-test results from this audit:

- `tests/test_risk_engine.py`: 36 passed;
- Phase 3/4 near-miss plus Phase 6 accident tests: 48 passed;
- full suite: 119 passed, 0 failed.
