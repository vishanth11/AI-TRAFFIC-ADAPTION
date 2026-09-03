# Member 4 — Final Work Report (Phases 1–7)

**Project:** Traffic Safety Intelligence — Near-Miss Detection, Accident Detection, and Safety Risk Fusion
**Member role:** Member 4 — Safety Intelligence / Model Layer
**Report date:** 2026-09-03
**Status:** Phases 1–7 complete · Phase 8 (this report) · Members 5–10 phases pending
**Verification:** every metric in this report was read from the project's own reports, JSON summaries, and model metadata on 2026-09-03, and the full test suite was re-run on that date (**119 passed, 0 failed**).

---

## Table of Contents

1. [Member 4 Role](#1-member-4-role)
2. [Datasets](#2-datasets)
3. [Phase-by-Phase Work](#3-phase-by-phase-work)
4. [Near-Miss System](#4-near-miss-system)
5. [Accident Detection System](#5-accident-detection-system)
6. [Safety Risk Fusion](#6-safety-risk-fusion)
7. [Final Member 4 Output](#7-final-member-4-output)
8. [What Is Actually Working](#8-what-is-actually-working)
9. [Testing](#9-testing)
10. [File / Artifact Inventory](#10-file--artifact-inventory)
11. [Limitations](#11-limitations)
12. [Member 5 Handoff](#12-member-5-handoff)
13. [Future Work](#13-future-work)
14. [Final Summary](#14-final-summary)

---

## 1. Member 4 Role

### 1.1 Purpose

Member 4 owns the **safety-intelligence modeling layer** of the project: the path from raw traffic data to a fused, explainable safety-risk assessment. Member 4 takes the two raw datasets (a DTC-FM traffic-conflict feature table and an accident CCTV object-detection corpus), audits and prepares them, trains and evaluates one probability model per dataset, calibrates both models' outputs into probabilities, and fuses the probabilities with caller-supplied severity and recency into a single transparent risk score with a LOW/MEDIUM/HIGH/CRITICAL level.

### 1.2 Responsibilities (as executed in Phases 1–7)

| Responsibility | Phases |
|---|---|
| Dataset audit, quality grading, leakage screening, dataset identification | 1 |
| Data preparation (processing artifacts, feature policy, split-integrity verification) | 2 |
| Near-miss train/validation/test split + model training + model selection | 3 |
| Near-miss probability calibration, error/robustness analysis, probability contract | 4 |
| Accident detector: baseline training, failure diagnosis, fix + improvement experiments | 5, 5B |
| Accident detector: checkpoint selection, threshold study, probability calibration, single frozen test evaluation | 6 |
| Risk fusion engine + orchestrator + config, integration of both finished models | 7 |

### 1.3 What Member 4 does NOT control

- **Traffic-signal control / actuation.** No Member 4 output may be used to control signals. Every artifact explicitly documents this (`risk_engine.py`, `probability_contract.md`, phase 6/7 reports). Signal-control and decision-making belong to later phases / Member 5.
- **Severity measurement.** Neither dataset carries severity labels. `severity` is a caller-supplied number in [0, 1]; Member 4 does not measure or define it.
- **Event timestamps.** `event_age_hours` is caller-supplied; Member 4 only applies the configured decay.
- **Label-semantics authority.** The DTC-FM class-1 semantics are unconfirmed; Member 4 deliberately does not hard-code an interpretation and reports this as an open item.
- **Dataset modification.** Original files under `dataset/` are read-only throughout; SHA-256 hashes are recorded and verified in Phase 6 metadata.
- **Accuracy optimization beyond what was done.** Detector accuracy work was explicitly deferred (Phase 6 rule: "no retraining, no architecture changes, no accuracy work").

---

## 2. Datasets

### 2.1 DTC-FM Near-Miss Dataset

| Property | Value (verified, Phase 1 audit) |
|---|---|
| Files | `dataset/filtered_features_DTC_FM.csv` (2,508 × 15), `dataset/target_labels_DTC_FM.csv` (2,508 × 1) |
| Samples | 2,508 interactions |
| Features | 13 numeric + 2 categorical (`class_object_1/2` ∈ {Car, VRU, Ped, LCV, HCV}), plus 4 engineered `FE_*` columns |
| Missing / infinite values | 0 / 0 |
| Duplicate rows | 5 groups (10 rows, 0.20%), identical features **and** identical labels; 0 conflicting labels |
| Alignment | Positional only — the labels file has no ID column |
| Label column | `event_type` ∈ {0.0, 1.0} |
| Class distribution | class_0: 1,867 (74.44%) · class_1: 641 (25.56%) — moderately imbalanced (~2.9:1) |
| Quality grade | **GOOD** |

**Confirmed/unconfirmed semantics.** The files contain **no label map**. Class-conditional statistics (class-1 rows have shorter PET — 1.85 s vs 2.38 s, smaller minimum distance — 2.04 m vs 6.26 m, smaller pixel distance — 26 px vs 104 px, larger encounter angle — 47.8° vs 29.5°) are *consistent with* "1 = conflict/near-miss", but this was never documentation-confirmed. All downstream artifacts therefore use neutral `class_0` / `class_1` terminology and carry an explicit unconfirmed-semantics caveat. This remains the single most important open data item for the project.

**Engineered features (reverse-engineered with evidence):**

| Feature | Verified formula | Max abs error | Status |
|---|---|---|---|
| `FE_inv_PET` | `1 / PET` | 3.6e-06 | redundant transform |
| `FE_dist_squared` | `target_dist_px²` | 5.8e-11 | redundant transform |
| `FE_log_mesafe` | `ln(1 + target_dist_px)` | 8.9e-16 | redundant transform |
| `FE_safety_index` | `(speed1 + speed2) / (2 · max(target_dist_px, 1e-6))` — resolved in Phase 3 | 0.0012 | redundant, numerically unstable; excluded from the primary config |

**Leakage screening (Phase 1 → 3):** no ID columns exist (no ID-based leakage possible); no target-derived features exist; `FE_safety_index` was downgraded MEDIUM → NONE once its formula was resolved (it is a pure transform of three *input* features, and including it never improved validation metrics); `PET` is a **post-event** measurement — a deployment-availability design concern, not statistical leakage; pixel-distance features are camera-geometry-dependent (domain-shift risk).

### 2.2 Accident CCTV Dataset

| Property | Value (verified, Phase 1 audit) |
|---|---|
| Source | Hugging Face `justjuu/traffic-accident-cctv-object-detection` (CC0-1.0; Roboflow "Accident and Non-accident" CCTV, letterboxed 640×640, train-only flip/brightness augmentation) |
| Format | 4 Parquet shards, HF image-object-detection schema: `image{bytes: PNG, path: null}` + `objects{bbox[], category[]}` |
| Images | 2,763 total, all 640×640 PNG (2,122 train / 316 validation / 325 test) |
| Objects | 5,400 COCO-style `[x, y, w, h]` absolute-pixel boxes (4,120 / 678 / 602 per split); 1–12 objects per image (avg 1.95) |
| Classes | `0 = accident`, `1 = non_accident` — confirmed from the source dataset card (not embedded in the files); recorded in `data/processed/accident/label_mapping.json` |
| Box validity | 100% of boxes within 640×640 bounds; bbox format verified empirically as xywh (not corners) |
| Duplicates | 26 exact-byte duplicate images **inside train only** (1.2%); **0 cross-split duplicates** (SHA-256 over all 2,763 PNGs) |
| Corruption | 0 (IHDR validated on all images, decode-sampled every 100th) |
| Schema | byte-identical Arrow schema across all 4 shards |
| Quality grade | **GOOD** |

**Object-level class distribution:**

| Split | accident | non_accident | accident % |
|---|---|---|---|
| train | 2,313 | 1,807 | 56.14% |
| validation | 338 | 340 | 49.85% |
| test | 391 | 211 | 64.95% |

**Image-level prevalence:** 98.59% (train) / 99.37% (validation) / 99.69% (test) of frames contain ≥1 accident object. Nearly every frame is accident-positive — this is why the accident task is framed as **object-level detection/localization**, and it later weakly identifies the event-level calibration (§5, §11). A train→test object-level accident-rate drift of +8.8 points is documented.

### 2.3 Dataset Splits

| Dataset | Split strategy | train | validation | test |
|---|---|---|---|---|
| DTC-FM near-miss | group-aware, deficit-based greedy stratified, seed 42, duplicate groups kept whole | 1,756 (70.02%) | 376 (14.99%) | 376 (14.99%) |
| Accident CCTV | pre-existing provider split, verified content-clean | 2,122 | 316 | 325 |

Near-miss split class ratios are preserved per split (74.4/25.6); integrity checks: 0 duplicate groups crossing splits, 0 identical feature rows crossing splits, 0 `sample_index` overlap, every sample assigned exactly once (1,756 + 376 + 376 = 2,508). The split is reproducible (`src/near_miss/split.py`, fixed seed, no library split shortcut).

### 2.4 Data-Quality and Leakage Checks (summary)

| Check | Near-miss (DTC-FM) | Accident CCTV |
|---|---|---|
| Missing values | 0 | 0 (no empty PNG bytes, no empty annotations) |
| Duplicates | 5 groups / 10 rows, benign, kept | 26 train-internal exact duplicates, kept, documented |
| Cross-split contamination | 0 (group-aware split) | 0 (SHA-256 verified) |
| Label-map availability | **UNCONFIRMED** (neutral class_0/class_1) | Confirmed via source dataset card |
| Leakage found | none statistical; PET post-event design concern; FE_safety_index resolved to a benign transform | none (no IDs/timestamps exist) |
| Split integrity | PASS | PASS (`split_integrity_report.json`) |
| Schema consistency | n/a (single CSV pair) | identical across 4 shards |

---

## 3. Phase-by-Phase Work

### Phase 1 — Dataset Audit

| Item | Content |
|---|---|
| **Objective** | Read-only audit of all six source files: identification, quality grading, duplicates, leakage screening, risk register for later phases. |
| **Work performed** | File-by-file inventory; CSV deep audit; Parquet metadata/streaming audit (no full load); bbox format probe; PNG integrity checks; SHA-256 dedup across all 2,763 images; engineered-feature reverse-engineering (evidence-based). |
| **Key decisions** | Treat the two dataset families as **separate modules** (never combined); classify the Parquet corpus as "D. Accident + object detection"; flag DTC-FM label semantics as unconfirmed and `FE_safety_index` as MEDIUM-suspicious; originals untouched. |
| **Methods** | `audit_scripts/csv_audit.py`, `parquet_audit.py`, `parquet_labels_audit.py`, `bbox_format_probe.py`; SHA-256 hashing; IHDR parsing; statistical class-conditional profiling. |
| **Outputs** | `reports/dataset_inventory.md`, `near_miss_dataset_audit.md`, `accident_dataset_audit.md`, `data_quality_report.md`, `dataset_audit.md`, `phase1_summary.json`. |
| **Tests** | None (audit-only phase; all checks are in-script verifications). |
| **Results** | All datasets graded **GOOD**; 0 missing/corrupt; 0 cross-split duplicates; risk register of 5 items for later phases; no models trained. |
| **Limitations** | DTC-FM label semantics unconfirmed; `FE_safety_index` formula unidentified at this point; no ground truth for "quality" beyond internal consistency checks. |

### Phase 2 — Data Preparation

| Item | Content |
|---|---|
| **Objective** | Produce processed, verified, model-ready artifacts without modifying originals. |
| **Work performed** | Positional DTC-FM merge (no sorting, no pre-merge drops) into `dtc_fm_combined.csv` + exact-float `.parquet` companion; per-feature keep/flag policy; accident index artifacts (`train/validation/test_index.parquet`) referencing `(source_file, row_index)` without copying image bytes; duplicate reports; split-integrity report; corruption report; data dictionary. |
| **Key decisions** | Preserve all rows (dedup deferred); nothing deleted, no scaler fitted, no feature selection; PET kept **with a post-event flag**; `FE_safety_index` flagged suspicious pending Phase 3; label mapping recorded with explicit unconfirmed status for DTC-FM and confirmed mapping for accident; byte-exact Parquet companion created because CSV round-trip shifts the last ULP of 23 floats (max 9e-16). |
| **Outputs** | `data/processed/near_miss/` (combined CSV/Parquet, duplicate report, label mapping, feature policy), `data/processed/accident/` (3 index parquets, duplicate report, split-integrity report, label mapping, corruption report), `reports/phase2_data_preparation.md`, `data_dictionary.md`, `class_balance_phase2.json`. |
| **Tests** | Post-preparation validation: features byte-identical in Parquet artifact; labels unchanged; index counts match scans (2,122/316/325); zero cross-split contamination; zero writes under `dataset/` — **all PASS**. |
| **Results** | 2,508 + 2,763 records processed with zero loss; 6 documented data-quality issues carried forward. |
| **Limitations** | DTC-FM alignment is positional (any filtering must preserve index alignment); camera-dependent pixel features noted; splitting deferred to Phase 3. |

### Phase 3 — Near-Miss Split + Model Training

| Item | Content |
|---|---|
| **Objective** | Create a leakage-safe split and train probability-producing near-miss/conflict classifiers. |
| **Work performed** | Group-aware stratified 70/15/15 split (seed 42); trained 10 scikit-learn pipelines (Dummy baseline + {Logistic Regression, Random Forest, Gradient Boosting} × {A, B, D}); full threshold analysis; feature importances; leakage review. |
| **Key decisions** | Configs: **A** = offline full minus `FE_safety_index` (14 features), **B** = A + `FE_safety_index` (15), **D** = deployment-realistic, PET-family excluded (12). Preprocessing (median imputation, standard scaling, one-hot with `handle_unknown="ignore"`) fitted on train only inside each saved pipeline. Class imbalance handled via class weights, not resampling. Selection on **validation metrics only** — no test metric entered model selection. |
| **Models** | DummyClassifier (majority), LogisticRegression, RandomForestClassifier, GradientBoostingClassifier (all exposing `predict_proba()`). |
| **Outputs** | `models/near_miss/{run_name}.joblib` (10 pipelines), splits + `split_assignments.csv`, `model_comparison.csv`, `model_results.json`, `threshold_analysis.csv`, `feature_importance.csv`, `split_report.md`, `leakage_review.md`, 5 figures. |
| **Tests** | `tests/test_near_miss_phase3.py` — 7 tests (split integrity, feature-list integrity, `sample_index` exclusion, pipeline contract). |
| **Results** | Test ROC-AUC 0.89–0.93 across real models vs 0.500 dummy; RF-A best (test ROC-AUC 0.929, PR-AUC 0.830); leakage review **PASS**. |
| **Limitations** | Label semantics unconfirmed (neutral class_0/class_1 metrics); RF memorizes train (train PR 0.999) → calibration deferred to Phase 4; models are offline-only while PET is included. |

### Phase 4 — Near-Miss Evaluation + Calibration

| Item | Content |
|---|---|
| **Objective** | Calibrate the selected model's probabilities, analyze errors and robustness, define the probability contract. |
| **Work performed** | Verified split integrity; compared raw/sigmoid/isotonic calibration fitted on validation (n=376); reliability diagrams; calibrated threshold analysis; FN/FP error analysis; validation→test robustness deltas; probability contract. |
| **Key decisions** | **Sigmoid (Platt) selected** — isotonic scored lower in-sample but showed MCE ≈ 0 on validation (overfit signature on n=376); sigmoid chosen for stability. Raw probabilities exposed as ranking scores; the risk engine (Phase 7) consumes calibrated probabilities, not hard thresholds. |
| **Outputs** | `models/near_miss/random_forest_A_calibrated.joblib` (+ meta JSON), `calibration_report.md`, `calibration_comparison.csv`, `threshold_analysis_calibrated.csv`, `error_analysis.md`, `robustness_analysis.md`, `probability_contract.md`, 4 figures. |
| **Tests** | `tests/test_near_miss_phase4.py` — 18 collected tests (calibration metrics, artifact round-trip, contract). |
| **Results** | See §4. Test Brier 0.0998 → 0.0898; test ECE 0.086 → 0.044 (sigmoid). |
| **Limitations** | Calibration fitted on a small validation set (376 samples); calibrated recall at t=0.5 drops to 0.6875 (recall-first operating points available at lower thresholds); label semantics still unconfirmed. |

### Phase 5 — Accident Detector Baseline

| Item | Content |
|---|---|
| **Objective** | Train and evaluate a first accident object detector. |
| **Work performed** | Environment setup (`.venv-accident`, Python 3.12.10, torch 2.14.0+cpu, CPU-only, 8 threads); Faster R-CNN training; COCO-style evaluation; debug visualizations. |
| **Key decisions** | `fasterrcnn_mobilenet_v3_large_320_fpn` (COCO-pretrained, `trainable_backbone_layers=3`), 320×320 inputs (CPU practicality), batch 4, SGD lr 5e-3, 2 epochs, seed 42. |
| **Outputs** | `models/accident/{best,final}_detector.pth`, `model_metadata.json`, `training_history.json`, `training_config.json`, validation/test results reports, `environment_report.md`, `accident_detection_contract.md`, debug + ground-truth figures. |
| **Tests** | `tests/test_accident_phase5.py` — 19 tests (record loading, box geometry, loss finiteness, tensor shapes, inference contract). |
| **Results** | Baseline was effectively broken: validation mAP@50 **0.0948**, accident AP@50 **0.0000**; test mAP@50 0.0987. See Phase 5B for the root cause. |
| **Limitations** | Misconfigured label mapping (diagnosed in 5B); 2 epochs is far from convergence; aggregate-only loss logging hid the problem; tests verified code mechanics, not learning correctness. |

### Phase 5B — Accident Detector Diagnosis + Improvement

| Item | Content |
|---|---|
| **Objective** | Root-cause the Phase 5 failure and fix it, keeping the baseline checkpoint immutable. |
| **Work performed** | Full-path box-pipeline audit; empirical prediction probe over the entire validation split; visual debugging of 24 validation images; label-space fix; two tracked experiments. |
| **Key decision / root cause** | **Confirmed label-mapping bug (primary cause):** torchvision detectors reserve internal label 0 for *background*. Phase 5 passed dataset labels (0=accident, 1=non_accident) un-shifted, so every accident annotation was supervised as background and the only foreground class the model ever saw was non_accident → accident AP = 0.0 *by construction*. Probe evidence: predicted model labels `{1: 981}` (label 0 never emitted), mean emitted confidence 0.156. Fix: `src/accident/labels.py` single source of truth with a +1 offset (dataset 0/1 → model 1/2, model 0 = background), `NUM_CLASSES = 3`, per-component loss logging, background-masked inference mapping labels back to dataset space. |
| **Experiments** | **EXP_A** — mapping fix only, 3 epochs (1,306 s): val mAP@50 **0.4686**, mAP@50:95 **0.2313**, accident AP@50 **0.7624**, accident recall 0.5083. **EXP_B** — fixed mapping, 25 epochs/early-stop 6/StepLR@8; interrupted after epoch 1 (val accident AP 0.7114, mAP@50 0.4047 — inferior to EXP_A; interrupted state retained for future accuracy work). **EXP_C** — not run: pretrained-backbone use was already verified in code/test. **EXP_D** — reserved, not needed. |
| **Outputs** | `src/accident/labels.py` (+ model/train/inference updates), `phase5b_expA_best.pth`, `phase5b_expB_best*.pth`, `phase5b_interrupted_checkpoint.pth`, `phase5b_checkpoint_metadata.json`, `phase5b_experiment_log.md`, `phase5b_experiments.csv`, histories + run logs, threshold-sweep table. |
| **Tests** | `tests/test_accident_phase5b.py` — 16 tests (mapping round-trip, no background labels in training targets, 3-class head width, full-dataset box validity, save/load, checkpoint selection, threshold analysis); all 19 Phase 5 tests still pass. |
| **Results** | The background/label collision alone explains the Phase 5 accident AP of 0; after the +1 offset, accident AP@50 = 0.76 after only 3 epochs with all other hyperparameters unchanged. |
| **Limitations** | 320×320 resolution kept (640 would be ~4× slower/epoch on this CPU); EXP_B incomplete; confidence values remain uncalibrated detector scores. |

### Phase 6 — Accident Evaluation + Confidence Calibration

| Item | Content |
|---|---|
| **Objective** | Select the best checkpoint on validation, derive and calibrate an event-level accident probability, run exactly ONE frozen test evaluation. **No retraining, no architecture changes, no accuracy work** (deferred). |
| **Work performed** | Checkpoint selection (validation-only, rule: max validation accident AP@50, mAP@50 tie-break); confidence threshold study; three candidate aggregation signals compared; Platt vs isotonic calibration with 5-fold CV tie-breaking; reliability tables; single frozen test run; SHA-256 verification that original Parquet files were unchanged. |
| **Key decisions** | Checkpoint: **`phase5b_expA_best.pth`** (val accident AP@50 0.7624 vs EXP_B 0.7114 vs baseline 0.0). Signal: **`noisy_or_top5`** (uncalibrated validation Brier 0.0427 vs max 0.0743, top3_mean 0.2084). Method: **Platt** (Brier gap vs isotonic 0.0002 = near tie; Platt CV log-loss 0.0382 vs isotonic 0.1058 decided it). Calibration fitted on **validation only**; test touched once, after freezing. |
| **Outputs** | `models/accident/calibrated_accident_detector.pth` (detector weights + calibrator), `calibration_metadata.json` (incl. dataset SHA-256 hashes), `calibration_results.md`, `threshold_analysis.csv`, `test_results_phase6.md`/`.json`, `phase6_summary.md`/`.json`, `probability_contract.md`, 3 figures. |
| **Tests** | `tests/test_accident_phase6.py` — 23 tests. Suite after Phase 6: **83 passed, 0 failed** (all 60 prior tests still green). |
| **Results** | See §5. |
| **Limitations** | Event-level calibration is weakly identified at ~99% accident prevalence; 5 of 324 accident-positive test images received no accident detection above the 0.05 floor; the single accident-negative test image scored 0.994 (confident event-level FP). |

### Phase 7 — Safety Risk Fusion

| Item | Content |
|---|---|
| **Objective** | Fuse the two finished probability models with caller-supplied severity and recency into a transparent, configurable risk score and level. Consume both models as-is — no retraining, no artifact modification. |
| **Work performed** | Built `src/config.py` (all fusion parameters), `src/risk_engine.py` (pure fusion + validation), `src/safety_intelligence.py` (lazy-loading orchestrator), 36 tests, this summary report. |
| **Key decisions** | Linear weighted fusion with renormalization over present inputs; severity and recency are caller-supplied with an explicit [0, 1] contract; raw detector confidence is *never* accepted as a probability (carried separately with a not-a-probability note; refusal if the calibrated field is absent); no weights/thresholds tuned on any split; the engine raises rather than guessing on invalid or all-missing inputs. |
| **Outputs** | `src/config.py`, `src/risk_engine.py`, `src/safety_intelligence.py`, `tests/test_risk_engine.py`, `reports/safety/phase7_summary.md`. |
| **Tests** | 36 risk-engine tests; full suite **119 passed, 0 failed**. |
| **Results** | See §6. |
| **Limitations** | The risk score is an engineering heuristic — no ground truth for "risk" exists, no calibration/validation of the score itself, no signal-control path. |

### Phase Summary Table

| Phase | Objective | Primary deliverables | Final result |
|---|---|---|---|
| 1 | Dataset audit | 5 audit reports | All datasets GOOD; 0 cross-split contamination |
| 2 | Data preparation | Processed indexes + combined DTC-FM + policy/quality artifacts | Zero data loss; all validation checks PASS |
| 3 | Near-miss split + models | 10 pipelines, split, comparison reports | RF-A selected: test ROC-AUC 0.929 |
| 4 | Calibration + evaluation | Calibrated RF artifact + contracts | Test Brier 0.0898, ECE 0.044 (sigmoid) |
| 5 | Accident baseline | Faster R-CNN checkpoint + reports | Broken baseline diagnosed (accident AP 0.0) |
| 5B | Fix + improve | Fixed 3-class training, EXP_A checkpoint | Val accident AP@50 0.7624 (from 0.0) |
| 6 | Calibration + frozen test | Calibrated detector + contracts | Test accident AP@50 0.6706; event Brier 0.0169 |
| 7 | Risk fusion | Risk engine + orchestrator | Transparent fusion, 36 tests, 119 total pass |

---

## 4. Near-Miss System

### 4.1 Preprocessing

Inside each saved scikit-learn pipeline (fitted on **train only**): median imputation → standard scaling for numeric features; one-hot encoding with `handle_unknown="ignore"` for `class_object_1`/`class_object_2` (HCV appears in only 2 rows — treated as rare). No scaler is fitted outside the pipeline; no feature selection beyond the documented config; `sample_index`, `event_type`, and `split` are stripped before modeling (verified by tests).

### 4.2 Features

Three configurations (all documented in `feature_policy.json` and `model_comparison.csv`):

| Config | n | Contents |
|---|---|---|
| **A** (primary, offline) | 14 | 9 base numeric + 2 categorical + `FE_inv_PET`, `FE_log_mesafe`, `FE_dist_squared` (excludes `FE_safety_index`) |
| **B** | 15 | A + `FE_safety_index` |
| **D** (deployment-realistic) | 12 | A minus `PET` and `FE_inv_PET` |

`FE_safety_index` is excluded from the primary configuration on parsimony and numerical-stability grounds (the 1e-6 distance floor produces quantized extremes up to 2.85e7); empirically it never improved any validation metric.

### 4.3 Split Strategy

Group-aware, deficit-based greedy stratified assignment, seed 42: duplicate feature-row groups (5 groups × 2 rows) are kept whole within one split; classes stratified to 74.4/25.6 per split; realized 70.02/14.99/14.99. Integrity: 0 crossing groups, 0 identical rows across splits, 0 sample_index overlap — all verified and re-verified in Phase 4.

### 4.4 Models Tested

10 pipelines (all seed 42): DummyClassifier (majority) + {Logistic Regression, Random Forest, Gradient Boosting} × {A, B, D}. Full table: `reports/near_miss/model_comparison.csv`.

### 4.5 Selected Model

**`random_forest_A`** (config A, 14 features) → **`random_forest_A_calibrated`** (sigmoid, Phase 4). Selection used **validation metrics only**:

1. Best validation ROC-AUC (0.9305) and PR-AUC (0.8540) among all real models.
2. Best validation F1 for the minority class (0.7817).
3. Not the best raw recall (LR-A 0.865) — but the threshold analysis shows RF-A reaches the same recall band with higher precision at t≈0.30.
4. Stable val→test generalization (ROC 0.9305 → 0.9290; PR 0.8540 → 0.8304).
5. Documented caveat: RF fits train almost perfectly (train PR ≈ 0.9996) — the reason Phase 4 calibration was mandatory.

### 4.6 Validation / Test Metrics

**Phase 3 experiment table (test split, threshold 0.50, minority class = class_1):**

| Model | Config | Accuracy | Precision (c1) | Recall (c1) | F1 (c1) | ROC-AUC | PR-AUC | FN | FNR |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (majority) | A | 0.745 | 0.000 | 0.000 | 0.000 | 0.500 | 0.256 | 96 | 1.000 |
| Logistic Regression | A | 0.806 | 0.586 | 0.813 | 0.681 | 0.912 | 0.818 | 18 | 0.188 |
| Logistic Regression | B | 0.809 | 0.590 | 0.823 | 0.687 | 0.913 | 0.813 | 17 | 0.177 |
| Logistic Regression | D (deploy) | 0.790 | 0.559 | 0.833 | 0.669 | 0.898 | 0.773 | 16 | 0.167 |
| **Random Forest** | **A** | **0.867** | **0.713** | 0.802 | **0.755** | **0.929** | **0.830** | 19 | 0.198 |
| Random Forest | B | 0.862 | 0.693 | 0.823 | 0.752 | 0.930 | 0.835 | 17 | 0.177 |
| Random Forest | D (deploy) | 0.854 | 0.688 | 0.781 | 0.732 | 0.912 | 0.777 | 21 | 0.219 |
| Gradient Boosting | A | 0.846 | 0.661 | 0.813 | 0.729 | 0.916 | 0.836 | 18 | 0.188 |
| Gradient Boosting | B | 0.840 | 0.650 | 0.813 | 0.722 | 0.917 | 0.839 | 18 | 0.188 |
| Gradient Boosting | D (deploy) | 0.843 | 0.655 | 0.813 | 0.726 | 0.901 | 0.749 | 18 | 0.188 |

**Confusion matrix — random_forest_A, TEST, t=0.50 (Phase 3 raw model):** TN 249 · FP 31 · FN 19 · TP 77.

**Validation ranking:** RF-A best ROC-AUC 0.9305 / PR-AUC 0.8540 / F1 0.7817; LR-A best recall 0.8646 (FN 13). Validation→test deltas for every model are tabulated in `robustness_analysis.md`; all models generalize within ~0.01 ROC-AUC.

### 4.7 Threshold Analysis

**Raw model (validation, random_forest_A):**

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.20 | 0.467 | 0.948 | 0.625 | 104 | 5 |
| 0.30 | 0.569 | 0.906 | 0.699 | 66 | 9 |
| 0.40 | 0.675 | 0.865 | 0.758 | 40 | 13 |
| 0.50 | 0.762 | 0.802 | 0.782 | 24 | 19 |
| 0.60 | 0.783 | 0.750 | 0.766 | 20 | 24 |
| 0.70 | 0.816 | 0.646 | 0.721 | 14 | 34 |
| 0.80 | 0.869 | 0.552 | 0.675 | 8 | 43 |

**Calibrated model (validation, sigmoid):** (from `threshold_analysis_calibrated.csv`)

| Threshold | Precision | Recall | F1 | FPR | FN |
|---|---|---|---|---|---|
| 0.20 | 0.646 | 0.875 | 0.743 | 0.164 | 12 |
| 0.30 | 0.736 | 0.8125 | 0.772 | 0.100 | 18 |
| 0.50 | 0.787 | 0.729 | 0.757 | 0.068 | 26 |

**No deployment threshold is permanently selected.** Documented operating-point references: t≈0.30 recall-first (raw: recall 0.906, FN 9); t≈0.50 balanced (raw: F1 max 0.782). The risk engine consumes raw probabilities, not a hard threshold, so these are references, not decisions.

### 4.8 Calibration

Calibration fitted on **validation only** (n = 376):

| Method | Val Brier | Test Brier | Val log-loss | Test log-loss | Val ECE | Test ECE | Test MCE |
|---|---|---|---|---|---|---|---|
| raw | 0.096016 | 0.099755 | 0.308279 | 0.321270 | 0.078816 | 0.086362 | 0.252102 |
| **sigmoid (selected)** | 0.088684 | **0.089807** | 0.298774 | **0.301506** | 0.046262 | **0.043901** | **0.145450** |
| isotonic | 0.080224 | 0.091872 | 0.250184 | 0.618604 | 0.000000 | 0.027530 | 0.572400 |

Isotonic was rejected despite lower in-sample scores: validation MCE ≈ 0 indicated overfit on the small calibration set (its test log-loss of 0.6186 confirms this).

**Calibrated test classification (t = 0.50):** accuracy 0.8777, precision 0.8049, recall 0.6875, F1 0.7416, ROC-AUC 0.9290 (unchanged — calibration is monotone), PR-AUC 0.8304 (unchanged), TP/TN/FP/FN = 66/264/16/30.

### 4.9 Error and Robustness Analysis

- **False negatives (30, calibrated, t=0.5):** associated with long PET (mean 2.18 s), moderate distances (mean min-dist 3.49 m), mid-range angles (mean 41.9°); predicted probabilities 0.046–0.499 (median 0.290). Full profile: `error_analysis.md`.
- **False positives (16):** associated with very small distances (mean min-dist 1.41 m, mean pixel distance 15.6 px) and large angles (mean 53.6°) — physically close interactions the model ranks as conflicts.
- **Robustness:** validation→test ROC-AUC deltas within ±0.008 for all nine real models (`robustness_analysis.md`); rankings are stable.
- **Feature importance (RF-A, association not causation):** PET 0.377, `min_dist_dual_check_m` 0.244, `target_dist_px` 0.121, `FE_log_mesafe` 0.088, `angle_degrees` 0.061.

### 4.10 Final Probability Output

`predict_proba()[:, 1]` from `random_forest_A_calibrated.joblib` — calibrated P(class_1), contract in `probability_contract.md`:

```json
{"class_0_probability": 0.12, "class_1_probability": 0.88,
 "model_version": "random_forest_A_calibrated", "calibrated": true}
```

### 4.11 Near-Miss Limitations

1. **PET is post-event/offline.** PET (and `FE_inv_PET`) is computable only after the interaction completes. Config D quantifies the cost of removing it: ~0.02 ROC-AUC and ~0.06–0.08 PR-AUC lost, but still ROC ≥ 0.90. **No model trained here may be described as real-time-deployable while PET is included.** A live system needs pre-event surrogates (projected time-to-collision, distance rate) — future work.
2. **Label semantics unconfirmed** — class_1 is only *statistically consistent* with a conflict/near-miss.
3. **Pixel features are camera-dependent** (`target_dist_px`, `FE_dist_squared`, `FE_log_mesafe`); metric features are the safer core.
4. **RF overfit** (train PR 0.999) — addressed by calibration for probabilities, but it remains a variance caveat.
5. Test split is small for the minority class (96 class-1 samples); metric uncertainty is material.

---

## 5. Accident Detection System

### 5.1 Why Object Detection, Not Image Classification

Image-level classification would be trivially biased: **~99% of frames** (train 98.59%, validation 99.37%, test 99.69%) contain at least one accident-labeled object. A classifier predicting "accident" for every frame would score ~99% accuracy while carrying no information. The meaningful task is **object-level accident localization** — which object, where (bounding box), and with what confidence. The dataset is annotated per-object (5,400 COCO-style boxes), so detection uses the supervision that actually exists.

### 5.2 Architecture

- **Model:** torchvision `fasterrcnn_mobilenet_v3_large_320_fpn` — Faster R-CNN with a MobileNetV3-Large-320 backbone and FPN neck.
- **Initialization:** COCO-pretrained (`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`), `trainable_backbone_layers=3`; 18,871,333 trainable parameters.
- **Label spaces (after the Phase 5B fix):** dataset {0 = accident, 1 = non_accident} ↔ model {0 = background, 1 = accident, 2 = non_accident}, via a +1 offset in `src/accident/labels.py` (torchvision reserves internal label 0 for background).
- **Training config (Phase 5 / EXP_A):** 320×320 input, batch 4, SGD (lr 0.005, momentum 0.9, weight decay 5e-4), StepLR, seed 42, CPU-only (torch 2.14.0+cpu, 8 threads, no CUDA).
- **Inference:** `AccidentDetector` infers head width from the checkpoint, masks background outputs, maps model labels back to dataset categories; boxes returned as `[x1, y1, x2, y2]` absolute pixels.

### 5.3 Phase 5 Baseline → Phase 5B Improvement

| Metric (validation) | Phase 5 baseline | 5B EXP_A (fix, 3 epochs) |
|---|---|---|
| mAP@50 | 0.0948 | **0.4686** |
| mAP@50:95 | 0.0374 | **0.2313** |
| accident AP@50 | 0.0000 | **0.7624** |
| accident recall (COCO AR) | 0.0000 | **0.5083** |
| test mAP@50 (baseline) | 0.0987 | — |

**Root cause (verified, primary):** the Phase 5 model used `NUM_CLASSES = 2` with dataset labels passed un-shifted, so every accident annotation was supervised as **background**; the only foreground class the model ever saw was non_accident, and accident AP was 0.0 *by construction*. Probe over the full validation split with the immutable baseline checkpoint: predicted model labels `{1: 981}` (label 0 never emitted), mean emitted confidence 0.156. Bounding boxes were verified correct along the entire path (a new test validates every box in every split); the model was already COCO-pretrained; the 2-epoch duration was a real but secondary limitation.

**Phase 5B experiments** (validation only; TEST locked):

| Experiment | Change | Val mAP@50 | Val accident AP@50 | Status |
|---|---|---|---|---|
| EXP_A | label-mapping fix only, 3 epochs | 0.4686 | **0.7624** | completed (1,306 s) |
| EXP_B | fix + 25 epochs, early stop 6, StepLR@8 | 0.4047 | 0.7114 | interrupted after epoch 1 (checkpoint retained) |
| EXP_C | pretrained-backbone comparison | — | — | not needed (already pretrained, verified) |

Early EXP_A threshold sweep (validation, pooled classes, IoU 0.5): t=0.10 → P 0.267 / R 0.631; t=0.30 → P 0.512 / R 0.516; t=0.50 → P 0.695 / R 0.441.

### 5.4 Selected Checkpoint

**`models/accident/phase5b_expA_best.pth`** — selected in Phase 6 by max validation accident AP@50 (0.7624 vs EXP_B 0.7114 vs baseline 0.0000), mAP@50 tie-break. The Phase 5 baseline checkpoint is kept immutable; the deployed artifact is `models/accident/calibrated_accident_detector.pth` (detector weights + fitted calibrator).

### 5.5 Validation Metrics (Phase 6, selected checkpoint)

| Metric | Value |
|---|---|
| mAP@50 | 0.4686 |
| mAP@50:95 | 0.2313 |
| mAP@75 | 0.2121 |
| accident AP@50 | 0.7624 |
| accident AP@50:95 | 0.3915 |
| accident recall (COCO AR) | 0.5083 |
| non_accident AP@50 | 0.1749 |

**Confidence threshold study (validation, accident class, IoU 0.5):**

| Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|
| 0.1 | 0.4348 | 0.8580 | 377 | 48 |
| 0.2 | 0.5644 | 0.8432 | 220 | 53 |
| 0.3 | 0.6447 | 0.8107 | 151 | 64 |
| 0.4 | 0.7038 | 0.7663 | 109 | 79 |
| 0.5 | 0.7590 | 0.7456 | 80 | 86 |
| 0.6 | 0.8087 | 0.7130 | 57 | 97 |
| 0.7 | 0.8388 | 0.6775 | 44 | 109 |
| 0.8 | 0.8680 | 0.6420 | 33 | 121 |
| 0.9 | 0.9118 | 0.5503 | 18 | 152 |

Usable precision/recall trade-off sits in the 0.30–0.50 band. A deployment threshold is **not** fixed — downstream consumers pick one from this table per their operating point.

### 5.6 Final Test Metrics (single frozen evaluation)

| Metric | Value |
|---|---|
| mAP@50 | 0.4381 |
| mAP@50:95 | 0.2116 |
| accident AP@50 | 0.6706 |
| accident AP@50:95 | 0.3344 |
| accident recall (COCO AR) | 0.4409 |
| non_accident AP@50 | 0.2056 |
| accident precision @0.5 | 0.6949 |
| accident recall @0.5 | 0.6292 |

### 5.7 Calibration Method

- **Signal:** `noisy_or_top5` — noisy-OR aggregation of the top-5 accident-class box scores (above the 0.05 detector floor). Selected on lowest uncalibrated validation Brier (0.0427 vs 0.0743 for max, 0.2084 for top3_mean).
- **Method:** **Platt** (a = 0.48997, b = 3.87115), fitted on validation image-level labels (316 images, 314 accident-positive). Platt vs isotonic: validation Brier 0.0063 vs 0.0061 (near tie ≤ 0.002) → decided by 5-fold CV (Platt CV log-loss 0.0382 vs isotonic 0.1058; isotonic step functions risk extreme held-out probabilities).
- **Validation fit:** Brier 0.0063, log loss 0.0356, ECE 0.0000.
- **TEST was touched exactly once**, after the checkpoint, signal, and calibrator were frozen. Nothing was tuned after the test numbers were produced.

### 5.8 Calibrated Accident Probability (test, event level)

| Metric | Value |
|---|---|
| Images | 325 (324 accident-positive) |
| Brier score | 0.0169 |
| Log loss | 0.0673 |
| Expected calibration error (10 bins) | 0.0176 |
| Reliability | 5 images in bin [0.0, 0.1) (predicted 0.052, observed 1.0 — all misses); 320 in bin [0.9, 1.0] (predicted 0.9938, observed 0.9969) |

**Important caveat (documented in `calibration_results.md`):** because ~99% of images are accident-positive, event-level Brier/log-loss/ECE are dominated by prevalence and are close to what a well-behaved constant predictor achieves. Discrimination is better read from the object-level AP and the threshold study. **The calibrator must be re-fit on any deployment data whose accident prevalence differs materially.**

### 5.9 Inference Output

Contract (`accident/probability_contract.md`); the two confidence quantities are distinct:

- `accident_detection_confidence` — raw `noisy_or_top5` signal, a **ranking score, not a probability**.
- `calibrated_accident_probability` — P(image contains an accident), through the fitted Platt calibrator.

```json
{
  "detector_version": "accident_calibrated_detector_v1",
  "image_size": [640, 640],
  "detections": [
    {"bbox": [x1, y1, x2, y2], "category": "accident", "confidence": 0.83}
  ],
  "accident_detection_confidence": 0.83,
  "calibrated_accident_probability": 0.71,
  "calibration": {"method": "platt", "signal": "noisy_or_top5", "fit_split": "validation"}
}
```

Per-box `confidence` values remain uncalibrated detector scores. The output does not encode severity, accident count, or localization quality.

### 5.10 Accident Limitations

1. **Modest absolute accuracy.** Test accident AP@50 0.6706, mAP@50 0.4381, accident recall (COCO AR) 0.4409 — usable but far from production grade; accuracy work was explicitly deferred.
2. **Event-level calibration is weakly identified** (~99% prevalence); the calibrator is prevalence-bound.
3. **Failure modes (test, descriptive):** 5 of 324 accident-positive images received no accident detection above the 0.05 floor (probability floor 0.052 — these dominate log loss); the single accident-negative image scored 0.994 (confident event-level false positive).
4. **Small validation/test sets** (316/325 images) → metric variance is material.
5. **320×320 training resolution** (source 640×640) — a CPU-practicality compromise.
6. **Letterboxed source frames** — scale information is partially baked in.
7. **Distribution shift risk** — the probability contract conditions validity on deployment frames resembling the dataset distribution.

---

## 6. Safety Risk Fusion (Phase 7)

### 6.1 What is fused

Four inputs, deliberately kept semantically separate (different models measuring different events are never merged into one probability):

| Input | Source | Calibrated | Fit split |
|---|---|---|---|
| `near_miss_probability` | Phase 4 `random_forest_A_calibrated` (sigmoid) — calibrated P(class_1) | yes | validation |
| `accident_probability` | Phase 6 `calibrated_accident_detector` (Platt) — `calibrated_accident_probability` | yes | validation |
| `severity` | **caller-supplied**, relative value in [0, 1]; neither dataset carries severity labels | n/a | n/a |
| `event_age_hours` | **caller-supplied**; converted to a recency factor | n/a | n/a |

The raw detector confidence (`accident_detection_confidence`) is never accepted as a probability. When a detector output is fused, it is carried separately under `detector_raw_signal` with an explicit not-a-probability note; a detector output lacking the calibrated field is **refused** rather than substituted with the raw signal.

### 6.2 Actual Formula (exactly as implemented in `src/risk_engine.py`)

```
recency_factor = 0.5 ** (event_age_hours / RECENCY_HALF_LIFE_HOURS)

risk_score = SCALE * Σ( weight_i × value_i   for PRESENT inputs i )
                   / Σ( weight_i             for PRESENT inputs i )

risk_level  = LOW       if risk_score <  25
              MEDIUM    if 25 ≤ risk_score < 50
              HIGH      if 50 ≤ risk_score < 75
              CRITICAL  if risk_score ≥ 75
```

- Thresholds are **inclusive lower bounds** of their level (score exactly 25 → MEDIUM, 50 → HIGH, 75 → CRITICAL; unit-tested).
- Missing inputs (`None`) are **excluded** and the weights of the present inputs are **renormalized** — absence of an input is treated as absence of evidence, not evidence of absence. With no inputs at all the engine raises `RiskInputError` instead of returning a confident-looking LOW.
- Invalid inputs (probabilities outside [0, 1], negative event age, booleans, non-numeric severity, NaN/inf) raise `RiskInputError` — the engine never guesses. Severity is accepted only as a number in [0, 1]; no named-severity vocabulary is invented.

### 6.3 Configurable Thresholds (all in `src/config.py`; nothing hardcoded in the engine)

| Parameter | Key | Default |
|---|---|---|
| Weight — near-miss probability | `RISK_WEIGHTS["near_miss"]` | 0.35 |
| Weight — accident probability | `RISK_WEIGHTS["accident"]` | 0.40 |
| Weight — severity | `RISK_WEIGHTS["severity"]` | 0.15 |
| Weight — recency | `RISK_WEIGHTS["recency"]` | 0.10 |
| Score scale | `RISK_SCORE_SCALE` | 100.0 |
| MEDIUM threshold (lower bound) | `RISK_LEVEL_THRESHOLDS["medium"]` | 25.0 |
| HIGH threshold (lower bound) | `RISK_LEVEL_THRESHOLDS["high"]` | 50.0 |
| CRITICAL threshold (lower bound) | `RISK_LEVEL_THRESHOLDS["critical"]` | 75.0 |
| Recency half-life | `RECENCY_HALF_LIFE_HOURS` | 24.0 hours |

All are overridable per call via the `config` / `risk_config` argument (validated: weights > 0 and finite; 0 < medium < high < critical; scale > 0; half-life > 0). **No weight or threshold was tuned on any dataset split** — the fusion is an engineering heuristic defined by `config.py`, fitted to nothing.

### 6.4 Worked Example (verified against the implementation)

Inputs: `near_miss_probability = 0.18146`, `accident_probability = 0.9959`, `severity = 0.4`, `event_age_hours = 6.0` (all four present; weights total 1.0).

```
recency_factor = 0.5 ** (6.0 / 24.0) = 0.840896
risk_score = 100 × (0.35×0.18146 + 0.40×0.9959 + 0.15×0.4 + 0.10×0.840896) / 1.0
           = 100 × (0.063511 + 0.398360 + 0.060000 + 0.084090)
           = 60.5961  →  HIGH
```

This is an actual recorded output of `SafetyIntelligence.assess_image(...)` on real test data (see §7).

### 6.5 Fusion Limitations

1. **Not scientifically validated** — no ground truth for "risk" exists; the score is a transparent heuristic, echoed as `is_probability: false` in every output.
2. **Not for traffic-signal control** — the layer produces assessments only.
3. **Severity and recency are caller-supplied** — the pipeline measures neither.
4. **Linear fusion is a design choice** for transparency/testability, not an empirically derived model.
5. **Missing evidence is not absence of danger** — renormalization can understate risk when the strongest signal (e.g., accident probability) is unavailable; `missing_inputs` is always reported so consumers can gate on evidence completeness.
6. Inherits all near-miss and accident caveats (§4.11, §5.10).

---

## 7. Final Member 4 Output (handoff artifact for Member 5)

### 7.1 What Member 4 produces

The public API is `SafetyIntelligence` (`src/safety_intelligence.py`):

- `assess_event(...)` — fuse a near-miss feature vector and/or a Phase 6 detector output (or an explicit calibrated accident probability) with caller-supplied `severity` and `event_age_hours`.
- `assess_image(image, ...)` — run the calibrated accident detector on an image (PIL / numpy / tensor), then fuse as above.
- `to_json(output)` — serialize; outputs are plain builtins, JSON-serializable by construction.

Every assessment carries: `engine_version`, both probabilities, `risk_score` (0–100), `risk_level`, `severity`, `event_age_hours`, `present_inputs` / `missing_inputs`, `model_provenance` (artifact paths, loaded flags, `retrained: false`), an optional `detector_raw_signal` block (explicitly not-a-probability), the full detector output under `detector_output` (detections with bboxes, categories, confidences, calibration block), and a formula-transparent `explanation` list.

### 7.2 Example Output — ACTUAL RECORDED INFERENCE

The JSON below is **not hand-written**. It was produced on 2026-09-03 by running the real deployed artifacts on real data: the first test-split DTC-FM row (`sample_index` = first row of `data/processed/near_miss/splits/test.parquet`; features: speeds 4/8 km/h, angle 29.19°, PET 2.87 s, min-dist 4.677 m, 48 objects in frame, Car/VRU pair) and the first test-split CCTV image (`test-00000-of-00001.parquet`, row 0, 1 annotated accident object), with caller-supplied `severity = 0.4` and `event_age_hours = 6.0`. The `explanation` list is omitted here only for brevity; it is present in the real output.

```json
{
  "engine_version": "safety_risk_fusion_v1",
  "near_miss_probability": 0.18146,
  "accident_probability": 0.9959,
  "risk_score": 60.5961,
  "risk_level": "HIGH",
  "severity": 0.4,
  "event_age_hours": 6.0,
  "present_inputs": ["accident", "near_miss", "recency", "severity"],
  "missing_inputs": [],
  "model_provenance": {
    "near_miss_artifact": "models/near_miss/random_forest_A_calibrated.joblib",
    "near_miss_model_loaded": true,
    "accident_checkpoint": "models/accident/calibrated_accident_detector.pth",
    "accident_model_loaded": true,
    "retrained": false
  },
  "detector_raw_signal": {
    "accident_detection_confidence": 0.9646,
    "note": "Raw detector confidence (ranking signal aggregated from accident-class box scores). This is NOT a probability; the probability is 'accident_probability' above (calibrated)."
  },
  "detector_output": {
    "detector_version": "accident_calibrated_detector_v1",
    "image_size": [640, 640],
    "n_detections": 1,
    "detections": [
      {"bbox": [76.3, 127.22, 191.61, 168.67], "category": "accident", "confidence": 0.9322}
    ],
    "calibration": {"method": "platt", "signal": "noisy_or_top5", "fit_split": "validation"}
  }
}
```

(The example in `reports/safety/phase7_summary.md` — near-miss 0.711654, no accident input, risk 65.5281 HIGH — is likewise a real output of the fusion engine with a near-miss-only input.)

### 7.3 Contract Guarantees

- Probabilities outside [0, 1], invalid types, negative ages, NaN/inf → `RiskInputError`; all inputs missing → refusal (never a confident LOW).
- Raw detector confidence is never substituted for a probability.
- The two probabilities are never merged into one; `is_probability: false` marks the fused score.
- Deterministic inference for fixed inputs (eval mode, CPU); original datasets are never modified.

---

## 8. What Is Actually Working

### Implemented and tested

| Capability | Evidence |
|---|---|
| Read-only dataset audit with quality grades, duplicate/leakage screening | Phase 1 reports; SHA-256 checks |
| Deterministic data preparation with zero data loss and verified originals | Phase 2 report; post-write validation; dataset SHA-256 in Phase 6 metadata |
| Leakage-safe, reproducible 70/15/15 near-miss split | `split_report.md`; 0 crossing groups (tested) |
| 10 trained near-miss pipelines with saved preprocessing | `models/near_miss/*.joblib`; Phase 3 tests |
| Calibrated near-miss probability model (sigmoid, val-fit) | `random_forest_A_calibrated.joblib`; test Brier 0.0898, ECE 0.0439 |
| Accident detector with corrected label mapping (Faster R-CNN, 3-class) | `phase5b_expA_best.pth`; val accident AP@50 0.7624 |
| Event-level calibrated accident probability (Platt, noisy_or_top5, val-fit) | `calibrated_accident_detector.pth`; test Brier 0.0169 |
| Transparent risk fusion with validation, renormalization, explanation strings | `risk_engine.py`; 36 tests |
| End-to-end orchestrator producing the Member 5 payload | `safety_intelligence.py`; recorded output in §7.2 |
| Full regression suite | **119 passed, 0 failed** (re-run 2026-09-03) |

### Implemented but not fully validated

| Item | Gap |
|---|---|
| Deployment-realistic near-miss variant (config D, no PET) | Trained and evaluated offline (ROC ≥ 0.90) but never deployed/validated live; PET replacement not built |
| Accident detector accuracy | Usable but modest (test mAP@50 0.4381); accuracy work explicitly deferred; EXP_B long-run incomplete |
| Event-level accident calibration | Weakly identified at ~99% prevalence; validity conditioned on distribution match |
| Risk score semantics | Heuristic fusion; no ground truth, no external validation of score quality |
| Severity/recency inputs | Accepted and fused per contract, but no data source supplies them — caller-owned |

### Planned / future work

See §13.

---

## 9. Testing

### 9.1 Test Totals (re-run 2026-09-03: **119 passed, 0 failed**, 239.8 s)

| Suite | Tests | Covers |
|---|---|---|
| `test_near_miss_phase3.py` | 7 | Split integrity, feature lists, `sample_index` exclusion, pipeline contract |
| `test_near_miss_phase4.py` | 18 | Calibration metrics, artifact round-trip, probability contract |
| `test_accident_phase5.py` | 19 | Record loading, box geometry, loss finiteness, tensor shapes, inference contract |
| `test_accident_phase5b.py` | 16 | Label-mapping round-trip, no background in training targets, 3-class head, full-dataset box validity, save/load, checkpoint selection, threshold analysis |
| `test_accident_phase6.py` | 23 | Signal aggregation, calibration fit, threshold study, frozen-test pipeline, dataset-hash guard |
| `test_risk_engine.py` | 36 | Fusion math, threshold boundaries, missing/invalid inputs, config overrides, JSON contract, integration with both real artifacts |
| **Total** | **119** | |

### 9.2 Important Validation Checks

- **Leakage checks:** 0 duplicate groups crossing near-miss splits; 0 identical feature rows crossing splits; 0 `sample_index` overlap; `sample_index` verified absent from every model's feature list; no target-derived features exist; FE_safety_index resolved to a benign input transform (no suspicious validation jump when included).
- **Calibration checks:** near-miss sigmoid test Brier 0.0898 / ECE 0.0439 (vs raw 0.0998 / 0.0864); isotonic rejected on overfit evidence (test log-loss 0.6186). Accident Platt validation fit Brier 0.0063 / ECE 0.0; 5-fold CV used to break the Platt/isotonic tie.
- **Dataset integrity:** SHA-256 over all 2,763 accident PNGs (0 cross-split duplicates, 26 benign train-internal); split-integrity report PASS; original Parquet files hash-verified unchanged before/after the Phase 6 run (`calibration_metadata.json`); DTC-FM features byte-identical in the Parquet artifact.
- **Process guards (tested):** TEST used for calibration: false; TEST used for threshold selection: false; accident TEST evaluations in Phase 6: exactly 1; near-miss model selection used validation metrics only.
- **Semantic guards:** raw detector confidence is never accepted as a probability (tested refusal); risk engine raises on all-missing inputs and on invalid ranges/types (tested); `is_probability: false` on risk output.

---

## 10. File / Artifact Inventory

### models/

| File | Phase | Purpose |
|---|---|---|
| `models/near_miss/random_forest_A_calibrated.joblib` (+ `..._meta.json`) | 4 | **Deployed near-miss model** (RF-A + sigmoid calibrator) |
| `models/near_miss/{dummy_baseline,logistic_regression_A/B/D,random_forest_A/B/D,gradient_boosting_A/B/D}.joblib` | 3 | Experiment pipelines (10 total) |
| `models/accident/calibrated_accident_detector.pth` | 6 | **Deployed accident detector** (Phase 5B EXP_A weights + Platt calibrator) |
| `models/accident/phase5b_expA_best.pth` | 5B | Selected raw checkpoint |
| `models/accident/phase5b_expB_best.pth`, `phase5b_expB_best_last.pth`, `phase5b_expA_best_last.pth`, `phase5b_interrupted_checkpoint.pth` | 5B | Experiment/interrupted checkpoints (retained for future accuracy work) |
| `models/accident/{best,final}_detector.pth`, `model_metadata.json` | 5 | Immutable broken baseline (kept for the record) |
| `models/accident/calibration_metadata.json` | 6 | Calibrator params, fit metrics, dataset SHA-256 hashes, leakage guard |
| `models/accident/phase5b_checkpoint_metadata.json` | 5B | EXP_B interrupted-state metadata |

### src/

| File | Phase | Purpose |
|---|---|---|
| `src/data_preparation.py` | 2 | Deterministic Phase 2 pipeline (re-runnable) |
| `src/near_miss/{split,train,evaluate,calibrate,probability_pipeline,reporting,plotting}.py` | 3–4 | Near-miss split/train/eval/calibrate + inference pipeline |
| `src/accident/{labels,model,dataset,transforms,train,inference,evaluate,metrics,reporting,visualization,debug_visuals,utils}.py` | 5–5B | Detector library (labels.py = label-space single source of truth) |
| `src/accident/{run_phase5,run_phase5b,phase5b,phase5b_analysis,phase5b_report}.py` | 5, 5B | Baseline + improvement experiment drivers |
| `src/accident/{calibration,calibrated_detector,phase6,phase6_report}.py` | 6 | Signal aggregation, Platt calibration, Phase 6 pipeline + report |
| `src/config.py` | 7 | All fusion weights/thresholds/paths |
| `src/risk_engine.py` | 7 | Pure fusion (`assess_risk`, `compute_risk_level`, `recency_factor`) |
| `src/safety_intelligence.py` | 7 | Orchestrator (lazy model loading, `assess_event`/`assess_image`) |

### reports/

| File | Phase |
|---|---|
| `dataset_inventory.md`, `near_miss_dataset_audit.md`, `accident_dataset_audit.md`, `data_quality_report.md`, `dataset_audit.md`, `phase1_summary.json` | 1 |
| `phase2_data_preparation.md`, `data_dictionary.md`, `class_balance_phase2.json` | 2 |
| `near_miss/{split_report,model_evaluation,leakage_review}.md`, `near_miss/{model_comparison,model_comparison_phase4,feature_importance,threshold_analysis,threshold_analysis_calibrated,calibration_comparison}.csv`, `near_miss/model_results.json`, `near_miss/phase4_summary.json` | 3–4 |
| `near_miss/{calibration_report,error_analysis,robustness_analysis,probability_contract}.md`, `near_miss/false_{negatives,positives}_test.csv` | 4 |
| `accident/{environment_report,dataset_summary,validation_results,test_results,error_analysis,accident_detection_contract}.md`, `accident/phase5_summary.json`, `accident/training_{config,history}.json` | 5 |
| `accident/phase5b_experiment_log.md`, `phase5b_experiments.csv`, `phase5b_exp_{a,b}_history.json`, `phase5b_exp{A,B}_run.log` | 5B |
| `accident/{calibration_results,test_results_phase6,phase6_summary,probability_contract}.md`, `accident/{phase6_summary.json,test_results.json,threshold_analysis.csv}`, `accident/phase6_run.log` | 6 |
| `safety/phase7_summary.md` | 7 |
| `member4/Member4_Final_Report.md` (+ PDF) | 8 |

### data/processed/

| File | Purpose |
|---|---|
| `near_miss/dtc_fm_combined.csv` / `.parquet` | Combined features+labels, 2,508 rows (Parquet = exact floats) |
| `near_miss/splits/{train,validation,test}.parquet`, `split_assignments.csv` | Reproducible 70/15/15 split |
| `near_miss/{duplicate_report.csv,label_mapping.json,feature_policy.json}` | Duplicate groups, neutral label map, per-feature decisions |
| `accident/{train,validation,test}_index.parquet` | 2,122/316/325 index rows → `(source_file, row_index)` + SHA-256 + annotations |
| `accident/{duplicate_report.csv,split_integrity_report.json,label_mapping.json,corruption_report.json}` | Integrity artifacts (all PASS) |

### tests/ and figures/

- `tests/` — 6 files, 119 tests (§9).
- `audit_scripts/` — `csv_audit.py`, `parquet_audit.py`, `parquet_labels_audit.py`, `bbox_format_probe.py`.
- `reports/figures/near_miss/` — 9 PNGs (ROC/PR curves, confusion matrix, threshold, calibration comparison, FN/FP analyses, feature importance).
- `reports/figures/accident/` — training curves, calibration curve, confidence-threshold curve, reliability diagram, GT example grids, 24 debug panels, 8 test-example panels.

---

## 11. Limitations

Stated as recorded in the project's own reports — no softening:

1. **Unconfirmed DTC-FM label semantics.** No label map exists in the files or the local documentation. class_1 is only *statistically consistent* with a conflict/near-miss. Every metric and output uses neutral class_0/class_1; interpreting class_1 as "dangerous" requires authoritative documentation. This blocks final safety-oriented framing of all near-miss results.
2. **PET is post-event/offline.** PET and `FE_inv_PET` are computable only after an interaction completes; the offline near-miss model is **not real-time-deployable as-is**. Removing them (config D) costs ~0.02 ROC-AUC and ~0.06–0.08 PR-AUC but stays usable (ROC ≥ 0.90).
3. **Accident dataset characteristics.** ~99% accident-positive frames (image level) make event-level classification/labeling weakly identifiable; object-level test class mix drifts +8.8 points vs train; 26 train-internal duplicate images; small validation/test sets (316/325 images) → high metric variance; letterboxed 640×640 frames; train-only flip/brightness augmentation by the provider.
4. **Accident model accuracy.** Test accident AP@50 0.6706 / mAP@50 0.4381 / accident recall 0.4409 — usable but modest; 5 of 324 accident-positive test images were missed entirely at event level; the single negative image was a confident event-level FP; accuracy improvement was explicitly deferred, not attempted.
5. **Calibration limitations.** Near-miss calibration fitted on n=376 (small); isotonic rejected as overfit. Accident event-level calibration is prevalence-bound (ECE 0.0176 on test with only two populated reliability bins) and must be re-fit on deployment data with different prevalence.
6. **Risk score is a heuristic.** No ground truth for "risk"; no calibration or validation of the score itself; weights/thresholds are engineering choices, not fitted values; linear fusion chosen for transparency; missing-input renormalization can understate risk.
7. **Severity and recency have no data source.** Neither dataset carries severity labels or timestamps; both are caller-supplied under a documented contract.
8. **No real-world deployment validation.** Everything is validated on this dataset's own splits. No live CCTV stream, no cross-camera transfer test, no operational latency measurements. Camera-dependent pixel features (`target_dist_px` family) may not transfer across camera geometries.
9. **Compute constraints.** CPU-only training (torch CPU, 8 threads) motivated 320×320 inputs, 2-epoch baseline, and the EXP_B interruption; 640×640 training would be ~4× slower per epoch on this machine.
10. **DTC-FM engineering caveats.** Positional alignment only (no IDs); 5 benign duplicate groups; `FE_safety_index` numerically unstable (1e-6 floor); acceleration values quantized (~0.56 m/s² steps); HCV class appears in 2 rows.

---

## 12. Member 5 Handoff

### 12.1 What Member 5 receives

| Input | Form | Notes |
|---|---|---|
| Near-miss/conflict probability | `SafetyIntelligence.assess_event(...)` field `near_miss_probability` — calibrated P(class_1) from `random_forest_A_calibrated.joblib` | Label semantics unconfirmed; carry the caveat |
| Accident probability | field `accident_probability` — calibrated P(image contains accident) from `calibrated_accident_detector.pth` | Per-image/event, not per-box; prevalence-bound calibration |
| Per-object detections | `detector_output.detections` — bbox `[x1,y1,x2,y2]`, category, uncalibrated confidence | For localization/display, not decisions |
| Raw detector signal | `detector_raw_signal.accident_detection_confidence` | Ranking score, explicitly NOT a probability |
| Fused risk assessment | `risk_score` (0–100), `risk_level` (LOW/MEDIUM/HIGH/CRITICAL) | Heuristic, `is_probability: false`, not validated |
| Evidence completeness | `present_inputs`, `missing_inputs`, `weights_applied`, `contributions`, `recency_factor` | Member 5 can gate on evidence completeness |
| Transparency | `explanation` list (formula, inputs, contributions, caveats) | Suitable for display/audit |
| Configurability | `src/config.py` (+ `risk_config` override) | Weights, thresholds, half-life, artifact paths — no code changes needed |
| Contracts | `near_miss/probability_contract.md`, `accident/probability_contract.md`, `accident/accident_detection_contract.md` | Schema + guarantees + non-guarantees + update policy |
| Provenance | `model_provenance` block + `calibration_metadata.json` | Artifact paths, fit splits, dataset hashes |

### 12.2 What remains Member 5's responsibility

- **All signal-control / decision-making.** Choosing any deployment threshold (Member 4 deliberately fixed none; the validation threshold tables are provided for that choice), mapping risk levels to actions, actuation, or intervention logic.
- **Supplying `severity` and `event_age_hours`** (or deciding not to) under the documented [0, 1] / hours contracts — Member 4 does not measure either.
- **Deciding how to treat `missing_inputs`** (gate, alert, or abstain) — Member 4 reports them but does not decide.
- **Event aggregation, streaming, latency, UI, and operational integration** — Member 4's layer is per-event/per-image batch inference.
- **Any production recalibration** if the deployment distribution differs (per the update policies in the probability contracts: refit on a fresh validation-like split, never on TEST; bump `detector_version`).

### 12.3 Explicit boundary

Per the Phase 7 rules and the risk engine's own output text: the risk score "must not be used to control traffic signals." Member 4 produces **assessments only**; no signal-control, actuation, or intervention path exists in Member 4's code, and none should be attributed to it.

---

## 13. Future Work (intentionally deferred)

**Accuracy improvement (explicitly deferred by Phase 6 rules):**
- Complete the interrupted EXP_B 25-epoch run (checkpoint retained) and add EXP_D-class changes: longer schedules, 640×640 resolution, augmentation, LR/scheduler tuning, backbone comparison from scratch.
- Address the two event-level failure modes: missed accident-positive images (5/324) and the confident event-level FP on the single negative image.
- Larger/more diverse accident data; per-class metrics are limited by the 316/325-image eval splits.

**Near-miss system:**
- Replace post-event PET with pre-event surrogates (projected time-to-collision, distance rate) for a real-time-deployable variant (config D is the documented baseline).
- Confirm DTC-FM label semantics from the dataset provider and re-frame all class_0/class_1 metrics in operational terms.
- Retire or stabilize `FE_safety_index`-style composites; camera calibration or metric-only features for cross-camera transfer.

**Calibration and validation:**
- Re-fit the accident calibrator on deployment-prevalence data; collect a real-world validation set for both models.
- Validate or empirically derive the risk-fusion weights (currently engineering choices); consider grounding severity in a defined scale.

**Engineering:**
- GPU training (the entire 5B/6 pipeline was CPU-bound); batch/streaming inference service; persistence of assessments; monitoring for drift.

---

## 14. Final Summary

**What has Member 4 successfully built by the end of Phase 7?**

A complete, tested, and documented safety-intelligence layer that turns two raw datasets into one fused, explainable risk assessment:

1. **Audited and prepared both datasets** without modifying a single original file — 2,508 DTC-FM interaction records and 2,763 accident CCTV frames (5,400 boxes), with quality grades (GOOD), zero cross-split contamination (SHA-256-verified), resolved engineered-feature formulas, documented label-mapping status, and reproducible, leakage-safe splits.
2. **Built and calibrated a near-miss/conflict probability model** — 10 trained pipelines, a validation-only selection of Random Forest (config A), and sigmoid calibration: test ROC-AUC 0.929, PR-AUC 0.830, calibrated test Brier 0.0898 / ECE 0.0439 — with the honest caveat that PET makes the primary configuration offline-only and the label semantics remain unconfirmed.
3. **Built and calibrated an accident detection system** — diagnosing and fixing a confirmed background/label-collision bug that had driven the Phase 5 baseline's accident AP to 0.0, then training Faster R-CNN to validation accident AP@50 0.7624, selecting the checkpoint on validation only, calibrating the `noisy_or_top5` event signal with Platt, and reporting a single frozen test evaluation: accident AP@50 0.6706, precision 0.6949 / recall 0.6292 @0.5, event-level Brier 0.0169.
4. **Fused both probabilities with severity and recency** into a transparent, configurable, and fully tested risk score (weights 0.35/0.40/0.15/0.10; thresholds 25/50/75; 24-hour recency half-life) that renormalizes over present evidence, refuses to guess on invalid or missing inputs, never treats raw detector confidence as a probability, and explains every assessment it emits.
5. **Delivered the Member 5 interface**: `SafetyIntelligence.assess_event` / `assess_image` returning the JSON payload shown in §7 — verified here on real data (risk 60.60, HIGH) — plus written probability contracts, update policies, and an explicit boundary: Member 4 assesses risk; it does not control signals, measure severity, or decide actions.

Everything is backed by 119 passing tests, per-phase reports, and honest limitation registers. The system is research-grade, not deployment-grade: the accuracy, calibration-prevalence, label-semantics, and real-world-validation gaps are documented in §11 and queued in §13 for the phases that follow.

---

*Report generated 2026-09-03 from the project's own artifacts. All metrics were cross-checked against `reports/`, `models/*/…metadata.json`, and a fresh full test run (119/119 passed). No dataset, model, or existing report was modified to produce this document.*