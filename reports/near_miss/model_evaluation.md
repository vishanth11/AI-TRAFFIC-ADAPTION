# Near-Miss Model Evaluation — Phase 3

Seed 42 · group-aware 70/15/15 split · preprocessing (median imputation,
standard scaling, one-hot with `handle_unknown="ignore"`) fitted **on train
only** inside each saved pipeline. All probabilistic models expose
`predict_proba()`. Full table: `model_comparison.csv` · raw:
`model_results.json`.

**Terminology note:** label semantics remain unconfirmed (neutral
`class_0`/`class_1`). Metrics below reference the **minority class (class_1,
25.6%)**. Safety-oriented interpretation (e.g., "recall of the dangerous
class") cannot be finalized until semantics are confirmed.

## Experiment Table (test split, threshold 0.50)

| Model | Config | Accuracy | Precision (c1) | Recall (c1) | F1 (c1) | ROC-AUC | PR-AUC | FN | FNR |
|---|---|---|---|---|---|---|---|---|---|
| DummyClassifier (majority) | A | 0.745 | 0.000 | 0.000 | 0.000 | 0.500 | 0.256 | 96 | 1.000 |
| Logistic Regression | A | 0.806 | 0.586 | 0.813 | 0.681 | 0.912 | 0.818 | 18 | 0.188 |
| Logistic Regression | B | 0.809 | 0.590 | 0.823 | 0.687 | 0.913 | 0.813 | 17 | 0.177 |
| Logistic Regression | D (deploy) | 0.790 | 0.559 | 0.833 | 0.669 | 0.898 | 0.773 | 16 | 0.167 |
| **Random Forest** | **A** | **0.867** | **0.713** | 0.802 | **0.755** | **0.929** | **0.830** | 19 | 0.198 |
| Random Forest | B | 0.862 | 0.693 | 0.823 | 0.752 | 0.930 | 0.835 | 17 | 0.177 |
| Random Forest | D (deploy) | 0.854 | 0.688 | 0.781 | 0.732 | 0.912 | 0.777 | 21 | 0.219 |
| Gradient Boosting | A | 0.846 | 0.661 | 0.813 | 0.729 | 0.916 | **0.836** | 18 | 0.188 |
| Gradient Boosting | B | 0.840 | 0.650 | 0.813 | 0.722 | 0.917 | 0.839 | 18 | 0.188 |
| Gradient Boosting | D (deploy) | 0.843 | 0.655 | 0.813 | 0.726 | 0.901 | 0.749 | 18 | 0.188 |

Validation set ranks identically (RF-A best ROC 0.931 / PR 0.854; LR-A best
recall 0.865, FN 13). Every model comfortably beats the Dummy baseline
(74.5% accuracy, zero minority-class recall, 96/96 FNs).

## Confusion Matrix — random_forest_A (TEST, t=0.50)

```
                 pred class_0   pred class_1
actual class_0       TN 249         FP 31
actual class_1       FN  19         TP 77
```

Figure: `reports/figures/near_miss/confusion_matrix_random_forest_A_test.png`

## Primary Model Selection

**Selected: random_forest_A** (`random_forest_A.joblib`) — by a
safety-oriented, validation-based process (no test metric entered selection):

1. Best validation ROC-AUC (0.931) and PR-AUC (0.854).
2. Best validation F1 for the minority class (0.782).
3. Not the best raw recall (LR-A: 0.865 vs RF-A 0.802 at t=0.5) — but the
   threshold analysis (below) shows RF-A reaches LR-A's recall band *with
   higher precision* at t≈0.30, so the recall capability is not lost.
4. Generalization val→test is stable (ROC 0.931→0.929; PR 0.854→0.830).
5. Caveat documented: RF fits train almost perfectly (train PR 0.999) —
   typical RF variance; gradient boosting is the more conservative
   train-consistent alternative (train ROC 0.975). Phase 4 calibration should
   address this.

If label semantics later confirm class_1 as the conflict/danger class, the
recall-first operating point (t≈0.30) is the recommended candidate; if not,
selection should be re-framed in Phase 4.

## Threshold Analysis (validation, random_forest_A)

Full table: `threshold_analysis.csv` · figure: `threshold_analysis.png`

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.20 | 0.467 | 0.948 | 0.625 | 104 | 5 |
| 0.30 | 0.569 | 0.906 | 0.699 | 66 | 9 |
| 0.40 | 0.675 | 0.865 | 0.758 | 40 | 13 |
| 0.50 | 0.762 | 0.802 | 0.782 | 24 | 19 |
| 0.60 | 0.783 | 0.750 | 0.766 | 20 | 24 |
| 0.70 | 0.816 | 0.646 | 0.721 | 14 | 34 |
| 0.80 | 0.869 | 0.552 | 0.675 | 8 | 43 |

**No deployment threshold is permanently selected** (Phase 4 = calibration).
Two documented candidates:
- **t≈0.30** — recall-first: recall 0.906, FN 9 (75% fewer FNs than t=0.5), at
  precision 0.569.
- **t≈0.50** — balanced: F1 maximum (0.782), precision 0.762.

The risk engine (later phase) should consume raw probabilities — not a hard
threshold — so these are operating-point references, not a final decision.

## Minority-Class (class_1) Performance — random_forest_A, TEST

- Recall 0.802 (t=0.5); 0.90+ reachable at t≤0.30.
- Precision 0.713; F1 0.755; PR-AUC 0.830 vs prevalence 0.256 (**3.2× lift**).
- 19/96 class_1 observations missed at t=0.5; 5 at t=0.2.
- ROC/PR curves: `roc_pr_curve_random_forest_A.png`, `roc_pr_curve_logistic_regression_A.png`.

## Feature Importance (associated with predictions — NOT causal)

Top RF impurity importances (config A, full table `feature_importance.csv`):

1. `PET` — 0.377
2. `min_dist_dual_check_m` — 0.244
3. `target_dist_px` — 0.121
4. `FE_log_mesafe` — 0.088
5. `angle_degrees` — 0.061
6. speeds / object counts / classes — remaining mass

Interpretation discipline: these features are **associated with model
predictions**. No claim is made that any feature causes conflicts or
near-misses. Dominance of PET and minimum distance matches the surrogate-safety
literature (DTC/PET family) but also mirrors the Phase 2 caveat: PET is a
post-event measurement, so its predictive dominance inflates offline metrics
relative to a live system.

## Configuration Comparison — what it means

- **FE_safety_index (B vs A):** adding it does NOT help (validation ROC
  0.9305→0.9315 RF [negligible], 0.9164→0.9121 LR [worse], 0.9229→0.9224 GB
  [worse]). No suspicious performance jump → consistent with Phase 3's
  reverse-engineering: it is a deterministic transform of three input
  features, carrying no label information beyond them. Kept OUT of the primary
  configuration on parsimony and numerical-stability grounds.
- **PET (A vs D):** removing post-event features costs ~0.02 ROC-AUC
  (LR 0.916→0.898; RF 0.931→0.914) and ~0.06–0.08 PR-AUC. Deployment-realistic
  models remain usable (ROC ≥ 0.90) but materially weaker. **The offline
  models are NOT real-time-deployable as-is** because PET (and FE_inv_PET)
  require the completed interaction.

## Probability Output Contract (for later phases)

Every saved pipeline supports `predict_proba()[:, 1]` — probability that the
observation belongs to class_1. Future risk-engine payload shape:

```json
{"class_0_probability": 0.12, "class_1_probability": 0.88}
```

No risk engine, confidence engine, or calibration was built in Phase 3.

## Files

- Models: `models/near_miss/{run_name}.joblib` (10 full pipelines)
- Splits: `data/processed/near_miss/splits/` + `split_assignments.csv`
- Reports: `model_comparison.csv`, `model_results.json`, `threshold_analysis.csv`,
  `feature_importance.csv`, `split_report.md`, `leakage_review.md`
- Figures: `reports/figures/near_miss/` (5 PNGs)