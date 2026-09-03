# Robustness Analysis — Phase 4

Validation vs test performance deltas for every Phase 3 model. Negative
delta = test performance below validation performance (generalization gap).

| Model | Config | ROC-AUC Δ | PR-AUC Δ | F1 Δ | Recall Δ | Precision Δ |
|---|---|---|---|---|---|---|
| dummy_baseline | A_offline_full_no_fesafety | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gradient_boosting_A | A_offline_full_no_fesafety | -0.0072 | 0.0127 | 0.0154 | 0.0208 | 0.0115 |
| gradient_boosting_B | B_offline_full_with_fesafety | -0.0053 | 0.0156 | 0.0113 | 0.0312 | -0.0022 |
| gradient_boosting_D | D_deployment_realistic | -0.0055 | -0.0168 | 0.0197 | 0.0625 | -0.0112 |
| logistic_regression_A | A_offline_full_no_fesafety | -0.0040 | -0.0278 | -0.0343 | -0.0521 | -0.0238 |
| logistic_regression_B | B_offline_full_with_fesafety | 0.0010 | -0.0255 | -0.0230 | -0.0312 | -0.0179 |
| logistic_regression_D | D_deployment_realistic | 0.0002 | 0.0004 | -0.0135 | -0.0417 | -0.0006 |
| random_forest_A | A_offline_full_no_fesafety | -0.0015 | -0.0236 | -0.0268 | 0.0000 | -0.0494 |
| random_forest_B | B_offline_full_with_fesafety | -0.0020 | -0.0170 | -0.0254 | 0.0208 | -0.0619 |
| random_forest_D | D_deployment_realistic | -0.0025 | -0.0127 | -0.0046 | 0.0104 | -0.0167 |

## Key Observations

- Random Forest config A shows a large train/test gap in raw probability
  calibration (train PR-AUC ~ 0.9996, test PR-AUC ~ 0.830). This is why
  calibration is required before using probabilities as safety scores.
- Validation and test rankings are stable: random_forest_A remains the best
  offline configuration by validation ROC-AUC / PR-AUC.
- Deployment-realistic config D (no PET/FE_inv_PET/FE_safety_index) loses
  ~0.02 ROC-AUC and ~0.06–0.08 PR-AUC versus config A, but still exceeds
  0.90 ROC-AUC. PET is a post-event measurement, so the offline config A
  model is not real-time-deployable without replacing PET-derived signals.

## Configuration Robustness

- **FE_safety_index (B vs A):** adding it does not materially improve
  validation metrics and introduces a numerically unstable composite.
  Keep it excluded on parsimony/stability grounds.
- **PET (A vs D):** removing PET and FE_inv_PET reduces offline
  discriminative power. For live deployment, plan to replace PET with
  pre-event surrogates (e.g., projected time-to-collision, distance rate).
