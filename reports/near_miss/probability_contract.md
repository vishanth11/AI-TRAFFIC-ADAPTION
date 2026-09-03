# Probability Inference Contract — Member 4

## Purpose

Defines the lightweight inference interface for the calibrated near-miss/
conflict probability model. Later phases (risk engine) consume these
probabilities; this contract does not emit risk levels or traffic commands.

## Model

- Calibrated artifact: `models\near_miss\random_forest_A_calibrated.joblib`
- Original model: `random_forest_A`
- Calibration method: `sigmoid`
- Feature configuration: `A_offline_full_no_fesafety`
- Random seed: `42`

## Input

A single feature vector matching the model's feature schema:

```json
{
  "speed_object_1_kph": "float | int | str (for categorical)",
  "speed_object_2_kph": "float | int | str (for categorical)",
  "angle_degrees": "float | int | str (for categorical)",
  "PET": "float | int | str (for categorical)",
  "acceleration_obj1_mps2": "float | int | str (for categorical)",
  "acceleration_obj2_mps2": "float | int | str (for categorical)",
  "min_dist_dual_check_m": "float | int | str (for categorical)",
  "object_count_at_frame1": "float | int | str (for categorical)",
  "target_dist_px": "float | int | str (for categorical)",
  "FE_inv_PET": "float | int | str (for categorical)",
  "FE_log_mesafe": "float | int | str (for categorical)",
  "FE_dist_squared": "float | int | str (for categorical)",
  "class_object_1": "float | int | str (for categorical)",
  "class_object_2": "float | int | str (for categorical)"
}
```

Required columns: speed_object_1_kph, speed_object_2_kph, angle_degrees, PET, acceleration_obj1_mps2, acceleration_obj2_mps2, min_dist_dual_check_m, object_count_at_frame1, target_dist_px, FE_inv_PET, FE_log_mesafe, FE_dist_squared, class_object_1, class_object_2.

## Output

```json
{
  "class_0_probability": "float in [0, 1]",
  "class_1_probability": "float in [0, 1]",
  "model_version": "random_forest_A_calibrated",
  "calibrated": true
}
```

- `class_0_probability + class_1_probability` ~ 1.0 (within numerical tolerance).
- Probabilities are calibrated on validation data; they approximate the
  observed frequency of class_1 outcomes.
- Label semantics remain unconfirmed; interpret `class_1` only after
  authoritative documentation is provided.

## Not Included

- risk_level (LOW/MEDIUM/HIGH/CRITICAL)
- confidence score
- accident probability
- traffic signal command
- Member 5 decision

## Loading Example

```python
import joblib
pipe = joblib.load('models/near_miss/random_forest_A_calibrated.joblib')
probs = pipe.predict_proba(X)  # shape (n_samples, 2)
```
