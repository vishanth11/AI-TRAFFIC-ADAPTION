# Error Analysis — Phase 4

Analysis of `random_forest_A` false positives and false negatives on the
test split using the selected calibrated probabilities (method=sigmoid, threshold 0.50).
Association is reported; no causal claims are made.

## False Negatives (class_1 predicted as class_0)

Count: 30

### Patterns associated with false negatives

```json
{
  "speed_object_1_kph": {
    "mean": 8.6,
    "median": 8.5,
    "min": 1.0,
    "max": 22.0
  },
  "speed_object_2_kph": {
    "mean": 9.8,
    "median": 9.0,
    "min": 1.0,
    "max": 19.0
  },
  "angle_degrees": {
    "mean": 41.923,
    "median": 41.185,
    "min": 1.32,
    "max": 88.48
  },
  "PET": {
    "mean": 2.178,
    "median": 2.17,
    "min": 1.57,
    "max": 3.0
  },
  "acceleration_obj1_mps2": {
    "mean": 0.4827,
    "median": 0.28,
    "min": 0.0,
    "max": 1.67
  },
  "acceleration_obj2_mps2": {
    "mean": 0.3713,
    "median": 0.0,
    "min": -1.11,
    "max": 2.22
  },
  "min_dist_dual_check_m": {
    "mean": 3.4943,
    "median": 2.669,
    "min": 0.909,
    "max": 9.105
  },
  "object_count_at_frame1": {
    "mean": 30.3,
    "median": 32.5,
    "min": 5.0,
    "max": 51.0
  },
  "target_dist_px": {
    "mean": 50.3132,
    "median": 38.23,
    "min": 8.431,
    "max": 135.971
  },
  "FE_inv_PET": {
    "mean": 0.4748,
    "median": 0.4608,
    "min": 0.3333,
    "max": 0.6369
  },
  "FE_log_mesafe": {
    "mean": 3.7351,
    "median": 3.6693,
    "min": 2.244,
    "max": 4.9198
  },
  "FE_dist_squared": {
    "mean": 3631.2715,
    "median": 1462.0909,
    "min": 71.0818,
    "max": 18488.1128
  },
  "class_object_1": {
    "Car": 18,
    "VRU": 7,
    "Ped": 5
  },
  "class_object_2": {
    "VRU": 13,
    "Car": 9,
    "Ped": 8
  }
}
```

### Probability distribution

```json
{
  "min": 0.0462,
  "max": 0.4989,
  "median": 0.2897
}
```

### Sample indices

- count: 30
- first 20: [64, 47, 927, 731, 2302, 240, 2072, 1097, 1934, 385, 1556, 1851, 259, 733, 1768, 1716, 531, 1789, 1844, 272]

## False Positives (class_0 predicted as class_1)

Count: 16

### Patterns associated with false positives

```json
{
  "speed_object_1_kph": {
    "mean": 8.125,
    "median": 7.5,
    "min": 2.0,
    "max": 19.0
  },
  "speed_object_2_kph": {
    "mean": 8.6875,
    "median": 9.5,
    "min": 3.0,
    "max": 18.0
  },
  "angle_degrees": {
    "mean": 53.6013,
    "median": 57.625,
    "min": 3.44,
    "max": 89.25
  },
  "PET": {
    "mean": 2.1819,
    "median": 2.265,
    "min": 1.4,
    "max": 2.93
  },
  "acceleration_obj1_mps2": {
    "mean": 0.3838,
    "median": 0.28,
    "min": 0.0,
    "max": 2.22
  },
  "acceleration_obj2_mps2": {
    "mean": 0.2781,
    "median": 0.0,
    "min": -0.56,
    "max": 1.67
  },
  "min_dist_dual_check_m": {
    "mean": 1.4134,
    "median": 1.144,
    "min": 0.59,
    "max": 2.712
  },
  "object_count_at_frame1": {
    "mean": 32.4375,
    "median": 36.5,
    "min": 8.0,
    "max": 46.0
  },
  "target_dist_px": {
    "mean": 15.5627,
    "median": 12.4905,
    "min": 1.506,
    "max": 34.946
  },
  "FE_inv_PET": {
    "mean": 0.4872,
    "median": 0.4416,
    "min": 0.3413,
    "max": 0.7143
  },
  "FE_log_mesafe": {
    "mean": 2.565,
    "median": 2.6011,
    "min": 0.9187,
    "max": 3.582
  },
  "FE_dist_squared": {
    "mean": 349.293,
    "median": 156.3519,
    "min": 2.268,
    "max": 1221.2229
  },
  "class_object_1": {
    "Car": 10,
    "LCV": 3,
    "Ped": 3
  },
  "class_object_2": {
    "VRU": 8,
    "Ped": 5,
    "Car": 3
  }
}
```

### Probability distribution

```json
{
  "min": 0.5123,
  "max": 0.8406,
  "median": 0.6274
}
```

### Sample indices

- count: 16
- first 20: [962, 963, 184, 1959, 2475, 1921, 1237, 1220, 1330, 740, 575, 660, 668, 1326, 653, 2207]

## Interpretation Discipline

- 'Associated with' does not mean 'causes'.
- The model may err on edge cases, low-severity events, or rare
  categorical combinations.
- No labels were manually corrected; the test set remains untouched.

## Detailed Tables

- `reports/near_miss/false_negatives_test.csv`
- `reports/near_miss/false_positives_test.csv`
