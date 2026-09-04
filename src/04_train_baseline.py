import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    os.path.join(PROCESSED_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(PROCESSED_DIR, "X_test.npy")
)

Y_train = np.load(
    os.path.join(PROCESSED_DIR, "Y_train.npy")
)

Y_test = np.load(
    os.path.join(PROCESSED_DIR, "Y_test.npy")
)


print("=" * 60)
print("BASELINE RANDOM FOREST MODEL")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# PREPARE TARGET
# ============================================================

# Y should already be:
# Train → (time, sensors)
# Test  → (time, sensors)

print("\nTarget shapes:")
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# FLATTEN INPUT
# ============================================================

# Each sample contains:
# 36 sensors × 48 features
#
# Convert:
# (samples, 36, 48)
#        ↓
# (samples, 1728)

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


print("\nFlattened input:")
print("X_train_flat:", X_train_flat.shape)
print("X_test_flat :", X_test_flat.shape)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_flat,
    Y_train
)

print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

Y_pred = model.predict(X_test_flat)

print("Prediction shape:", Y_pred.shape)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    Y_test,
    Y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        Y_test,
        Y_pred
    )
)

r2 = r2_score(
    Y_test,
    Y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ============================================================
# SENSOR-WISE PERFORMANCE
# ============================================================

print("\nSensor-wise MAE:")

sensor_mae = np.mean(
    np.abs(Y_test - Y_pred),
    axis=0
)

for sensor, error in enumerate(sensor_mae):
    print(
        f"Sensor {sensor:02d}: MAE = {error:.6f}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "baseline_predictions.npy"
    ),
    Y_pred
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "sensor_mae.npy"
    ),
    sensor_mae
)

print("\nPredictions saved to:")
print(
    os.path.join(
        RESULTS_DIR,
        "baseline_predictions.npy"
    )
)

print("\nBaseline model completed successfully.")