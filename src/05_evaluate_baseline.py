import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

Y_test = np.load(
    os.path.join(
        PROCESSED_DIR,
        "Y_test.npy"
    )
)

Y_pred = np.load(
    os.path.join(
        RESULTS_DIR,
        "baseline_predictions.npy"
    )
)

sensor_mae = np.load(
    os.path.join(
        RESULTS_DIR,
        "sensor_mae.npy"
    )
)

print("=" * 60)
print("BASELINE MODEL EVALUATION")
print("=" * 60)

print("Actual shape    :", Y_test.shape)
print("Predicted shape :", Y_pred.shape)


# ============================================================
# 1. ACTUAL VS PREDICTED — SENSOR 0
# ============================================================

sensor = 0

plt.figure(figsize=(12, 5))

plt.plot(
    Y_test[:, sensor],
    label="Actual"
)

plt.plot(
    Y_pred[:, sensor],
    label="Predicted"
)

plt.title(
    f"Actual vs Predicted Traffic - Sensor {sensor}"
)

plt.xlabel("Time Step")
plt.ylabel("Normalized Traffic")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "actual_vs_predicted_sensor_0.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. ACTUAL VS PREDICTED — SENSOR 1
# ============================================================

sensor = 1

plt.figure(figsize=(12, 5))

plt.plot(
    Y_test[:, sensor],
    label="Actual"
)

plt.plot(
    Y_pred[:, sensor],
    label="Predicted"
)

plt.title(
    f"Actual vs Predicted Traffic - Sensor {sensor}"
)

plt.xlabel("Time Step")
plt.ylabel("Normalized Traffic")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "actual_vs_predicted_sensor_1.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. SENSOR-WISE MAE
# ============================================================

plt.figure(figsize=(12, 5))

plt.bar(
    range(36),
    sensor_mae
)

plt.title(
    "Sensor-wise Prediction Error"
)

plt.xlabel("Sensor ID")
plt.ylabel("MAE")

plt.xticks(
    range(36)
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sensor_wise_mae.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. PREDICTION SCATTER
# ============================================================

actual = Y_test.flatten()
predicted = Y_pred.flatten()

plt.figure(figsize=(7, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.3
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Traffic"
)

plt.xlabel(
    "Actual Traffic"
)

plt.ylabel(
    "Predicted Traffic"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "actual_vs_predicted_scatter.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION VISUALIZATION COMPLETED")
print("=" * 60)

print("\nSaved files:")

print("1. actual_vs_predicted_sensor_0.png")
print("2. actual_vs_predicted_sensor_1.png")
print("3. sensor_wise_mae.png")
print("4. actual_vs_predicted_scatter.png")

print("\nLocation:")
print(RESULTS_DIR)