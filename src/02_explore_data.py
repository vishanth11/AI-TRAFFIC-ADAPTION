import sys
import os

sys.path.append(os.path.dirname(__file__))

from importlib import import_module

loader = import_module("01_load_data")

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

X_train, X_test, Y_train, Y_test, adjacency = loader.load_dataset()


# --------------------------------------------------
# 1. Basic statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])
print("Sensors         :", X_train.shape[1])
print("Features        :", X_train.shape[2])

print("\nTarget statistics:")
print("Minimum:", Y_train.min())
print("Maximum:", Y_train.max())
print("Mean   :", Y_train.mean())
print("Std    :", Y_train.std())


# --------------------------------------------------
# 2. Traffic of Sensor 0
# --------------------------------------------------

sensor = 0

plt.figure(figsize=(12, 5))

plt.plot(Y_train[sensor])

plt.title("Traffic Volume - Sensor 0")
plt.xlabel("15-minute Time Steps")
plt.ylabel("Normalized Traffic Volume")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "sensor_0_traffic.png"),
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 3. Average traffic across all sensors
# --------------------------------------------------

average_traffic = Y_train.mean(axis=0)

plt.figure(figsize=(12, 5))

plt.plot(average_traffic)

plt.title("Average Traffic Volume Across 36 Sensors")
plt.xlabel("15-minute Time Steps")
plt.ylabel("Normalized Traffic Volume")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "average_traffic.png"),
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 4. Spatial adjacency matrix
# --------------------------------------------------

plt.figure(figsize=(8, 7))

plt.imshow(adjacency)

plt.title("Traffic Sensor Spatial Connectivity")
plt.xlabel("Sensor")
plt.ylabel("Sensor")

plt.colorbar(label="Connectivity")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "adjacency_matrix.png"),
    dpi=300
)

plt.show()
plt.close()


print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print("Results saved to:")
print(RESULTS_DIR)