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


# ============================================================
# LOAD ADJACENCY MATRIX
# ============================================================

adjacency = np.load(
    os.path.join(
        PROCESSED_DIR,
        "adjacency.npy"
    )
)


print("=" * 60)
print("GRAPH / SENSOR NETWORK ANALYSIS")
print("=" * 60)

print("Adjacency shape:", adjacency.shape)

print("\nMinimum:", adjacency.min())
print("Maximum:", adjacency.max())
print("Mean   :", adjacency.mean())

print(
    "\nNon-zero connections:",
    np.count_nonzero(adjacency)
)


# ============================================================
# SENSOR DEGREE
# ============================================================

degree = np.count_nonzero(
    adjacency,
    axis=1
)

print("\nSensor connections:")

for sensor, d in enumerate(degree):
    print(
        f"Sensor {sensor:02d}: {d} connections"
    )


# ============================================================
# MOST CONNECTED SENSOR
# ============================================================

most_connected = np.argmax(degree)

print(
    "\nMost connected sensor:",
    most_connected
)

print(
    "Number of connections:",
    degree[most_connected]
)


# ============================================================
# ADJACENCY MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    adjacency,
    aspect="auto"
)

plt.colorbar(
    label="Connection Weight"
)

plt.title(
    "Traffic Sensor Network - Adjacency Matrix"
)

plt.xlabel("Sensor")
plt.ylabel("Sensor")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "sensor_network.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# CONNECTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    range(36),
    degree
)

plt.title(
    "Number of Connections per Traffic Sensor"
)

plt.xlabel("Sensor ID")
plt.ylabel("Number of Connections")

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
        "sensor_connections.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("GRAPH ANALYSIS COMPLETED")
print("=" * 60)

print("\nSaved:")
print("1. sensor_network.png")
print("2. sensor_connections.png")