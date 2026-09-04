import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))

from importlib import import_module

loader = import_module("01_load_data")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

X_train, X_test, Y_train, Y_test, adjacency = loader.load_dataset()


# --------------------------------------------------
# Convert target orientation
# --------------------------------------------------
# Original:
# Y = (36 sensors, time)
#
# Convert to:
# Y = (time, sensors)

Y_train = Y_train.T
Y_test = Y_test.T


print("=" * 60)
print("PREPROCESSING")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)

print("Adjacency:", adjacency.shape)


# --------------------------------------------------
# Verify temporal alignment
# --------------------------------------------------

print("\nTemporal alignment:")

print("X_train time steps:", X_train.shape[0])
print("Y_train time steps:", Y_train.shape[0])

print("X_test time steps :", X_test.shape[0])
print("Y_test time steps :", Y_test.shape[0])


# --------------------------------------------------
# Separate feature groups
# --------------------------------------------------

historical_traffic = X_train[:, :, 0:10]

temporal_features = X_train[:, :, 10:39]

categorical_features = X_train[:, :, 39:48]


print("\nFeature groups:")
print("Historical traffic :", historical_traffic.shape)
print("Temporal features  :", temporal_features.shape)
print("Other features     :", categorical_features.shape)


# --------------------------------------------------
# Save processed arrays
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


np.save(
    os.path.join(PROCESSED_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(PROCESSED_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(PROCESSED_DIR, "Y_train.npy"),
    Y_train
)

np.save(
    os.path.join(PROCESSED_DIR, "Y_test.npy"),
    Y_test
)

np.save(
    os.path.join(PROCESSED_DIR, "adjacency.npy"),
    adjacency
)


print("\nProcessed data saved to:")

print(PROCESSED_DIR)

print("\nPreprocessing completed successfully.")