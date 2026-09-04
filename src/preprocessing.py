import numpy as np
from scipy.io import loadmat
from scipy import sparse
import os


RAW_FILE = "data/raw/traffic_dataset.mat"
OUTPUT_FILE = "data/processed/traffic_processed.npz"


def convert_sparse_samples(samples):
    """
    Convert MATLAB object array containing sparse matrices
    into a dense NumPy array.

    Input:
        (number_of_samples,) where each element is 36 x 48

    Output:
        (number_of_samples, 36, 48)
    """

    converted = []

    for sample in samples.ravel():

        if sparse.issparse(sample):
            sample = sample.toarray()
        else:
            sample = np.asarray(sample)

        converted.append(sample)

    return np.stack(converted)


def main():

    print("Loading dataset...")

    data = loadmat(RAW_FILE)

    # Training input
    X_train = convert_sparse_samples(data["tra_X_tr"])

    # Test input
    X_test = convert_sparse_samples(data["tra_X_te"])

    # Targets
    Y_train = np.asarray(data["tra_Y_tr"], dtype=np.float32).T
    Y_test = np.asarray(data["tra_Y_te"], dtype=np.float32).T

    # Spatial adjacency matrix
    adjacency = np.asarray(
        data["tra_adj_mat"],
        dtype=np.float32
    )

    print("\nOriginal converted shapes:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)
    print("Adjacency:", adjacency.shape)

    # Convert input from:
    # (samples, sensors, features)
    #
    # to:
    # (samples, features, sensors)
    #
    # This format is convenient for temporal models.
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed dataset
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        adjacency=adjacency
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()