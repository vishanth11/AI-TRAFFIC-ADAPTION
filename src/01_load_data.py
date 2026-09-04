import scipy.io as sio
import numpy as np

DATA_PATH = r"D:\traffic_prediction\data\raw\traffic_dataset.mat"


def load_dataset():
    data = sio.loadmat(DATA_PATH)

    X_train_raw = data["tra_X_tr"]
    X_test_raw = data["tra_X_te"]

    Y_train = np.asarray(data["tra_Y_tr"], dtype=np.float32)
    Y_test = np.asarray(data["tra_Y_te"], dtype=np.float32)

    adjacency = np.asarray(
        data["tra_adj_mat"],
        dtype=np.float32
    )

    # Convert sparse matrices inside object arrays
    X_train = np.stack([
        X_train_raw[0, i].toarray()
        for i in range(X_train_raw.shape[1])
    ])

    X_test = np.stack([
        X_test_raw[0, i].toarray()
        for i in range(X_test_raw.shape[1])
    ])

    print("=" * 60)
    print("TRAFFIC DATASET")
    print("=" * 60)

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)
    print("Adjacency:", adjacency.shape)

    print("\nExpected:")
    print("X_train → (1261, 36, 48)")
    print("X_test  → (840, 36, 48)")
    print("Y_train → (36, 1261)")
    print("Y_test  → (36, 840)")

    return X_train, X_test, Y_train, Y_test, adjacency


if __name__ == "__main__":
    load_dataset()