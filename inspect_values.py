from scipy.io import loadmat
import numpy as np
from scipy import sparse

file_path = "data/raw/traffic_dataset.mat"

data = loadmat(file_path)

print("=" * 60)
print("TRAFFIC DATASET STRUCTURE")
print("=" * 60)

for name in ["tra_X_tr", "tra_X_te", "tra_Y_tr", "tra_Y_te", "tra_adj_mat"]:

    arr = data[name]

    print("\n" + "=" * 60)
    print(name)
    print("Outer shape:", arr.shape)
    print("Outer dtype:", arr.dtype)

    # Object arrays such as tra_X_tr
    if arr.dtype == object:

        first = arr.flat[0]

        print("First element type:", type(first))

        if sparse.issparse(first):
            dense = first.toarray()

            print("Sparse matrix shape:", first.shape)
            print("Dense matrix shape:", dense.shape)
            print("Stored non-zero values:", first.nnz)

            print("\nFirst 5 rows × first 10 columns:")
            print(dense[:5, :10])

            print("\nMinimum:", dense.min())
            print("Maximum:", dense.max())
            print("Mean:", dense.mean())

    else:
        print("\nFirst 5 rows × first 10 columns:")
        print(arr[:5, :10])

        print("\nMinimum:", arr.min())
        print("Maximum:", arr.max())
        print("Mean:", arr.mean())