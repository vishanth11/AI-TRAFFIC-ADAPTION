from scipy.io import loadmat

file_path = "data/raw/traffic_dataset.mat"

data = loadmat(file_path)

print("Variables inside traffic_dataset.mat:")
print("=" * 50)

for key, value in data.items():
    if not key.startswith("__"):
        print("Name:", key)
        print("Type:", type(value))
        print("Shape:", getattr(value, "shape", "No shape"))
        print()