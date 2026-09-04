import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
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
# 2. LOAD DATA
# ============================================================

X_train = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X_train.npy"
    )
).astype(np.float32)

X_test = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X_test.npy"
    )
).astype(np.float32)

Y_train = np.load(
    os.path.join(
        PROCESSED_DIR,
        "Y_train.npy"
    )
).astype(np.float32)

Y_test = np.load(
    os.path.join(
        PROCESSED_DIR,
        "Y_test.npy"
    )
).astype(np.float32)

adjacency = np.load(
    os.path.join(
        PROCESSED_DIR,
        "adjacency.npy"
    )
).astype(np.float32)


print("=" * 60)
print("SPATIAL GRAPH NEURAL NETWORK")
print("=" * 60)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("Y_train :", Y_train.shape)
print("Y_test  :", Y_test.shape)
print("Adj     :", adjacency.shape)


# ============================================================
# 3. NORMALIZE ADJACENCY MATRIX
# ============================================================

def normalize_adjacency(A):

    # Add self-loops
    A = A + np.eye(A.shape[0])

    # Degree matrix
    degree = np.sum(A, axis=1)

    # Avoid division by zero
    degree_inv_sqrt = np.power(
        degree,
        -0.5
    )

    degree_inv_sqrt[
        np.isinf(degree_inv_sqrt)
    ] = 0.0

    D_inv_sqrt = np.diag(
        degree_inv_sqrt
    )

    # Symmetric normalization
    A_normalized = (
        D_inv_sqrt
        @ A
        @ D_inv_sqrt
    )

    return A_normalized.astype(
        np.float32
    )


adjacency_normalized = normalize_adjacency(
    adjacency
)

print(
    "\nNormalized adjacency:",
    adjacency_normalized.shape
)


# ============================================================
# 4. CONVERT TO PYTORCH
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
).to(device)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

Y_train_tensor = torch.tensor(
    Y_train,
    dtype=torch.float32
).to(device)

Y_test_tensor = torch.tensor(
    Y_test,
    dtype=torch.float32
).to(device)

A_tensor = torch.tensor(
    adjacency_normalized,
    dtype=torch.float32
).to(device)


# ============================================================
# 5. GRAPH CONVOLUTION LAYER
# ============================================================

class GraphConvolution(nn.Module):

    def __init__(
        self,
        input_features,
        output_features
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_features,
            output_features
        )

    def forward(
        self,
        X,
        A
    ):

        # Spatial aggregation
        X = torch.matmul(
            A,
            X
        )

        # Feature transformation
        X = self.linear(X)

        return X


# ============================================================
# 6. GNN MODEL
# ============================================================

class TrafficGNN(nn.Module):

    def __init__(
        self,
        input_features=48,
        hidden_features=64
    ):

        super().__init__()

        self.gcn1 = GraphConvolution(
            input_features,
            hidden_features
        )

        self.gcn2 = GraphConvolution(
            hidden_features,
            hidden_features
        )

        self.output_layer = nn.Linear(
            hidden_features,
            1
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        X,
        A
    ):

        # First graph convolution
        X = self.gcn1(
            X,
            A
        )

        X = self.relu(X)

        # Second graph convolution
        X = self.gcn2(
            X,
            A
        )

        X = self.relu(X)

        # Predict traffic for each sensor
        X = self.output_layer(X)

        # Remove final dimension
        X = X.squeeze(-1)

        return X


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = TrafficGNN(
    input_features=48,
    hidden_features=64
).to(device)

print("\nModel:")
print(model)


# ============================================================
# 8. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 9. TRAINING
# ============================================================

epochs = 100

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predictions = model(
        X_train_tensor,
        A_tensor
    )

    loss = criterion(
        predictions,
        Y_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 10. EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_tensor,
        A_tensor
    )

Y_pred = predictions.cpu().numpy()


# ============================================================
# 11. METRICS
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
print("GNN PERFORMANCE")
print("=" * 60)

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)

print(
    f"R²   : {r2:.6f}"
)


# ============================================================
# 12. SENSOR-WISE PERFORMANCE
# ============================================================

sensor_mae = np.mean(
    np.abs(
        Y_test - Y_pred
    ),
    axis=0
)

print("\nSensor-wise MAE:")

for sensor, error in enumerate(
    sensor_mae
):

    print(
        f"Sensor {sensor:02d}: "
        f"MAE = {error:.6f}"
    )


# ============================================================
# 13. SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULTS_DIR,
    "traffic_gnn.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    RESULTS_DIR,
    "gnn_predictions.npy"
)

np.save(
    prediction_path,
    Y_pred
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "gnn_sensor_mae.npy"
    ),
    sensor_mae
)


# ============================================================
# 15. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("GNN TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved:")
print(model_path)

print("\nPredictions saved:")
print(prediction_path)