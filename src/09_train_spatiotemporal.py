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
    os.path.join(PROCESSED_DIR, "X_train.npy")
).astype(np.float32)

X_test = np.load(
    os.path.join(PROCESSED_DIR, "X_test.npy")
).astype(np.float32)

Y_train = np.load(
    os.path.join(PROCESSED_DIR, "Y_train.npy")
).astype(np.float32)

Y_test = np.load(
    os.path.join(PROCESSED_DIR, "Y_test.npy")
).astype(np.float32)

adjacency = np.load(
    os.path.join(PROCESSED_DIR, "adjacency.npy")
).astype(np.float32)


print("=" * 60)
print("SPATIO-TEMPORAL GNN")
print("=" * 60)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("Y_train :", Y_train.shape)
print("Y_test  :", Y_test.shape)
print("Adj     :", adjacency.shape)


# ============================================================
# 3. EXTRACT HISTORICAL TRAFFIC
# ============================================================

# First 10 features = historical traffic observations

X_train_history = X_train[:, :, :10]

X_test_history = X_test[:, :, :10]


print("\nHistorical traffic:")
print(
    "Train:",
    X_train_history.shape
)

print(
    "Test :",
    X_test_history.shape
)


# ============================================================
# 4. NORMALIZE ADJACENCY
# ============================================================

def normalize_adjacency(A):

    # Add self connections
    A = A + np.eye(A.shape[0])

    degree = np.sum(
        A,
        axis=1
    )

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

    return (
        D_inv_sqrt
        @ A
        @ D_inv_sqrt
    ).astype(np.float32)


A = normalize_adjacency(
    adjacency
)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)


# ============================================================
# 6. TENSOR CONVERSION
# ============================================================

# Convert:
#
# (time, sensors, history)
#
# to:
#
# (time, history, sensors)

X_train_history = np.transpose(
    X_train_history,
    (0, 2, 1)
)

X_test_history = np.transpose(
    X_test_history,
    (0, 2, 1)
)


X_train_tensor = torch.tensor(
    X_train_history,
    dtype=torch.float32
).to(device)

X_test_tensor = torch.tensor(
    X_test_history,
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
    A,
    dtype=torch.float32
).to(device)


# ============================================================
# 7. SPATIO-TEMPORAL MODEL
# ============================================================

class SpatioTemporalGNN(nn.Module):

    def __init__(
        self,
        num_sensors=36,
        hidden_size=64
    ):

        super().__init__()

        self.num_sensors = num_sensors

        # ----------------------------------------------------
        # TEMPORAL MODEL
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # ----------------------------------------------------
        # GRAPH LAYERS
        # ----------------------------------------------------

        self.graph_linear1 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.graph_linear2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            hidden_size,
            1
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        X,
        A
    ):

        # X shape:
        #
        # batch × history × sensors
        #
        # Example:
        # 1261 × 10 × 36

        batch_size = X.shape[0]

        # ----------------------------------------------------
        # TEMPORAL PROCESSING
        # ----------------------------------------------------

        # Process each sensor's history independently.
        #
        # Rearrange:
        #
        # batch × history × sensors
        #
        # →
        #
        # batch × sensors × history

        X = X.transpose(
            1,
            2
        )

        # Now:
        #
        # batch × sensors × history
        #
        # Treat history as sequence.

        X = X.reshape(
            batch_size * self.num_sensors,
            X.shape[2],
            1
        )

        # GRU

        temporal_output, _ = self.gru(X)

        # Last historical representation

        temporal_output = temporal_output[:, -1, :]

        # Restore sensor dimension

        temporal_output = temporal_output.reshape(
            batch_size,
            self.num_sensors,
            -1
        )

        # ----------------------------------------------------
        # GRAPH CONVOLUTION 1
        # ----------------------------------------------------

        spatial_output = torch.matmul(
            A,
            temporal_output
        )

        spatial_output = self.graph_linear1(
            spatial_output
        )

        spatial_output = self.relu(
            spatial_output
        )

        # ----------------------------------------------------
        # GRAPH CONVOLUTION 2
        # ----------------------------------------------------

        spatial_output = torch.matmul(
            A,
            spatial_output
        )

        spatial_output = self.graph_linear2(
            spatial_output
        )

        spatial_output = self.relu(
            spatial_output
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        output = self.output_layer(
            spatial_output
        )

        output = output.squeeze(
            -1
        )

        return output


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = SpatioTemporalGNN().to(device)

print("\nModel:")
print(model)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# ============================================================
# 10. TRAINING
# ============================================================

epochs = 150

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
# 11. TEST PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_tensor,
        A_tensor
    )

Y_pred = predictions.cpu().numpy()


# ============================================================
# 12. METRICS
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
print("SPATIO-TEMPORAL GNN PERFORMANCE")
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
# 13. SENSOR-WISE PERFORMANCE
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
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULTS_DIR,
    "spatiotemporal_gnn.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    RESULTS_DIR,
    "spatiotemporal_predictions.npy"
)

np.save(
    prediction_path,
    Y_pred
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "spatiotemporal_sensor_mae.npy"
    ),
    sensor_mae
)


# ============================================================
# 16. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("SPATIO-TEMPORAL GNN COMPLETED")
print("=" * 60)

print("\nModel saved:")
print(model_path)

print("\nPredictions saved:")
print(prediction_path)