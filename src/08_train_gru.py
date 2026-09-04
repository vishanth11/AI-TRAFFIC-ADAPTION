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


print("=" * 60)
print("TEMPORAL GRU TRAFFIC MODEL")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# 3. EXTRACT HISTORICAL TRAFFIC
# ============================================================

# First 10 features are historical traffic observations.

X_train_history = X_train[:, :, :10]
X_test_history = X_test[:, :, :10]

print("\nHistorical traffic:")
print(
    "X_train_history:",
    X_train_history.shape
)

print(
    "X_test_history :",
    X_test_history.shape
)


# ============================================================
# 4. REARRANGE DATA
# ============================================================

# Current:
# (time, sensors, history)

# We want:
# (time, history, sensors)

X_train_history = np.transpose(
    X_train_history,
    (0, 2, 1)
)

X_test_history = np.transpose(
    X_test_history,
    (0, 2, 1)
)

print("\nGRU input:")
print(
    "X_train:",
    X_train_history.shape
)

print(
    "X_test :",
    X_test_history.shape
)


# ============================================================
# 5. PYTORCH DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)


# ============================================================
# 6. CONVERT TO TENSORS
# ============================================================

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


# ============================================================
# 7. GRU MODEL
# ============================================================

class TrafficGRU(nn.Module):

    def __init__(
        self,
        input_size=36,
        hidden_size=64,
        output_size=36
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, X):

        # X:
        # batch × history × sensors

        output, hidden = self.gru(X)

        # Use final time step
        last_output = output[:, -1, :]

        prediction = self.fc(
            last_output
        )

        return prediction


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = TrafficGRU().to(device)

print("\nModel:")
print(model)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. TRAIN
# ============================================================

epochs = 100

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predictions = model(
        X_train_tensor
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
# 11. PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    predictions = model(
        X_test_tensor
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
print("GRU PERFORMANCE")
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
# 13. SENSOR-WISE MAE
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
    "traffic_gru.pth"
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
    "gru_predictions.npy"
)

np.save(
    prediction_path,
    Y_pred
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "gru_sensor_mae.npy"
    ),
    sensor_mae
)


# ============================================================
# 16. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("GRU TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved:")
print(model_path)

print("\nPredictions saved:")
print(prediction_path)