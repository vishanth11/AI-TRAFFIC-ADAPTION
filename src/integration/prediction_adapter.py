import os
import numpy as np
import torch
import torch.nn as nn


class SpatioTemporalGNN(nn.Module):
    """
    Same architecture used by Member 3 during training.

    Input:
        (batch, history, sensors)

    Example:
        (1, 10, 36)

    Output:
        (batch, 36)
    """

    def __init__(self, num_sensors=36, hidden_size=64):
        super().__init__()

        self.num_sensors = num_sensors

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.graph_linear1 = nn.Linear(hidden_size, hidden_size)
        self.graph_linear2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, X, adjacency):
        """
        X shape:
            (batch, history, sensors)

        adjacency shape:
            (sensors, sensors)
        """

        batch_size = X.shape[0]

        # Convert:
        # (batch, history, sensors)
        # ->
        # (batch, sensors, history)
        X = X.transpose(1, 2)

        num_sensors = X.shape[1]
        history = X.shape[2]

        # Each sensor becomes an individual GRU sequence
        X = X.reshape(batch_size * num_sensors, history, 1)

        # Temporal modeling
        gru_out, _ = self.gru(X)

        # Last timestep
        X = gru_out[:, -1, :]

        # Back to:
        # (batch, sensors, hidden)
        X = X.reshape(batch_size, num_sensors, -1)

        # Spatial graph propagation
        X = torch.matmul(adjacency, X)
        X = self.relu(self.graph_linear1(X))

        X = torch.matmul(adjacency, X)
        X = self.relu(self.graph_linear2(X))

        # Prediction
        X = self.output_layer(X)

        return X.squeeze(-1)


class TrafficPredictionAdapter:
    """
    Inference wrapper for Member 3's trained GNN.
    """

    def __init__(
        self,
        model_path=None,
        adjacency_path=None,
        device="cpu"
    ):

        # Project root:
        # AI-TRAFFIC-ADAPTION/
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        if model_path is None:
            model_path = os.path.join(
                project_root,
                "results",
                "spatiotemporal_gnn.pth"
            )

        if adjacency_path is None:
            adjacency_path = os.path.join(
                project_root,
                "data",
                "processed",
                "adjacency.npy"
            )

        self.device = torch.device(device)

        # Load adjacency
        adjacency = np.load(adjacency_path).astype(np.float32)

        if adjacency.shape != (36, 36):
            raise ValueError(
                f"Expected adjacency matrix shape (36, 36), "
                f"got {adjacency.shape}"
            )

        self.num_sensors = adjacency.shape[0]

        # Normalize adjacency exactly as used during training
        adjacency = adjacency + np.eye(
            self.num_sensors,
            dtype=np.float32
        )

        degree = adjacency.sum(axis=1)

        degree_inv_sqrt = np.zeros_like(
            degree,
            dtype=np.float32
        )

        np.power(
            degree,
            -0.5,
            out=degree_inv_sqrt,
            where=degree > 0
        )

        normalized_adjacency = (
            degree_inv_sqrt[:, None]
            * adjacency
            * degree_inv_sqrt[None, :]
        )

        self.adjacency = torch.tensor(
            normalized_adjacency,
            dtype=torch.float32,
            device=self.device
        )

        # Create model
        self.model = SpatioTemporalGNN(
            num_sensors=self.num_sensors,
            hidden_size=64
        )

        # Load trained weights
        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print("Traffic Prediction Adapter initialized")
        print(f"Model: {model_path}")
        print(f"Adjacency: {adjacency_path}")
        print(f"Sensors: {self.num_sensors}")
        print(f"Device: {self.device}")

    def predict(self, traffic_history):
        """
        Predict traffic for all 36 sensors.

        Input:
            traffic_history

        Expected shape:
            (10, 36)
            OR
            (1, 10, 36)

        Returns:
            numpy array with shape (36,)
        """

        traffic_history = np.asarray(
            traffic_history,
            dtype=np.float32
        )

        # Add batch dimension if necessary
        if traffic_history.ndim == 2:
            traffic_history = traffic_history[np.newaxis, :]

        if traffic_history.ndim != 3:
            raise ValueError(
                "traffic_history must have shape "
                "(10, 36) or (1, 10, 36)"
            )

        if traffic_history.shape[1] != 10:
            raise ValueError(
                f"Expected 10 historical timesteps, "
                f"got {traffic_history.shape[1]}"
            )

        if traffic_history.shape[2] != 36:
            raise ValueError(
                f"Expected 36 sensors, "
                f"got {traffic_history.shape[2]}"
            )

        X = torch.tensor(
            traffic_history,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            prediction = self.model(
                X,
                self.adjacency
            )

        return prediction.cpu().numpy()[0]

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Traffic Prediction Adapter")
    print("=" * 60)

    predictor = TrafficPredictionAdapter()

    # Dummy traffic history:
    # 10 timesteps × 36 sensors
    test_input = np.zeros(
        (10, 36),
        dtype=np.float32
    )

    predictions = predictor.predict(test_input)

    print()
    print("Input shape:")
    print(test_input.shape)

    print()
    print("Prediction shape:")
    print(predictions.shape)

    print()
    print("Predictions:")
    print(predictions)

    print()
    print("Adapter test completed successfully.")