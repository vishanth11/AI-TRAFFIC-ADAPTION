import numpy as np


class TrafficHistory:
    """
    Maintains the historical traffic states required
    by the SpatioTemporal GNN.

    The GNN expects:

        10 timesteps × 36 sensors
    """

    def __init__(self, history_length=10, num_sensors=36):

        self.history_length = history_length
        self.num_sensors = num_sensors

        self.history = []

    def add_state(self, state):

        state = np.asarray(
            state,
            dtype=np.float32
        )

        if state.shape != (self.num_sensors,):
            raise ValueError(
                f"Expected state shape "
                f"({self.num_sensors},), "
                f"got {state.shape}"
            )

        self.history.append(state)

        # Keep only the latest N states
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def is_ready(self):

        return len(self.history) >= self.history_length

    def get_history(self):

        if not self.is_ready():
            raise ValueError(
                f"Need {self.history_length} "
                f"timesteps, but only "
                f"{len(self.history)} available"
            )

        return np.array(
            self.history,
            dtype=np.float32
        )

    def get_padded_history(self):

        """
        Returns history even when fewer than 10
        timesteps are available.

        Missing earlier timesteps are filled
        with zeros.

        Useful when starting a live system.
        """

        if len(self.history) == 0:

            return np.zeros(
                (self.history_length, self.num_sensors),
                dtype=np.float32
            )

        history = list(self.history)

        while len(history) < self.history_length:

            history.insert(
                0,
                history[0].copy()
            )

        return np.array(
            history,
            dtype=np.float32
        )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Traffic History")
    print("=" * 60)

    history = TrafficHistory(
        history_length=10,
        num_sensors=36
    )

    # Add 10 example states
    for i in range(10):

        state = np.zeros(
            36,
            dtype=np.float32
        )

        state[0] = i + 1

        history.add_state(state)

    print()
    print("History ready:")
    print(history.is_ready())

    result = history.get_history()

    print()
    print("History shape:")
    print(result.shape)

    print()
    print("Sensor 0 history:")
    print(result[:, 0])

    print()
    print("Traffic History test completed successfully.")