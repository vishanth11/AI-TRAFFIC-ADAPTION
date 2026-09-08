class TrafficStateHistory:

    def __init__(self, max_length=60):

        # Maximum number of states to remember
        self.max_length = max_length

        # List containing previous traffic states
        self.history = []

    def add_state(self, traffic_state):

        # Add the newest state
        self.history.append(traffic_state)

        # Keep only the most recent states
        if len(self.history) > self.max_length:

            self.history.pop(0)

    def get_history(self):

        return self.history

    def get_latest(self):

        if not self.history:

            return None

        return self.history[-1]

    def get_queue_history(self, direction):

        queue_history = []

        for state in self.history:

            approaches = state.get(
                "approaches",
                {}
            )

            if direction in approaches:

                queue = approaches[
                    direction
                ]["queue_length"]

                queue_history.append(queue)

        return queue_history

    def get_waiting_history(self, direction):

        waiting_history = []

        for state in self.history:

            approaches = state.get(
                "approaches",
                {}
            )

            if direction in approaches:

                waiting = approaches[
                    direction
                ]["average_waiting_time"]

                waiting_history.append(waiting)

        return waiting_history

    def get_speed_history(self, direction):

        speed_history = []

        for state in self.history:

            approaches = state.get(
                "approaches",
                {}
            )

            if direction in approaches:

                speed = approaches[
                    direction
                ]["average_speed"]

                speed_history.append(speed)

        return speed_history

    def clear(self):

        self.history = []

    def size(self):

        return len(self.history)