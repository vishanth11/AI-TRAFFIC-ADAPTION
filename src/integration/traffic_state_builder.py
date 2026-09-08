import numpy as np


VEHICLE_TYPES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "emergency_vehicle"
}


class TrafficStateBuilder:
    """
    Converts Member 2 perception output into
    the 36-sensor representation required by
    Member 3's GNN.

    Each sensor represents a configurable region.

    Sensor configuration format:

    {
        "sensor_id": 0,
        "x1": 0,
        "y1": 0,
        "x2": 100,
        "y2": 100
    }
    """

    def __init__(self, sensors=None, num_sensors=36):

        self.num_sensors = num_sensors

        if sensors is None:
            self.sensors = self._create_default_sensors()
        else:
            self.sensors = sensors

        if len(self.sensors) != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensors, "
                f"got {len(self.sensors)}"
            )

    def _create_default_sensors(self):
        """
        Creates 36 placeholder sensor regions.

        These are only for testing.

        Real deployment should load sensor regions
        from the road/junction topology configuration.
        """

        sensors = []

        for i in range(self.num_sensors):

            sensors.append({
                "sensor_id": i,

                # Placeholder region.
                # Replace with topology-derived regions later.
                "x1": 0,
                "y1": 0,
                "x2": 99999,
                "y2": 99999
            })

        return sensors

    @staticmethod
    def _vehicle_center(detection):

        position = detection.get("position", {})

        return (
            float(position.get("x", 0)),
            float(position.get("y", 0))
        )

    @staticmethod
    def _inside_region(x, y, sensor):

        return (
            sensor["x1"] <= x <= sensor["x2"]
            and
            sensor["y1"] <= y <= sensor["y2"]
        )

    def build_current_state(self, detections):

        state = np.zeros(
            self.num_sensors,
            dtype=np.float32
        )

        for detection in detections:

            # Ignore invalid detections
            if not isinstance(detection, dict):
                continue

            # Get detected object type
            vehicle_type = detection.get("type")

            # Ignore pedestrians and unknown object types
            if vehicle_type not in VEHICLE_TYPES:
                continue

            # Get vehicle center position
            x, y = self._vehicle_center(detection)

            # Find which sensor contains the vehicle
            for sensor in self.sensors:

                if self._inside_region(x, y, sensor):

                    sensor_id = sensor["sensor_id"]

                    state[sensor_id] += 1

                    break

        return state

    def build_from_frame(self, frame_data):

        if not isinstance(frame_data, dict):
            raise ValueError(
                "frame_data must be a dictionary"
            )

        detections = frame_data.get(
            "detections",
            []
        )

        return self.build_current_state(
            detections
        )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Traffic State Builder")
    print("=" * 60)

    builder = TrafficStateBuilder()

    test_frame = {
        "frame": 1,
        "detections": [
            {
                "vehicle_id": 1,
                "type": "car",
                "confidence": 0.91,
                "position": {
                    "x": 100,
                    "y": 100
                },
                "speed": 35.0,
                "direction": "right",
                "emergency": False
            },
            {
                "vehicle_id": 2,
                "type": "truck",
                "confidence": 0.88,
                "position": {
                    "x": 300,
                    "y": 200
                },
                "speed": 20.0,
                "direction": "left",
                "emergency": False
            },
            {
                "vehicle_id": 3,
                "type": "person",
                "confidence": 0.95,
                "position": {
                    "x": 500,
                    "y": 300
                },
                "speed": 5.0,
                "direction": "right",
                "emergency": False
            }
        ]
    }

    state = builder.build_from_frame(test_frame)

    print()
    print("Number of sensors:")
    print(len(state))

    print()
    print("State shape:")
    print(state.shape)

    print()
    print("Total detected vehicles:")
    print(int(state.sum()))

    print()
    print("Sensor state:")
    print(state)

    print()
    print("Traffic State Builder test completed successfully.")