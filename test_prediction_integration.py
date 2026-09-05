import numpy as np

from src.integration.traffic_state_builder import TrafficStateBuilder
from src.integration.traffic_history import TrafficHistory
from src.integration.prediction_adapter import TrafficPredictionAdapter


def test_traffic_prediction_integration():

    print("\n" + "=" * 60)
    print("Testing Traffic State → History → GNN Prediction")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Create Traffic State Builder
    # --------------------------------------------------

    builder = TrafficStateBuilder(
        num_sensors=36
    )

    # --------------------------------------------------
    # 2. Create Traffic History
    # --------------------------------------------------

    history = TrafficHistory(
        history_length=10,
        num_sensors=36
    )

    # --------------------------------------------------
    # 3. Simulate 10 traffic frames
    # --------------------------------------------------

    for frame_number in range(10):

        frame_data = {
            "frame": frame_number,

            "detections": [
                {
                    "vehicle_id": 1,
                    "type": "car",
                    "confidence": 0.95,
                    "position": {
                        "x": 100,
                        "y": 100
                    },
                    "speed": 30.0,
                    "direction": "right",
                    "emergency": False
                },

                {
                    "vehicle_id": 2,
                    "type": "truck",
                    "confidence": 0.90,
                    "position": {
                        "x": 300,
                        "y": 200
                    },
                    "speed": 20.0,
                    "direction": "left",
                    "emergency": False
                },

                # This should NOT be counted
                {
                    "vehicle_id": 3,
                    "type": "person",
                    "confidence": 0.92,
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

        # Convert perception output
        # into 36-sensor state
        current_state = builder.build_from_frame(
            frame_data
        )

        # Add state to history
        history.add_state(
            current_state
        )

    # --------------------------------------------------
    # 4. Verify history
    # --------------------------------------------------

    assert history.is_ready()

    traffic_history = history.get_history()

    print()
    print("Traffic History Shape:")
    print(traffic_history.shape)

    assert traffic_history.shape == (10, 36)

    # --------------------------------------------------
    # 5. Verify vehicle count
    # --------------------------------------------------

    print()
    print("Vehicles in latest state:")
    print(int(current_state.sum()))

    assert int(current_state.sum()) == 2

    # --------------------------------------------------
    # 6. Load GNN prediction adapter
    # --------------------------------------------------

    predictor = TrafficPredictionAdapter()

    # --------------------------------------------------
    # 7. Generate prediction
    # --------------------------------------------------

    predictions = predictor.predict(
        traffic_history
    )

    print()
    print("Prediction Shape:")
    print(predictions.shape)

    print()
    print("Predictions:")
    print(predictions)

    # --------------------------------------------------
    # 8. Validate prediction
    # --------------------------------------------------

    assert isinstance(
        predictions,
        np.ndarray
    )

    assert predictions.shape == (36,)

    assert np.all(
        np.isfinite(predictions)
    )

    print()
    print("=" * 60)
    print("GNN INTEGRATION TEST PASSED")
    print("=" * 60)