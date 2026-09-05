import numpy as np

from src.integration.topology import JunctionTopology
from src.integration.conflict_graph import ConflictGraph
from src.integration.phase_generator import PhaseGenerator
from src.integration.traffic_state_builder import TrafficStateBuilder
from src.integration.traffic_history import TrafficHistory
from src.integration.prediction_adapter import TrafficPredictionAdapter
from src.integration.decision_engine import DecisionEngine


def test_gnn_to_decision_integration():

    print("\n" + "=" * 70)
    print("Testing GNN → Phase → Decision Engine Integration")
    print("=" * 70)

    # ==========================================================
    # 1. CREATE JUNCTION TOPOLOGY
    # ==========================================================

    topology = JunctionTopology("J1")

    topology.add_road("A", lanes=2)
    topology.add_road("B", lanes=2)
    topology.add_road("C", lanes=1)

    topology.add_movement(
        "A_to_B",
        "A",
        "B",
        lanes=2
    )

    topology.add_movement(
        "A_to_C",
        "A",
        "C",
        lanes=1
    )

    topology.add_movement(
        "B_to_A",
        "B",
        "A",
        lanes=2
    )

    topology.add_movement(
        "C_to_A",
        "C",
        "A",
        lanes=1
    )

    assert topology.validate() is True

    print()
    print("Topology: PASS")

    # ==========================================================
    # 2. CREATE CONFLICT GRAPH
    # ==========================================================

    graph = ConflictGraph(topology)

    graph.add_conflict(
        "A_to_B",
        "C_to_A"
    )

    graph.add_conflict(
        "A_to_C",
        "B_to_A"
    )

    assert graph.validate() is True

    print("Conflict Graph: PASS")

    # ==========================================================
    # 3. GENERATE VALID SIGNAL PHASES
    # ==========================================================

    generator = PhaseGenerator(graph)

    phases = generator.generate_maximal_phases()

    assert len(phases) > 0

    print("Phase Generator: PASS")

    print()
    print("Generated phases:")

    for index, phase in enumerate(phases, start=1):
        print(f"Phase_{index}: {phase}")

    # ==========================================================
    # 4. CREATE TRAFFIC STATE BUILDER
    # ==========================================================

    builder = TrafficStateBuilder(
        num_sensors=36
    )

    # ==========================================================
    # 5. CREATE TRAFFIC HISTORY
    # ==========================================================

    history = TrafficHistory(
        history_length=10,
        num_sensors=36
    )

    # ==========================================================
    # 6. SIMULATE 10 TRAFFIC FRAMES
    # ==========================================================

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

                {
                    "vehicle_id": 3,
                    "type": "motorcycle",
                    "confidence": 0.91,
                    "position": {
                        "x": 500,
                        "y": 300
                    },
                    "speed": 25.0,
                    "direction": "up",
                    "emergency": False
                },

                # Pedestrian should be ignored
                {
                    "vehicle_id": 4,
                    "type": "person",
                    "confidence": 0.94,
                    "position": {
                        "x": 600,
                        "y": 400
                    },
                    "speed": 4.0,
                    "direction": "up",
                    "emergency": False
                }
            ]
        }

        # Convert detections into 36-sensor state
        current_state = builder.build_from_frame(
            frame_data
        )

        # Store state
        history.add_state(
            current_state
        )

    # ==========================================================
    # 7. VERIFY TRAFFIC HISTORY
    # ==========================================================

    assert history.is_ready()

    traffic_history = history.get_history()

    assert traffic_history.shape == (10, 36)

    print()
    print("Traffic History: PASS")
    print("History shape:", traffic_history.shape)

    # Three vehicles should be counted
    # Pedestrian should be ignored.
    assert int(current_state.sum()) == 3

    print(
        "Current vehicle count:",
        int(current_state.sum())
    )

    # ==========================================================
    # 8. LOAD MEMBER 3 GNN
    # ==========================================================

    predictor = TrafficPredictionAdapter()

    # ==========================================================
    # 9. GENERATE TRAFFIC PREDICTIONS
    # ==========================================================

    predictions = predictor.predict(
        traffic_history
    )

    assert isinstance(
        predictions,
        np.ndarray
    )

    assert predictions.shape == (36,)

    assert np.all(
        np.isfinite(predictions)
    )

    print()
    print("GNN Prediction: PASS")
    print("Prediction shape:", predictions.shape)

    # ==========================================================
    # 10. CREATE DECISION ENGINE
    # ==========================================================

    engine = DecisionEngine()

    # ==========================================================
    # 11. MAP GNN PREDICTIONS TO PHASES
    # ==========================================================
    #
    # For this integration test, we assign sensor groups
    # to phases.
    #
    # NOTE:
    # These are TEST mappings only.
    #
    # Real deployment must obtain sensor-to-movement
    # mappings from topology/SUMO configuration.

    phase_sensor_mapping = {}

    for index, phase in enumerate(
        phases,
        start=1
    ):

        # Example mapping:
        # phase index determines test sensor group.
        #
        # We use three sensors per phase so that
        # different phases receive different prediction
        # values.

        start_sensor = (
            (index - 1) * 3
        )

        sensor_ids = [
            start_sensor,
            start_sensor + 1,
            start_sensor + 2
        ]

        # Make sure sensor IDs are valid.
        sensor_ids = [
            sensor_id
            for sensor_id in sensor_ids
            if sensor_id < 36
        ]

        phase_sensor_mapping[
            f"Phase_{index}"
        ] = sensor_ids

    print()
    print("Phase → Sensor mapping:")
    print(phase_sensor_mapping)

    # ==========================================================
    # 12. CALCULATE PHASE SCORES
    # ==========================================================

    phase_scores = {}

    for index, phase in enumerate(
        phases,
        start=1
    ):

        phase_id = f"Phase_{index}"

        sensor_ids = phase_sensor_mapping[
            phase_id
        ]

        # Convert GNN prediction into
        # phase-level prediction score.
        prediction_score = (
            engine.calculate_phase_prediction(
                predictions,
                sensor_ids
            )
        )

        # ------------------------------------------------------
        # Test values for other decision factors.
        #
        # These will later come from:
        #
        # Demand   → Traffic State
        # Priority → Emergency/Public Transport
        # Fairness → Waiting time
        # Risk     → Member 4 Safety Intelligence
        # ------------------------------------------------------

        demand = 0.7

        priority = 0.5

        fairness = 0.6

        risk = 0.2

        score = engine.calculate_phase_score(
            demand=demand,
            prediction=prediction_score,
            priority=priority,
            fairness=fairness,
            risk=risk
        )

        phase_scores[
            phase_id
        ] = score

        print()
        print(phase_id)
        print("  Movements:", phase)
        print("  Sensors:", sensor_ids)
        print(
            "  Prediction:",
            prediction_score
        )
        print(
            "  Final Score:",
            score
        )

    # ==========================================================
    # 13. SELECT BEST PHASE
    # ==========================================================

    result = engine.select_best_phase(
        phase_scores
    )

    # ==========================================================
    # 14. VALIDATE DECISION
    # ==========================================================

    assert result["selected_phase"] in phase_scores

    assert 0 <= result["score"] <= 1

    assert len(
        result["all_scores"]
    ) == len(phases)

    print()
    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    print(
        "Selected Phase:",
        result["selected_phase"]
    )

    print(
        "Score:",
        result["score"]
    )

    print()
    print("=" * 70)
    print("GNN → PHASE → DECISION INTEGRATION PASSED")
    print("=" * 70)