import os
import sys

import traci
import numpy as np

from integration.prediction_adapter import TrafficPredictionAdapter
from integration.traffic_history import TrafficHistory


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "simulation",
    "configs",
    "corridor.sumocfg"
)


SUMO_CMD = [
    "sumo",
    "-c",
    SUMO_CONFIG
]


# ============================================================
# GNN CONFIGURATION
# ============================================================

NUM_SENSORS = 36
HISTORY_LENGTH = 10

TOTAL_STEPS = 50

PREDICTION_INTERVAL = 10


# ============================================================
# TOPOLOGY CONFIGURATION
# ============================================================

# Main corridor junctions.
#
# These are the junctions whose incoming roads form the
# observation space for the current corridor.
#
# If the SUMO network is changed later, this list can be
# replaced by automatic junction selection.

TARGET_JUNCTIONS = [
    "A0",
    "B0",
    "C0"
]


# Every incoming road gets three observation zones:
#
#   FAR    -> vehicles far from junction
#   MIDDLE -> vehicles approaching junction
#   NEAR   -> vehicles close to junction
#
# 12 incoming roads x 3 zones = 36 sensors.

ZONES_PER_ROAD = 3


# ============================================================
# TOPOLOGY-AWARE SENSOR BUILDER
# ============================================================

class SUMOTrafficStateBuilder:
    """
    Converts SUMO traffic into the 36-sensor representation
    required by Member 3's GNN.

    Sensor construction:

        Junction
            |
            +-- incoming road 1
            |      +-- FAR
            |      +-- MIDDLE
            |      +-- NEAR
            |
            +-- incoming road 2
            |      +-- FAR
            |      +-- MIDDLE
            |      +-- NEAR
            |
            ...

    For the current A0/B0/C0 corridor:

        3 junctions
        x 4 incoming roads
        x 3 spatial zones
        = 36 sensors

    IMPORTANT:

    The original physical meaning of Member 3's 36 training
    sensors is unavailable in the repository.

    Therefore this class creates a deterministic deployment
    observation mapping based on actual SUMO topology.

    It does NOT claim that these sensor IDs reproduce the
    original training sensor locations.
    """

    def __init__(
        self,
        num_sensors=NUM_SENSORS,
        target_junctions=None
    ):

        self.num_sensors = num_sensors

        if target_junctions is None:
            target_junctions = TARGET_JUNCTIONS

        self.target_junctions = list(
            target_junctions
        )

        self.incoming_roads = []

        self.sensor_mapping = {}

        self._build_topology_mapping()

    # ========================================================
    # BUILD TOPOLOGY MAPPING
    # ========================================================

    def _build_topology_mapping(self):

        discovered_roads = []

        for junction_id in self.target_junctions:

            if junction_id not in traci.trafficlight.getIDList():
                print(
                    f"WARNING: Junction {junction_id} "
                    f"is not a traffic light."
                )
                continue

            controlled_links = (
                traci.trafficlight.getControlledLinks(
                    junction_id
                )
            )

            junction_roads = set()

            for link_group in controlled_links:

                if not link_group:
                    continue

                for link in link_group:

                    if not link:
                        continue

                    from_lane = link[0]

                    if not from_lane:
                        continue

                    try:

                        from_edge = (
                            traci.lane.getEdgeID(
                                from_lane
                            )
                        )

                    except Exception:
                        continue

                    # Internal SUMO edges are not incoming
                    # physical roads.
                    if from_edge.startswith(":"):
                        continue

                    junction_roads.add(
                        from_edge
                    )

            for road_id in sorted(junction_roads):

                discovered_roads.append(
                    (
                        junction_id,
                        road_id
                    )
                )

        # Remove duplicates while preserving topology order.
        unique_roads = []

        seen = set()

        for junction_id, road_id in discovered_roads:

            key = (
                junction_id,
                road_id
            )

            if key in seen:
                continue

            seen.add(key)

            unique_roads.append(
                (
                    junction_id,
                    road_id
                )
            )

        self.incoming_roads = unique_roads

        expected_roads = (
            self.num_sensors
            // ZONES_PER_ROAD
        )

        if len(self.incoming_roads) != expected_roads:

            print()
            print(
                "WARNING: Topology does not contain "
                f"{expected_roads} incoming roads."
            )

            print(
                f"Discovered incoming roads: "
                f"{len(self.incoming_roads)}"
            )

            print(
                f"Required for {self.num_sensors} sensors: "
                f"{expected_roads}"
            )

        # ----------------------------------------------------
        # Create deterministic sensor IDs.
        # ----------------------------------------------------

        sensor_id = 0

        for junction_id, road_id in self.incoming_roads:

            for zone_id in range(
                ZONES_PER_ROAD
            ):

                if sensor_id >= self.num_sensors:
                    break

                zone_name = self._zone_name(
                    zone_id
                )

                self.sensor_mapping[
                    sensor_id
                ] = {
                    "sensor_id": sensor_id,
                    "junction_id": junction_id,
                    "road_id": road_id,
                    "zone_id": zone_id,
                    "zone": zone_name
                }

                sensor_id += 1

        # ----------------------------------------------------
        # If topology has fewer than 36 sensors,
        # remaining positions stay zero.
        # ----------------------------------------------------

        while sensor_id < self.num_sensors:

            self.sensor_mapping[
                sensor_id
            ] = {
                "sensor_id": sensor_id,
                "junction_id": None,
                "road_id": None,
                "zone_id": None,
                "zone": "unused"
            }

            sensor_id += 1

    # ========================================================
    # ZONE NAME
    # ========================================================

    @staticmethod
    def _zone_name(zone_id):

        if zone_id == 0:
            return "far"

        if zone_id == 1:
            return "middle"

        if zone_id == 2:
            return "near"

        return "unknown"

    # ========================================================
    # BUILD CURRENT STATE
    # ========================================================

    def build_state(self):

        state = np.zeros(
            self.num_sensors,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Build lookup:
        #
        # (junction, road) -> sensor IDs
        # ----------------------------------------------------

        road_to_sensors = {}

        for sensor_id, info in self.sensor_mapping.items():

            junction_id = info["junction_id"]
            road_id = info["road_id"]

            if junction_id is None:
                continue

            key = (
                junction_id,
                road_id
            )

            if key not in road_to_sensors:

                road_to_sensors[key] = []

            road_to_sensors[key].append(
                sensor_id
            )

        # ----------------------------------------------------
        # Examine all active vehicles.
        # ----------------------------------------------------

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        for vehicle_id in vehicle_ids:

            try:

                lane_id = (
                    traci.vehicle.getLaneID(
                        vehicle_id
                    )
                )

                if not lane_id:
                    continue

                edge_id = (
                    traci.lane.getEdgeID(
                        lane_id
                    )
                )

                # Ignore internal SUMO junction edges.
                if edge_id.startswith(":"):
                    continue

                # Find the junction associated with this
                # incoming road.
                junction_id = (
                    self._find_junction_for_road(
                        edge_id,
                        road_to_sensors
                    )
                )

                if junction_id is None:
                    continue

                key = (
                    junction_id,
                    edge_id
                )

                sensor_ids = (
                    road_to_sensors.get(
                        key,
                        []
                    )
                )

                if len(sensor_ids) != ZONES_PER_ROAD:
                    continue

                # ------------------------------------------------
                # Determine position along incoming road.
                # ------------------------------------------------

                lane_position = (
                    traci.vehicle.getLanePosition(
                        vehicle_id
                    )
                )

                lane_length = (
                    traci.lane.getLength(
                        lane_id
                    )
                )

                if lane_length <= 0:
                    continue

                # Normalize position:
                #
                # 0.0 = beginning of lane
                # 1.0 = junction end
                #
                normalized_position = (
                    lane_position
                    / lane_length
                )

                normalized_position = max(
                    0.0,
                    min(
                        1.0,
                        normalized_position
                    )
                )

                # ------------------------------------------------
                # Determine spatial zone.
                # ------------------------------------------------

                if normalized_position < 1.0 / 3.0:

                    zone_id = 0

                elif normalized_position < 2.0 / 3.0:

                    zone_id = 1

                else:

                    zone_id = 2

                sensor_id = sensor_ids[
                    zone_id
                ]

                state[sensor_id] += 1.0

            except Exception:

                # Ignore vehicles that disappear between
                # TraCI calls.
                continue

        return state

    # ========================================================
    # FIND JUNCTION FOR ROAD
    # ========================================================

    @staticmethod
    def _find_junction_for_road(
        road_id,
        road_to_sensors
    ):

        matching_keys = [
            key
            for key in road_to_sensors
            if key[1] == road_id
        ]

        if len(matching_keys) == 1:

            return matching_keys[0][0]

        return None

    # ========================================================
    # PRINT TOPOLOGY
    # ========================================================

    def print_mapping(self):

        print()
        print("=" * 70)
        print("TOPOLOGY-DERIVED 36-SENSOR MAPPING")
        print("=" * 70)

        print()

        print(
            f"Incoming roads discovered: "
            f"{len(self.incoming_roads)}"
        )

        print(
            f"Sensors configured: "
            f"{self.num_sensors}"
        )

        print()

        for sensor_id in range(
            self.num_sensors
        ):

            info = self.sensor_mapping[
                sensor_id
            ]

            if info["road_id"] is None:

                print(
                    f"  Sensor {sensor_id}: UNUSED"
                )

            else:

                print(
                    f"  Sensor {sensor_id}: "
                    f"{info['junction_id']} | "
                    f"{info['road_id']} | "
                    f"{info['zone']}"
                )

        print()


# ============================================================
# PRINT TRAFFIC STATE
# ============================================================

def print_state(
    simulation_step,
    state,
    state_builder
):

    print()
    print("=" * 70)

    print(
        f"TRAFFIC STATE - STEP {simulation_step}"
    )

    print("=" * 70)

    print()

    print(
        f"Total vehicles represented: "
        f"{int(state.sum())}"
    )

    print(
        f"Non-zero sensors: "
        f"{int(np.count_nonzero(state))}"
    )

    print()

    print("Sensor state:")

    for sensor_id, value in enumerate(
        state
    ):

        if value <= 0:
            continue

        info = state_builder.sensor_mapping[
            sensor_id
        ]

        print(
            f"  Sensor {sensor_id}: "
            f"{int(value)} vehicles | "
            f"{info['junction_id']} | "
            f"{info['road_id']} | "
            f"{info['zone']}"
        )


# ============================================================
# PRINT PREDICTION
# ============================================================

def print_prediction(
    simulation_step,
    prediction,
    state_builder
):

    print()
    print("=" * 70)

    print(
        f"GNN PREDICTION - STEP {simulation_step}"
    )

    print("=" * 70)

    print()

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    print(
        f"Total predicted traffic: "
        f"{prediction.sum():.4f}"
    )

    print(
        f"Maximum predicted sensor: "
        f"{int(np.argmax(prediction))}"
    )

    print(
        f"Maximum prediction: "
        f"{prediction.max():.4f}"
    )

    print()

    print("Predicted traffic:")

    for sensor_id, value in enumerate(
        prediction
    ):

        info = state_builder.sensor_mapping[
            sensor_id
        ]

        print(
            f"  Sensor {sensor_id}: "
            f"{value:.6f} | "
            f"{info['junction_id']} | "
            f"{info['road_id']} | "
            f"{info['zone']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SUMO + TOPOLOGY + MEMBER 3 GNN TEST"
    )
    print("=" * 70)

    print()

    print(
        f"SUMO configuration: "
        f"{SUMO_CONFIG}"
    )

    # --------------------------------------------------------
    # Validate SUMO configuration.
    # --------------------------------------------------------

    if not os.path.exists(
        SUMO_CONFIG
    ):

        raise FileNotFoundError(
            "SUMO configuration not found: "
            f"{SUMO_CONFIG}"
        )

    # --------------------------------------------------------
    # Start SUMO.
    # --------------------------------------------------------

    print()
    print("Starting SUMO...")

    traci.start(
        SUMO_CMD
    )

    try:

        # ----------------------------------------------------
        # Load prediction model.
        # ----------------------------------------------------

        print()
        print(
            "Loading Member 3 prediction model..."
        )

        predictor = (
            TrafficPredictionAdapter()
        )

        # ----------------------------------------------------
        # Create history.
        # ----------------------------------------------------

        history = (
            TrafficHistory(
                history_length=HISTORY_LENGTH,
                num_sensors=NUM_SENSORS
            )
        )

        # ----------------------------------------------------
        # Create topology-aware state builder.
        # ----------------------------------------------------

        state_builder = (
            SUMOTrafficStateBuilder(
                num_sensors=NUM_SENSORS,
                target_junctions=TARGET_JUNCTIONS
            )
        )

        # ----------------------------------------------------
        # Display sensor mapping.
        # ----------------------------------------------------

        state_builder.print_mapping()

        print(
            "Topology-aware SUMO prediction "
            "pipeline initialized."
        )

        print()

        # ----------------------------------------------------
        # Simulation loop.
        # ----------------------------------------------------

        for step in range(
            TOTAL_STEPS
        ):

            simulation_step = (
                step + 1
            )

            # ------------------------------------------------
            # Advance SUMO.
            # ------------------------------------------------

            traci.simulationStep()

            # ------------------------------------------------
            # Build topology-aware state.
            # ------------------------------------------------

            current_state = (
                state_builder.build_state()
            )

            # ------------------------------------------------
            # Add state to GNN history.
            # ------------------------------------------------

            history.add_state(
                current_state
            )

            # ------------------------------------------------
            # Print current state.
            # ------------------------------------------------

            print_state(
                simulation_step,
                current_state,
                state_builder
            )

            # ------------------------------------------------
            # Run GNN every 10 steps.
            # ------------------------------------------------

            if (
                history.is_ready()
                and
                simulation_step
                % PREDICTION_INTERVAL
                == 0
            ):

                traffic_history = (
                    history.get_history()
                )

                print()
                print(
                    "GNN input shape: "
                    f"{traffic_history.shape}"
                )

                # --------------------------------------------
                # GNN prediction.
                # --------------------------------------------

                prediction = (
                    predictor.predict(
                        traffic_history
                    )
                )

                # --------------------------------------------
                # Print prediction.
                # --------------------------------------------

                print_prediction(
                    simulation_step,
                    prediction,
                    state_builder
                )

        # ----------------------------------------------------
        # Final status.
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "SUMO + TOPOLOGY + GNN TEST COMPLETE"
        )
        print("=" * 70)

        print()

        print(
            f"Simulation time: "
            f"{traci.simulation.getTime():.1f}"
        )

        print(
            f"History ready: "
            f"{history.is_ready()}"
        )

        print()

    finally:

        # ----------------------------------------------------
        # Close SUMO.
        # ----------------------------------------------------

        traci.close()

        print(
            "SUMO connection closed."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()