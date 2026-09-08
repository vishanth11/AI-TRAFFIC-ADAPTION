import traci

from state.state_engine import TrafficStateEngine
from state.state_history import TrafficStateHistory
from state.junction_config import JUNCTIONS
from state.corridor_state import CorridorStateEngine
from state.decision_interface import DecisionInterface
from control.adaptive_signal import AdaptiveSignalController


def main():

    print("=" * 60)
    print("TRAFFIC AI - CORRIDOR DECISION PIPELINE")
    print("=" * 60)

    engines = {}
    histories = {}

    # --------------------------------------------------
    # Create state engines for J1, J2 and J3
    # --------------------------------------------------

    for junction_name, config in JUNCTIONS.items():

        engines[junction_name] = TrafficStateEngine(
            junction_id=config["junction_id"],
            approaches=config["approaches"],
            traffic_lights=config["traffic_lights"],
            approach_length=200.0
        )

        histories[junction_name] = TrafficStateHistory(
            max_length=60
        )

    # --------------------------------------------------
    # Corridor state engine
    # --------------------------------------------------

    corridor_engine = CorridorStateEngine(
        engines
    )

    # --------------------------------------------------
    # Member 5 decision interface
    # --------------------------------------------------

    decision_interface = DecisionInterface()

    # --------------------------------------------------
    # Adaptive signal controller
    # --------------------------------------------------

    signal_controller = AdaptiveSignalController(
        min_green=20,
        max_green=60,
        congestion_threshold=0.50,
        extension_seconds=10
    )

    print()
    print("Configured junctions:")

    for junction_name, config in JUNCTIONS.items():

        print(
            f"  {junction_name} -> "
            f"{config['junction_id']}"
        )

    print()
    print("Decision interface: READY")

    print()
    print("Phase-aware adaptive signal controller: READY")

    print()
    print("Starting SUMO...")
    print()

    # --------------------------------------------------
    # Start SUMO
    # --------------------------------------------------

    traci.start([
        "sumo",
        "-c",
        "simulation/configs/corridor.sumocfg"
    ])

    try:

        # --------------------------------------------------
        # Main simulation loop
        # --------------------------------------------------

        for step in range(900):

            traci.simulationStep()

            # --------------------------------------------------
            # Calculate corridor state
            # --------------------------------------------------

            corridor_state = (
                corridor_engine.calculate_corridor_state(
                    traci
                )
            )

            # --------------------------------------------------
            # Store state history
            # --------------------------------------------------

            for junction_name, state in (
                corridor_state["junctions"].items()
            ):

                histories[junction_name].add_state(
                    state
                )

            # --------------------------------------------------
            # Build Member 5 decision interface
            # --------------------------------------------------

            decision_state = (
                decision_interface.build_corridor_decision(
                    corridor_state
                )
            )

            # --------------------------------------------------
            # Adaptive signal decisions
            # --------------------------------------------------

            signal_decisions = {}

            for junction_name, state in (
                corridor_state["junctions"].items()
            ):

                # IMPORTANT:
                # TraCI is passed to the controller so it can
                # inspect the actual SUMO signal phase and
                # controlled links.

                decision = signal_controller.decide(
                    state,
                    traci
                )

                applied = (
                    signal_controller.apply_decision(
                        traci,
                        decision
                    )
                )

                decision["applied_to_sumo"] = applied

                signal_decisions[junction_name] = (
                    decision
                )

            # --------------------------------------------------
            # Print every 10 simulation steps
            # --------------------------------------------------

            if step % 10 == 0:

                print()
                print("-" * 60)

                print(
                    f"SIMULATION STEP: {step}"
                )

                print("-" * 60)

                print(
                    f"Corridor vehicles: "
                    f"{decision_state['total_vehicles']}"
                )

                print(
                    f"Worst junction: "
                    f"{decision_state['worst_junction']}"
                )

                print(
                    f"Highest congestion: "
                    f"{decision_state['highest_congestion_score']:.3f}"
                )

                print()
                print("MEMBER 5 DECISION INTERFACE")

                print(
                    f"  Corridor ID: "
                    f"{decision_state['corridor_id']}"
                )

                print(
                    f"  Junction count: "
                    f"{decision_state['junction_count']}"
                )

                for junction_name, state in (
                    decision_state["junctions"].items()
                ):

                    print()
                    print(
                        f"  {junction_name} "
                        f"({state['junction_id']})"
                    )

                    print(
                        f"    Vehicles: "
                        f"{state['vehicle_count']}"
                    )

                    print(
                        f"    Congestion: "
                        f"{state['congestion_score']:.3f}"
                    )

                    print(
                        f"    Worst approach: "
                        f"{state['most_congested_approach']}"
                    )

                # --------------------------------------------------
                # Signal decisions
                # --------------------------------------------------

                print()
                print("PHASE-AWARE SIGNAL DECISIONS")

                for junction_name, decision in (
                    signal_decisions.items()
                ):

                    print(
                        f"  {junction_name}: "
                        f"{decision['action']} "
                        f"(applied="
                        f"{decision['applied_to_sumo']})"
                    )

                    print(
                        f"    Current phase: "
                        f"{decision['current_phase']}"
                    )

                    print(
                        f"    Target phase: "
                        f"{decision['target_phase']}"
                    )

                    print(
                        f"    Congested approach: "
                        f"{decision['most_congested_approach']}"
                    )

                    print(
                        f"    Serves congested approach: "
                        f"{decision['serves_congested_approach']}"
                    )

                    print(
                        f"    Reason: "
                        f"{decision['reason']}"
                    )

                # --------------------------------------------------
                # State history
                # --------------------------------------------------

                print()
                print("STATE HISTORY")

                for junction_name, history in (
                    histories.items()
                ):

                    print(
                        f"  {junction_name}: "
                        f"{history.size()} states stored"
                    )

    finally:

        traci.close()

        print()
        print("=" * 60)
        print("SIMULATION FINISHED")
        print("=" * 60)


if __name__ == "__main__":
    main()