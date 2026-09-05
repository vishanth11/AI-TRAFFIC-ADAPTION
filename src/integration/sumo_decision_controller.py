import traci

from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.sumo_phase_adapter import SUMOPhaseAdapter
from integration.sumo_conflict_adapter import SUMOConflictAdapter
from integration.phase_generator import PhaseGenerator
from integration.decision_engine import DecisionEngine
from integration.sumo_demand import SUMODemandCalculator


# ============================================================
# SUMO CONFIGURATION
# ============================================================

SUMO_CMD = [
    "sumo",
    "-c",
    "simulation/configs/corridor.sumocfg"
]


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

# Number of SUMO simulation steps before
# collecting traffic demand.
WARMUP_STEPS = 500


# Junction currently being tested.
JUNCTION_ID = "A0"


# ============================================================
# PRINT GENERATED PHASES
# ============================================================

def print_generated_phases(junction_id, phases):

    print()
    print("=" * 70)
    print(f"GENERATED PHASES: {junction_id}")
    print("=" * 70)

    print()

    print(
        f"Total generated phases: "
        f"{len(phases)}"
    )

    for index, phase in enumerate(phases):

        print()

        print(
            f"Phase {index}"
        )

        print(
            f"  {phase}"
        )


# ============================================================
# PRINT MOVEMENT DEMAND
# ============================================================

def print_movement_demand(movement_demand):

    print()
    print("=" * 70)
    print("MOVEMENT DEMAND")
    print("=" * 70)

    print()

    for movement_id, demand in movement_demand.items():

        print(
            f"  {movement_id}: "
            f"{demand}"
        )


# ============================================================
# PRINT PHASE DEMAND
# ============================================================

def print_phase_demand(phase_demand):

    print()
    print("=" * 70)
    print("PHASE DEMAND")
    print("=" * 70)

    print()

    for phase_id, demand in phase_demand.items():

        print(
            f"  {phase_id}: "
            f"{demand}"
        )


# ============================================================
# PRINT NORMALIZED PHASE SCORES
# ============================================================

def print_normalized_scores(phase_scores):

    print()
    print("=" * 70)
    print("NORMALIZED PHASE SCORES")
    print("=" * 70)

    print()

    for phase_id, score in phase_scores.items():

        print(
            f"  {phase_id}: "
            f"{score:.3f}"
        )


# ============================================================
# PRINT DECISION ENGINE RESULT
# ============================================================

def print_decision(decision):

    print()
    print("=" * 70)
    print("DECISION ENGINE RESULT")
    print("=" * 70)

    print()

    print(
        f"Selected phase: "
        f"{decision['selected_phase']}"
    )

    print(
        f"Score: "
        f"{decision['score']:.3f}"
    )

    print()

    print("All scores:")

    for phase_id, score in decision["all_scores"].items():

        print(
            f"  {phase_id}: "
            f"{score:.3f}"
        )


# ============================================================
# PRINT SELECTED MOVEMENTS
# ============================================================

def print_selected_movements(
    topology,
    phases,
    selected_phase_id
):

    selected_index = int(
        selected_phase_id.split("_")[1]
    )

    selected_movements = (
        phases[selected_index]
    )

    print()
    print("=" * 70)
    print("SELECTED MOVEMENTS")
    print("=" * 70)

    print()

    print(
        f"{selected_phase_id}:"
    )

    for movement_id in selected_movements:

        movement = (
            topology.movements[movement_id]
        )

        print(
            f"  {movement_id}: "
            f"{movement.from_road} "
            f"-> "
            f"{movement.to_road}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # START SUMO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING SUMO")
    print("=" * 70)

    print()

    traci.start(
        SUMO_CMD
    )

    try:

        # ----------------------------------------------------
        # CREATE TOPOLOGY ADAPTER
        # ----------------------------------------------------

        topology_adapter = (
            SUMOTopologyAdapter()
        )

        # ----------------------------------------------------
        # CREATE PHASE ADAPTER
        # ----------------------------------------------------

        phase_adapter = (
            SUMOPhaseAdapter()
        )

        # ----------------------------------------------------
        # CREATE CONFLICT ADAPTER
        # ----------------------------------------------------

        conflict_adapter = (
            SUMOConflictAdapter(
                phase_adapter
            )
        )

        # ----------------------------------------------------
        # CREATE DECISION ENGINE
        # ----------------------------------------------------

        decision_engine = (
            DecisionEngine()
        )

        # ----------------------------------------------------
        # CREATE DEMAND CALCULATOR
        # ----------------------------------------------------

        demand_calculator = (
            SUMODemandCalculator()
        )

        # ----------------------------------------------------
        # DISCOVER ALL JUNCTIONS
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("DISCOVERING TOPOLOGY")
        print("=" * 70)

        print()

        topologies = (
            topology_adapter
            .discover_all_junctions()
        )

        print(
            f"Discovered "
            f"{len(topologies)} "
            f"traffic-light junctions."
        )

        # ----------------------------------------------------
        # CHECK JUNCTION
        # ----------------------------------------------------

        junction_id = JUNCTION_ID

        if junction_id not in topologies:

            raise ValueError(
                f"Junction {junction_id} "
                f"was not found in SUMO."
            )

        topology = (
            topologies[junction_id]
        )

        print()

        print(
            f"Testing junction: "
            f"{junction_id}"
        )

        # ----------------------------------------------------
        # BUILD CONFLICT GRAPH
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("BUILDING CONFLICT GRAPH")
        print("=" * 70)

        print()

        conflict_graph = (
            conflict_adapter
            .build_conflict_graph(
                junction_id,
                topology,
                topology_adapter.sumo_metadata
            )
        )

        print(
            "Conflict graph created successfully."
        )

        # ----------------------------------------------------
        # GENERATE MAXIMAL PHASES
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("GENERATING PHASES")
        print("=" * 70)

        print()

        generator = (
            PhaseGenerator(
                conflict_graph
            )
        )

        phases = (
            generator
            .generate_maximal_phases()
        )

        # ----------------------------------------------------
        # PRINT GENERATED PHASES
        # ----------------------------------------------------

        print_generated_phases(
            junction_id,
            phases
        )

        # ----------------------------------------------------
        # WARM UP SUMO
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("SUMO WARM-UP")
        print("=" * 70)

        print()

        print(
            f"Advancing simulation "
            f"for {WARMUP_STEPS} steps..."
        )

        for step in range(
            WARMUP_STEPS
        ):

            traci.simulationStep()

        # ----------------------------------------------------
        # GET SIMULATION TIME
        # ----------------------------------------------------

        simulation_time = (
            traci.simulation.getTime()
        )

        # ----------------------------------------------------
        # GET ACTIVE VEHICLES
        # ----------------------------------------------------

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        # ----------------------------------------------------
        # PRINT CURRENT STATE
        # ----------------------------------------------------

        print()

        print(
            f"Simulation time: "
            f"{simulation_time:.1f}"
        )

        print(
            f"Active vehicles: "
            f"{len(vehicle_ids)}"
        )

        # ----------------------------------------------------
        # CALCULATE MOVEMENT DEMAND
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("CALCULATING MOVEMENT DEMAND")
        print("=" * 70)

        print()

        movement_demand = (
            demand_calculator
            .calculate_movement_demand(
                topology,
                topology_adapter.sumo_metadata
            )
        )

        # ----------------------------------------------------
        # PRINT MOVEMENT DEMAND
        # ----------------------------------------------------

        print_movement_demand(
            movement_demand
        )

        # ----------------------------------------------------
        # CALCULATE PHASE DEMAND
        # ----------------------------------------------------

        phase_demand = (
            demand_calculator
            .calculate_phase_demand(
                phases,
                movement_demand
            )
        )

        # ----------------------------------------------------
        # PRINT PHASE DEMAND
        # ----------------------------------------------------

        print_phase_demand(
            phase_demand
        )

        # ----------------------------------------------------
        # NORMALIZE PHASE DEMAND
        # ----------------------------------------------------

        phase_scores = (
            demand_calculator
            .normalize_phase_demand(
                phase_demand
            )
        )

        # ----------------------------------------------------
        # PRINT NORMALIZED SCORES
        # ----------------------------------------------------

        print_normalized_scores(
            phase_scores
        )

        # ----------------------------------------------------
        # RUN DECISION ENGINE
        # ----------------------------------------------------

        decision = (
            decision_engine
            .select_best_phase(
                phase_scores
            )
        )

        # ----------------------------------------------------
        # PRINT DECISION
        # ----------------------------------------------------

        print_decision(
            decision
        )

        # ----------------------------------------------------
        # PRINT SELECTED MOVEMENTS
        # ----------------------------------------------------

        print_selected_movements(
            topology,
            phases,
            decision[
                "selected_phase"
            ]
        )

        # ----------------------------------------------------
        # TEST COMPLETE
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("REAL DEMAND DECISION TEST COMPLETE")
        print("=" * 70)

        print()

    finally:

        # ----------------------------------------------------
        # CLOSE SUMO
        # ----------------------------------------------------

        traci.close()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()