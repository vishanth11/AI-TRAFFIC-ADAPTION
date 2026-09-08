import traci

from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.sumo_phase_adapter import SUMOPhaseAdapter
from integration.sumo_conflict_adapter import SUMOConflictAdapter
from integration.phase_generator import PhaseGenerator


SUMO_CMD = [
    "sumo",
    "-c",
    "simulation/configs/corridor.sumocfg"
]


def print_generated_phases(junction_id, phases):

    print()
    print("=" * 70)
    print(f"GENERATED PHASES: {junction_id}")
    print("=" * 70)

    print()
    print(f"Total generated phases: {len(phases)}")

    for index, phase in enumerate(phases):

        print()
        print(f"Phase {index}")
        print(f"  {phase}")


try:

    traci.start(SUMO_CMD)

    # ------------------------------------------------------------
    # Create adapters
    # ------------------------------------------------------------

    topology_adapter = SUMOTopologyAdapter()

    phase_adapter = SUMOPhaseAdapter()

    conflict_adapter = SUMOConflictAdapter(
        phase_adapter
    )

    # ------------------------------------------------------------
    # Discover topology
    # ------------------------------------------------------------

    topologies = (
        topology_adapter
        .discover_all_junctions()
    )

    # ------------------------------------------------------------
    # Test A0, B0 and C0
    # ------------------------------------------------------------

    for junction_id in ["A0", "B0", "C0"]:

        topology = topologies[junction_id]

        # --------------------------------------------------------
        # Build conflict graph
        # --------------------------------------------------------

        conflict_graph = (
            conflict_adapter
            .build_conflict_graph(
                junction_id,
                topology,
                topology_adapter.sumo_metadata
            )
        )

        # --------------------------------------------------------
        # Generate maximal phases
        # --------------------------------------------------------

        generator = PhaseGenerator(
            conflict_graph
        )

        phases = (
            generator
            .generate_maximal_phases()
        )

        # --------------------------------------------------------
        # Print results
        # --------------------------------------------------------

        print_generated_phases(
            junction_id,
            phases
        )


finally:

    traci.close()