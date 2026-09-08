import traci

from integration.conflict_graph import ConflictGraph
from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.sumo_phase_adapter import SUMOPhaseAdapter


class SUMOConflictAdapter:
    """
    Converts SUMO signal-phase information into the project's
    generic ConflictGraph.

    The JunctionTopology already contains all movements.

    SUMO phases are used as the initial compatibility source:

        Same green phase
            -> compatible

        Never green together
            -> conflict
    """

    def __init__(self, phase_adapter):
        self.phase_adapter = phase_adapter

    # ============================================================
    # BUILD CONFLICT GRAPH
    # ============================================================

    def build_conflict_graph(
        self,
        junction_id,
        topology,
        movement_metadata
    ):
        """
        Build a ConflictGraph for one SUMO junction.
        """

        # ConflictGraph requires JunctionTopology.
        #
        # The topology already contains all movements,
        # so we do NOT call graph.add_movement().
        graph = ConflictGraph(topology)

        # --------------------------------------------------------
        # Get movements belonging to this junction
        # --------------------------------------------------------

        junction_movements = []

        for movement_id, metadata in movement_metadata.items():

            if metadata["junction_id"] == junction_id:

                junction_movements.append(
                    movement_id
                )

        # --------------------------------------------------------
        # Get valid green phases
        # --------------------------------------------------------

        phases = (
            self.phase_adapter
            .build_phase_representation(
                junction_id,
                movement_metadata
            )
        )

        # --------------------------------------------------------
        # Find compatible movement pairs
        # --------------------------------------------------------

        compatible_pairs = set()

        for phase in phases:

            movements = phase["movements"]

            for i in range(len(movements)):

                for j in range(i + 1, len(movements)):

                    movement_a = movements[i]
                    movement_b = movements[j]

                    pair = tuple(
                        sorted(
                            (
                                movement_a,
                                movement_b
                            )
                        )
                    )

                    compatible_pairs.add(pair)

        # --------------------------------------------------------
        # Add conflicts
        # --------------------------------------------------------

        for i in range(len(junction_movements)):

            for j in range(i + 1, len(junction_movements)):

                movement_a = junction_movements[i]
                movement_b = junction_movements[j]

                pair = tuple(
                    sorted(
                        (
                            movement_a,
                            movement_b
                        )
                    )
                )

                # If these movements are never allowed
                # to be green together by the SUMO program,
                # treat them as conflicting.
                if pair not in compatible_pairs:

                    graph.add_conflict(
                        movement_a,
                        movement_b
                    )

        # --------------------------------------------------------
        # Validate graph
        # --------------------------------------------------------

        graph.validate()

        return graph


# ================================================================
# PRINT CONFLICT GRAPH
# ================================================================

def print_conflict_graph(
    graph,
    topology,
    junction_id
):
    """
    Print the conflict graph.

    ConflictGraph does not expose a 'nodes' attribute.
    The movement IDs come from topology.movements.
    """

    print()
    print("=" * 70)
    print(
        f"CONFLICT GRAPH: {junction_id}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Movements
    # ------------------------------------------------------------

    print()
    print("MOVEMENTS:")

    movement_ids = list(
        topology.movements.keys()
    )

    for movement_id in movement_ids:

        movement = topology.movements[
            movement_id
        ]

        print(
            f"  {movement_id}: "
            f"{movement.from_road} -> "
            f"{movement.to_road}"
        )

    # ------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------

    print()
    print("CONFLICTS:")

    printed_pairs = set()

    for movement_id in movement_ids:

        conflicts = graph.get_conflicts(
            movement_id
        )

        for conflict_id in conflicts:

            pair = tuple(
                sorted(
                    (
                        movement_id,
                        conflict_id
                    )
                )
            )

            if pair in printed_pairs:
                continue

            printed_pairs.add(pair)

            print(
                f"  {pair[0]} <-> {pair[1]}"
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()

    print(
        f"Total movements: "
        f"{len(movement_ids)}"
    )

    print(
        f"Total conflict pairs: "
        f"{len(printed_pairs)}"
    )


# ================================================================
# MAIN TEST
# ================================================================

if __name__ == "__main__":

    SUMO_CMD = [
        "sumo",
        "-c",
        "simulation/configs/corridor.sumocfg"
    ]

    try:

        # --------------------------------------------------------
        # Start SUMO
        # --------------------------------------------------------

        traci.start(SUMO_CMD)

        # --------------------------------------------------------
        # Create adapters
        # --------------------------------------------------------

        topology_adapter = (
            SUMOTopologyAdapter()
        )

        phase_adapter = (
            SUMOPhaseAdapter()
        )

        conflict_adapter = (
            SUMOConflictAdapter(
                phase_adapter
            )
        )

        # --------------------------------------------------------
        # Discover SUMO topology
        # --------------------------------------------------------

        topologies = (
            topology_adapter
            .discover_all_junctions()
        )

        print()
        print(
            "Discovered traffic lights:"
        )

        for junction_id in topologies:

            print(
                f"  {junction_id}"
            )

        # --------------------------------------------------------
        # Build conflict graph
        # --------------------------------------------------------

        for junction_id in [
            "A0",
            "B0",
            "C0"
        ]:

            topology = topologies[
                junction_id
            ]

            graph = (
                conflict_adapter
                .build_conflict_graph(
                    junction_id,
                    topology,
                    topology_adapter.sumo_metadata
                )
            )

            print_conflict_graph(
                graph,
                topology,
                junction_id
            )

    finally:

        traci.close()