class PhaseGenerator:
    """
    Generates valid traffic signal phases from a ConflictGraph.

    A phase is a group of movements that can receive green
    simultaneously because none of them conflict.
    """

    def __init__(self, conflict_graph):
        self.graph = conflict_graph

    def is_compatible(self, movements):
        """
        Check whether all movements can operate together.
        """

        movements = list(movements)

        for i in range(len(movements)):
            for j in range(i + 1, len(movements)):

                if self.graph.are_conflicting(
                    movements[i],
                    movements[j]
                ):
                    return False

        return True

    def generate_phases(self):
        """
        Generate all non-empty compatible movement groups.

        Only allowed movements are considered.
        """

        allowed_movements = [
            movement.id
            for movement in self.graph.topology.get_allowed_movements()
        ]

        phases = []

        def build_phase(start_index, current_phase):
            if current_phase:
                phases.append(tuple(current_phase))

            for i in range(
                start_index,
                len(allowed_movements)
            ):
                movement = allowed_movements[i]

                if self.is_compatible(
                    current_phase + [movement]
                ):
                    current_phase.append(movement)

                    build_phase(
                        i + 1,
                        current_phase
                    )

                    current_phase.pop()

        build_phase(0, [])

        return phases

    def generate_maximal_phases(self):
        """
        Return only phases that cannot accept another
        compatible movement.

        These are maximal compatible movement groups.
        """

        all_phases = self.generate_phases()

        maximal_phases = []

        for phase in all_phases:
            phase_set = set(phase)

            is_maximal = True

            for other_phase in all_phases:
                other_set = set(other_phase)

                if (
                    phase_set < other_set
                    and phase_set.issubset(other_set)
                ):
                    is_maximal = False
                    break

            if is_maximal:
                maximal_phases.append(phase)

        return maximal_phases


if __name__ == "__main__":

    from topology import JunctionTopology
    from conflict_graph import ConflictGraph

    # -------------------------------------------------
    # Create topology
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Create conflict graph
    # -------------------------------------------------

    graph = ConflictGraph(topology)

    graph.add_conflict(
        "A_to_B",
        "C_to_A"
    )

    graph.add_conflict(
        "A_to_C",
        "B_to_A"
    )

    # -------------------------------------------------
    # Generate phases
    # -------------------------------------------------

    generator = PhaseGenerator(graph)

    phases = generator.generate_phases()

    maximal_phases = generator.generate_maximal_phases()

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("Topology is valid.")
    print("Conflict graph is valid.")
    print()

    print("All valid phases:")

    for number, phase in enumerate(phases, start=1):
        print(f"Phase {number}: {phase}")

    print()

    print("Maximal phases:")

    for number, phase in enumerate(
        maximal_phases,
        start=1
    ):
        print(f"Phase {number}: {phase}")