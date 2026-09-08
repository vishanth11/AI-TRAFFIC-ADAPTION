class PhaseGenerator:
    """
    Generates valid traffic signal phases from a ConflictGraph.

    A phase is a group of movements that can receive green
    simultaneously because none of them conflict.
    """

    def __init__(self, conflict_graph, max_phases=64):
        if not isinstance(max_phases, int) or isinstance(max_phases, bool) or max_phases < 1:
            raise ValueError("max_phases must be a positive integer")

        self.graph = conflict_graph
        self.max_phases = max_phases

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
        """Return bounded, greedily constructed maximal phases."""
        allowed_movements = sorted(
            (movement.id for movement in self.graph.topology.get_allowed_movements()),
            key=str,
        )
        maximal_phases = []
        seen_phases = set()

        for seed in allowed_movements:
            phase = [seed]

            for candidate in allowed_movements:
                if candidate == seed:
                    continue

                if self.is_compatible(phase + [candidate]):
                    phase.append(candidate)

            canonical_phase = tuple(sorted(phase, key=str))
            if canonical_phase in seen_phases:
                continue

            seen_phases.add(canonical_phase)
            maximal_phases.append(canonical_phase)
            if len(maximal_phases) >= self.max_phases:
                break

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