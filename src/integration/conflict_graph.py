class ConflictGraph:
    """
    Represents conflicts between traffic movements.

    Each movement is a node.
    An edge between two movements means:
        those two movements cannot receive green
        at the same time.
    """

    def __init__(self, topology):
        self.topology = topology

        # movement_id -> set of conflicting movement_ids
        self.conflicts = {}

        # Initialize every movement
        for movement_id in topology.movements:
            self.conflicts[movement_id] = set()

    def add_conflict(self, movement_a, movement_b):
        """
        Add a two-way conflict between two movements.
        """

        if movement_a not in self.topology.movements:
            raise ValueError(
                f"Unknown movement: {movement_a}"
            )

        if movement_b not in self.topology.movements:
            raise ValueError(
                f"Unknown movement: {movement_b}"
            )

        if movement_a == movement_b:
            raise ValueError(
                "A movement cannot conflict with itself"
            )

        self.conflicts[movement_a].add(movement_b)
        self.conflicts[movement_b].add(movement_a)

    def get_conflicts(self, movement_id):
        """
        Return all movements that conflict
        with the given movement.
        """

        if movement_id not in self.conflicts:
            raise ValueError(
                f"Unknown movement: {movement_id}"
            )

        return self.conflicts[movement_id]

    def are_conflicting(self, movement_a, movement_b):
        """
        Check whether two movements conflict.
        """

        if movement_a not in self.conflicts:
            raise ValueError(
                f"Unknown movement: {movement_a}"
            )

        if movement_b not in self.conflicts:
            raise ValueError(
                f"Unknown movement: {movement_b}"
            )

        return movement_b in self.conflicts[movement_a]

    def validate(self):
        """
        Validate the conflict graph.
        """

        for movement_id, conflicts in self.conflicts.items():

            for conflicting_movement in conflicts:

                if conflicting_movement not in self.conflicts:
                    raise ValueError(
                        f"Conflict references unknown movement: "
                        f"{conflicting_movement}"
                    )

                # Conflict must be symmetric
                if movement_id not in self.conflicts[
                    conflicting_movement
                ]:
                    raise ValueError(
                        f"Conflict is not symmetric: "
                        f"{movement_id} <-> "
                        f"{conflicting_movement}"
                    )

        return True

    def to_dict(self):
        """
        Convert conflict graph into a dictionary.
        """

        return {
            movement_id: sorted(list(conflicts))
            for movement_id, conflicts in self.conflicts.items()
        }


if __name__ == "__main__":

    # Import our topology module
    from topology import JunctionTopology

    # -------------------------------------------------
    # Create example topology
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

    # Example conflict relationships.
    #
    # These are explicitly supplied for the test.
    # We are NOT assuming that road names determine
    # physical conflicts.

    graph.add_conflict(
        "A_to_B",
        "C_to_A"
    )

    graph.add_conflict(
        "A_to_C",
        "B_to_A"
    )

    # -------------------------------------------------
    # Validate
    # -------------------------------------------------

    topology.validate()
    graph.validate()

    print("Topology is valid.")
    print("Conflict graph is valid.")
    print()
    print(graph.to_dict())