class Road:
    def __init__(self, road_id, lanes=1):
        if lanes < 1:
            raise ValueError("lanes must be >= 1")

        self.id = str(road_id)
        self.lanes = lanes

    def to_dict(self):
        return {
            "road_id": self.id,
            "lanes": self.lanes
        }


class Movement:
    def __init__(
        self,
        movement_id,
        from_road,
        to_road,
        lanes=1,
        allowed=True
    ):
        if lanes < 1:
            raise ValueError("lanes must be >= 1")

        self.id = str(movement_id)
        self.from_road = str(from_road)
        self.to_road = str(to_road)
        self.lanes = lanes
        self.allowed = bool(allowed)

    @property
    def movement_id(self):
        """Backward-compatible alias for the canonical ``id`` field."""
        return self.id

    def to_dict(self):
        return {
            "movement_id": self.id,
            "from_road": self.from_road,
            "to_road": self.to_road,
            "lanes": self.lanes,
            "allowed": self.allowed
        }


class JunctionTopology:
    def __init__(self, junction_id):
        self.id = str(junction_id)
        self.roads = {}
        self.movements = {}

    def add_road(self, road_id, lanes=1):
        road = Road(road_id, lanes)
        self.roads[road.id] = road

    def add_movement(
        self,
        movement_id,
        from_road,
        to_road,
        lanes=1,
        allowed=True
    ):
        if str(from_road) not in self.roads:
            raise ValueError(
                f"Unknown from_road: {from_road}"
            )

        if str(to_road) not in self.roads:
            raise ValueError(
                f"Unknown to_road: {to_road}"
            )

        if str(from_road) == str(to_road):
            raise ValueError(
                "from_road and to_road cannot be the same"
            )

        movement = Movement(
            movement_id,
            from_road,
            to_road,
            lanes,
            allowed
        )

        self.movements[movement.id] = movement

    def get_allowed_movements(self):
        return [
            movement
            for movement in self.movements.values()
            if movement.allowed
        ]

    def validate(self):
        for movement in self.movements.values():

            if movement.from_road not in self.roads:
                raise ValueError(
                    f"Movement {movement.id} references "
                    f"unknown road {movement.from_road}"
                )

            if movement.to_road not in self.roads:
                raise ValueError(
                    f"Movement {movement.id} references "
                    f"unknown road {movement.to_road}"
                )

        return True

    def to_dict(self):
        return {
            "junction_id": self.id,
            "roads": {
                road_id: road.to_dict()
                for road_id, road in self.roads.items()
            },
            "movements": {
                movement_id: movement.to_dict()
                for movement_id, movement in self.movements.items()
            }
        }


if __name__ == "__main__":

    topology = JunctionTopology("J1")

    # Three roads with different lane counts
    topology.add_road("A", lanes=2)
    topology.add_road("B", lanes=2)
    topology.add_road("C", lanes=1)

    # Allowed movements
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

    # Validate topology
    topology.validate()

    print("Topology is valid.")
    print()
    print(topology.to_dict())