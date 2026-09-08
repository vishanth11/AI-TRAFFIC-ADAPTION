from src.integration.topology import JunctionTopology
from src.integration.conflict_graph import ConflictGraph
from src.integration.phase_generator import PhaseGenerator


def test_topology_to_phases():

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

    # Validate topology
    assert topology.validate() is True

    # Create conflict graph
    graph = ConflictGraph(topology)

    graph.add_conflict(
        "A_to_B",
        "C_to_A"
    )

    graph.add_conflict(
        "A_to_C",
        "B_to_A"
    )

    # Validate conflict graph
    assert graph.validate() is True

    # Generate phases
    generator = PhaseGenerator(graph)

    phases = generator.generate_phases()
    maximal_phases = generator.generate_maximal_phases()

    # There must be valid phases
    assert len(phases) > 0

    # There must be maximal phases
    assert len(maximal_phases) > 0

    # Every maximal phase must be conflict-free
    for phase in maximal_phases:
        assert generator.is_compatible(phase)

    # Every movement must appear in at least one phase
    movement_ids = set(topology.movements.keys())

    phase_movements = set()

    for phase in phases:
        phase_movements.update(phase)

    assert movement_ids.issubset(phase_movements)