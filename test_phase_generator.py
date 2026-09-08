import time

from src.integration.conflict_graph import ConflictGraph
from src.integration.phase_generator import PhaseGenerator
from src.integration.topology import JunctionTopology


def _small_graph(reverse=False):
    topology = JunctionTopology("J1")
    for road_id in ("A", "B", "C"):
        topology.add_road(road_id)

    movements = [
        ("A_to_B", "A", "B"),
        ("A_to_C", "A", "C"),
        ("B_to_A", "B", "A"),
        ("C_to_A", "C", "A"),
    ]
    if reverse:
        movements.reverse()

    for movement_id, from_road, to_road in movements:
        topology.add_movement(movement_id, from_road, to_road)

    graph = ConflictGraph(topology)
    graph.add_conflict("A_to_B", "C_to_A")
    graph.add_conflict("A_to_C", "B_to_A")
    return graph


def _assert_phases_are_maximal(generator, phases):
    allowed = {
        movement.id
        for movement in generator.graph.topology.get_allowed_movements()
    }

    for phase in phases:
        assert generator.is_compatible(phase)
        phase_movements = set(phase)
        for candidate in allowed - phase_movements:
            assert any(
                generator.graph.are_conflicting(candidate, selected)
                for selected in phase
            )


def test_small_existing_graph_produces_maximal_phases():
    generator = PhaseGenerator(_small_graph())

    phases = generator.generate_maximal_phases()

    assert phases
    _assert_phases_are_maximal(generator, phases)


def test_maximal_phases_are_conflict_free():
    generator = PhaseGenerator(_small_graph())

    for phase in generator.generate_maximal_phases():
        assert generator.is_compatible(phase)


def test_maximal_phases_are_deduplicated():
    topology = JunctionTopology("J1")
    topology.add_road("A")
    topology.add_road("B")
    for index in range(4):
        topology.add_movement(f"m{index}", "A", "B")

    generator = PhaseGenerator(ConflictGraph(topology))

    assert generator.generate_maximal_phases() == [
        ("m0", "m1", "m2", "m3")
    ]


def test_maximal_phase_limit_is_respected():
    topology = JunctionTopology("J1")
    topology.add_road("A")
    topology.add_road("B")
    movement_ids = [f"m{index}" for index in range(5)]
    for movement_id in movement_ids:
        topology.add_movement(movement_id, "A", "B")

    graph = ConflictGraph(topology)
    for index, movement_a in enumerate(movement_ids):
        for movement_b in movement_ids[index + 1:]:
            graph.add_conflict(movement_a, movement_b)

    generator = PhaseGenerator(graph, max_phases=3)

    assert generator.generate_maximal_phases() == [("m0",), ("m1",), ("m2",)]


def test_maximal_phase_output_is_deterministic():
    phases = PhaseGenerator(_small_graph()).generate_maximal_phases()
    reversed_phases = PhaseGenerator(_small_graph(reverse=True)).generate_maximal_phases()

    assert phases == reversed_phases
    assert all(tuple(sorted(phase)) == phase for phase in phases)


def test_32_movement_graph_completes_quickly():
    topology = JunctionTopology("large")
    topology.add_road("A")
    topology.add_road("B")
    for index in range(32):
        topology.add_movement(f"m{index:02d}", "A", "B")

    generator = PhaseGenerator(ConflictGraph(topology))
    started = time.perf_counter()
    phases = generator.generate_maximal_phases()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert phases == [tuple(f"m{index:02d}" for index in range(32))]
    _assert_phases_are_maximal(generator, phases)


def test_max_phases_requires_positive_integer():
    graph = _small_graph()

    for invalid_limit in (0, -1, 1.5, True):
        try:
            PhaseGenerator(graph, max_phases=invalid_limit)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid max_phases={invalid_limit!r}")
