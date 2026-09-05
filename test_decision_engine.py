from src.integration.topology import JunctionTopology
from src.integration.conflict_graph import ConflictGraph
from src.integration.phase_generator import PhaseGenerator
from src.integration.decision_engine import DecisionEngine


def test_full_decision_pipeline():

    # -----------------------------
    # 1. Create topology
    # -----------------------------

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

    assert topology.validate() is True

    # -----------------------------
    # 2. Create conflict graph
    # -----------------------------

    graph = ConflictGraph(topology)

    graph.add_conflict(
        "A_to_B",
        "C_to_A"
    )

    graph.add_conflict(
        "A_to_C",
        "B_to_A"
    )

    assert graph.validate() is True

    # -----------------------------
    # 3. Generate valid phases
    # -----------------------------

    generator = PhaseGenerator(graph)

    phases = generator.generate_maximal_phases()

    assert len(phases) > 0

    # -----------------------------
    # 4. Create Decision Engine
    # -----------------------------

    engine = DecisionEngine()

    # -----------------------------
    # 5. Score each phase
    # -----------------------------

    phase_scores = {}

    for index, phase in enumerate(phases, start=1):

        score = engine.calculate_phase_score(
            demand=0.8,
            prediction=0.7,
            priority=0.5,
            fairness=0.6,
            risk=0.2
        )

        phase_scores[f"Phase_{index}"] = score

    # -----------------------------
    # 6. Select best phase
    # -----------------------------

    result = engine.select_best_phase(
        phase_scores
    )

    # -----------------------------
    # 7. Validate result
    # -----------------------------

    assert result["selected_phase"] in phase_scores

    assert 0 <= result["score"] <= 1

    assert len(result["all_scores"]) == len(phases)

    print()
    print("Full Decision Pipeline:")
    print("Topology        : PASS")
    print("Conflict Graph  : PASS")
    print("Phase Generator : PASS")
    print("Decision Engine  : PASS")
    print()
    print("Selected Phase:", result["selected_phase"])
    print("Score:", result["score"])