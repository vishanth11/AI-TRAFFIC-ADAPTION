from types import SimpleNamespace

from src.integration.conflict_graph import ConflictGraph
from src.integration.emergency import EmergencyCorridorManager, EmergencyVehicleState
from src.integration.phase_generator import PhaseGenerator
from src.integration.topology import JunctionTopology


def _topology():
    topology = JunctionTopology("J1")
    topology.add_road("in", lanes=1)
    topology.add_road("out", lanes=1)
    topology.add_road("cross", lanes=1)
    topology.add_movement("m_in_out", "in", "out")
    topology.add_movement("m_cross_out", "cross", "out")
    graph = ConflictGraph(topology)
    graph.add_conflict("m_in_out", "m_cross_out")
    return topology, PhaseGenerator(graph)


def test_emergency_urgency_and_zero_speed_are_safe():
    stopped = EmergencyVehicleState(
        vehicle_id="ambulance_0", vehicle_type="ambulance", current_edge="in",
        current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
        speed_mps=0.0, route=("in", "out"), current_junction="J1",
        next_junction=None, destination="out", emergency=True, confidence=1.0,
        eta_seconds=None,
    )
    approaching = EmergencyVehicleState(
        vehicle_id="ambulance_1", vehicle_type="ambulance", current_edge="in",
        current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
        speed_mps=10.0, route=("in", "out"), current_junction="J1",
        next_junction=None, destination="out", emergency=True, confidence=1.0,
        eta_seconds=5.0,
    )
    assert EmergencyCorridorManager.urgency(stopped) == 0.35
    assert EmergencyCorridorManager.urgency(approaching) > 0.8


def test_emergency_movement_and_candidate_phase_are_topology_derived():
    topology, generator = _topology()
    manager = EmergencyCorridorManager(SimpleNamespace())
    state = EmergencyVehicleState(
        vehicle_id="ambulance_0", vehicle_type="ambulance", current_edge="in",
        current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
        speed_mps=10.0, route=("in", "out"), current_junction="J1",
        next_junction=None, destination="out", emergency=True, confidence=1.0,
        eta_seconds=5.0,
    )
    phases = [
        {"phase_id": "J1_P0", "movements": ["m_in_out"]},
        {"phase_id": "J1_P1", "movements": ["m_cross_out"]},
    ]
    plan = manager.plan_for_junction(state, "J1", topology, phases, generator)
    assert plan.required_movement == "m_in_out"
    assert plan.candidate_phases == ("J1_P0",)
    assert plan.selected_phase == "J1_P0"
    assert "m_cross_out" in plan.conflicting_movements
    assert plan.safe is True


def test_conflicting_phase_is_not_an_emergency_candidate():
    topology, generator = _topology()
    manager = EmergencyCorridorManager(SimpleNamespace())
    state = EmergencyVehicleState(
        vehicle_id="ambulance_0", vehicle_type="ambulance", current_edge="in",
        current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
        speed_mps=10.0, route=("in", "out"), current_junction="J1",
        next_junction=None, destination="out", emergency=True, confidence=1.0,
        eta_seconds=5.0,
    )
    phases = [{"phase_id": "J1_P1", "movements": ["m_cross_out"]}]
    plan = manager.plan_for_junction(state, "J1", topology, phases, generator)
    assert plan.selected_phase is None
    assert plan.safe is False


def test_multiple_emergency_plans_choose_highest_urgency():
    topology, generator = _topology()
    manager = EmergencyCorridorManager(SimpleNamespace())
    states = [
        EmergencyVehicleState(
            vehicle_id="slow", vehicle_type="ambulance", current_edge="in",
            current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
            speed_mps=5.0, route=("in", "out"), current_junction="J1",
            next_junction=None, destination="out", emergency=True, confidence=1.0,
            eta_seconds=10.0,
        ),
        EmergencyVehicleState(
            vehicle_id="fast", vehicle_type="ambulance", current_edge="in",
            current_lane="in_0", current_position=5.0, distance_to_junction=50.0,
            speed_mps=10.0, route=("in", "out"), current_junction="J1",
            next_junction=None, destination="out", emergency=True, confidence=1.0,
            eta_seconds=3.0,
        ),
    ]
    runtimes = {
        "J1": {
            "topology": topology,
            "generator": generator,
            "phases": [{"phase_id": "J1_P0", "movements": ["m_in_out"]}],
        }
    }
    plans = manager.plans(states, {"J1": topology}, runtimes)
    assert plans["J1"].vehicle_id == "fast"


def test_emergency_vehicle_release_is_detected():
    manager = EmergencyCorridorManager(SimpleNamespace())
    manager.active_vehicle_ids = {"ambulance_0"}
    released = manager.release_completed([])
    assert released == {"ambulance_0"}
    assert manager.active_vehicle_ids == set()
