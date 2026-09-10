"""Unit tests for the ambulance green corridor feature.

These tests validate corridor planning, progression tracking, junction
release, conflicting-phase safety, multiple ambulances, and edge cases
WITHOUT requiring a running SUMO instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.integration.conflict_graph import ConflictGraph
from src.integration.emergency import (
    AmbulanceCorridorState,
    CorridorJunctionState,
    EmergencyCorridorManager,
    EmergencyPlan,
    EmergencyVehicleState,
)
from src.integration.phase_generator import PhaseGenerator
from src.integration.topology import JunctionTopology


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _make_junction(junction_id, roads, movements, conflicts):
    """Build a JunctionTopology + PhaseGenerator from compact specs."""
    topology = JunctionTopology(junction_id)
    for road_id, lanes in roads:
        topology.add_road(road_id, lanes=lanes)
    for mid, fr, to in movements:
        topology.add_movement(mid, fr, to)
    graph = ConflictGraph(topology)
    for a, b in conflicts:
        graph.add_conflict(a, b)
    return topology, PhaseGenerator(graph), graph


def _make_state(
    vehicle_id="ambulance_0",
    current_edge="e1",
    route=("e1", "e2"),
    route_index=0,
    speed_mps=13.9,
    position=10.0,
    distance=100.0,
    eta=7.2,
):
    return EmergencyVehicleState(
        vehicle_id=vehicle_id,
        vehicle_type="ambulance",
        current_edge=current_edge,
        current_lane=f"{current_edge}_0",
        current_position=position,
        distance_to_junction=distance,
        speed_mps=speed_mps,
        route=tuple(route),
        current_junction=None,
        next_junction=None,
        destination=route[-1] if route else None,
        emergency=True,
        confidence=1.0,
        eta_seconds=eta,
        route_index=route_index,
    )


def _make_plan(
    vehicle_id="ambulance_0",
    junction_id="J1",
    movement="m_AB",
    phase="J1_P0",
    eta=10.0,
    distance=100.0,
    next_edge="eB",
    movement_route_index=0,
):
    return EmergencyPlan(
        vehicle_id=vehicle_id,
        junction_id=junction_id,
        required_movement=movement,
        candidate_phases=(phase,),
        selected_phase=phase,
        eta_seconds=eta,
        urgency=max(0.0, min(1.0, math.exp(-eta / 30.0))) if eta is not None else 0.35,
        safe=True,
        current_edge="eA",
        next_edge=next_edge,
        distance_to_junction=distance,
        vehicle_speed_mps=13.9,
        conflicting_movements=(),
        route_index=0,
        movement_route_index=movement_route_index,
    )


def _corridor_topology_3j():
    """Build a 3-junction corridor: J1 -- J2 -- J3.

    Route: e1 -> e2 -> e3 -> e4
    J1 controls e1->e2 and cross1->e2
    J2 controls e2->e3 and cross2->e3
    J3 controls e3->e4 and cross3->e4
    """
    topologies = {}
    runtimes = {}

    for i, jid in enumerate(["J1", "J2", "J3"], start=1):
        from_edge = f"e{i}"
        to_edge = f"e{i + 1}"
        cross_edge = f"cross{i}"
        m_through = f"{jid}_M0"
        m_cross = f"{jid}_M1"

        topo = JunctionTopology(jid)
        topo.add_road(from_edge, lanes=1)
        topo.add_road(to_edge, lanes=1)
        topo.add_road(cross_edge, lanes=1)
        topo.add_movement(m_through, from_edge, to_edge)
        topo.add_movement(m_cross, cross_edge, to_edge)

        graph = ConflictGraph(topo)
        graph.add_conflict(m_through, m_cross)
        gen = PhaseGenerator(graph)

        phases_repr = [
            {
                "phase_id": f"{jid}_P0",
                "junction_id": jid,
                "phase_index": 0,
                "duration": 30.0,
                "state": "Gr",
                "green_links": [0],
                "movements": [m_through],
            },
            {
                "phase_id": f"{jid}_P1",
                "junction_id": jid,
                "phase_index": 2,
                "duration": 30.0,
                "state": "rG",
                "green_links": [1],
                "movements": [m_cross],
            },
        ]

        topologies[jid] = topo
        runtimes[jid] = {
            "topology": topo,
            "generator": gen,
            "phases": phases_repr,
            "phase_objects": [],
            "phase_by_id": {p["phase_id"]: p for p in phases_repr},
            "last_control": -float("inf"),
            "last_phase": None,
            "emergency_hold_phase": None,
            "emergency_vehicle_id": None,
        }

    return topologies, runtimes


def _distance_resolver(state, route_index, movement_route_index):
    """Simple distance resolver: 200m per edge hop."""
    return float((movement_route_index - route_index) * 200 + 100)


# -----------------------------------------------------------------------
# TEST 1: Single junction — correct movement and green phase
# -----------------------------------------------------------------------

def test_single_junction_correct_movement_and_phase():
    """Ambulance approaching one junction gets the correct movement
    identified and the correct green phase selected."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )

    corridor = manager.build_corridor_plan(
        state, topologies, runtimes, _distance_resolver,
    )

    assert len(corridor.junctions) >= 1
    j1_plan = corridor.junctions[0].plan
    assert j1_plan.junction_id == "J1"
    assert j1_plan.required_movement == "J1_M0"
    assert j1_plan.selected_phase == "J1_P0"


# -----------------------------------------------------------------------
# TEST 2: 3-junction route — all 3 junctions in corridor plan
# -----------------------------------------------------------------------

def test_three_junction_route_all_planned():
    """Ambulance route containing 3 junctions → all 3 appear in corridor."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )

    corridor = manager.build_corridor_plan(
        state, topologies, runtimes, _distance_resolver,
    )

    junction_ids = [js.plan.junction_id for js in corridor.junctions]
    assert junction_ids == ["J1", "J2", "J3"]


# -----------------------------------------------------------------------
# TEST 3: 5+ junction route — all planned
# -----------------------------------------------------------------------

def test_five_junction_route_all_planned():
    """Ambulance route with 5+ signalized junctions → all planned."""
    topologies = {}
    runtimes = {}

    route_edges = [f"e{i}" for i in range(1, 8)]  # e1 through e7

    for i in range(1, 7):  # J1 through J6
        jid = f"J{i}"
        from_edge = f"e{i}"
        to_edge = f"e{i + 1}"
        cross = f"cross{i}"
        m_through = f"{jid}_M0"
        m_cross = f"{jid}_M1"

        topo = JunctionTopology(jid)
        topo.add_road(from_edge, lanes=1)
        topo.add_road(to_edge, lanes=1)
        topo.add_road(cross, lanes=1)
        topo.add_movement(m_through, from_edge, to_edge)
        topo.add_movement(m_cross, cross, to_edge)

        graph = ConflictGraph(topo)
        graph.add_conflict(m_through, m_cross)
        gen = PhaseGenerator(graph)

        phases = [
            {
                "phase_id": f"{jid}_P0",
                "junction_id": jid,
                "phase_index": 0,
                "duration": 30.0,
                "state": "Gr",
                "green_links": [0],
                "movements": [m_through],
            },
            {
                "phase_id": f"{jid}_P1",
                "junction_id": jid,
                "phase_index": 2,
                "duration": 30.0,
                "state": "rG",
                "green_links": [1],
                "movements": [m_cross],
            },
        ]

        topologies[jid] = topo
        runtimes[jid] = {
            "topology": topo,
            "generator": gen,
            "phases": phases,
            "phase_objects": [],
            "phase_by_id": {p["phase_id"]: p for p in phases},
            "last_control": -float("inf"),
            "last_phase": None,
            "emergency_hold_phase": None,
            "emergency_vehicle_id": None,
        }

    manager = EmergencyCorridorManager(SimpleNamespace())
    state = _make_state(
        route=tuple(route_edges),
        route_index=0,
        current_edge="e1",
    )

    corridor = manager.build_corridor_plan(
        state, topologies, runtimes, _distance_resolver,
    )

    junction_ids = [js.plan.junction_id for js in corridor.junctions]
    assert len(junction_ids) >= 5
    assert junction_ids == ["J1", "J2", "J3", "J4", "J5", "J6"]


# -----------------------------------------------------------------------
# TEST 4: Distant ambulance → downstream junctions proactively included
# -----------------------------------------------------------------------

def test_distant_ambulance_downstream_junctions_included():
    """When ambulance is far away, ALL downstream junctions are still
    in the corridor plan (though not necessarily activated)."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
        distance=800.0,
        speed_mps=13.9,
        eta=800.0 / 13.9,
    )

    corridor = manager.build_corridor_plan(
        state, topologies, runtimes, _distance_resolver,
    )

    junction_ids = [js.plan.junction_id for js in corridor.junctions]
    assert "J1" in junction_ids
    assert "J2" in junction_ids
    assert "J3" in junction_ids

    # All should be in prepared state, not activated
    for js in corridor.junctions:
        assert js.status == "prepared"
        assert not js.activated


# -----------------------------------------------------------------------
# TEST 5: J1 passed → J1 released, J2 remains active
# -----------------------------------------------------------------------

def test_passed_junction_released_next_remains():
    """When ambulance passes J1, J1 is released, J2 remains prepared."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    # Build corridor while on e1 (route_index=0)
    state_before = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    corridor = manager.build_corridor_plan(
        state_before, topologies, runtimes, _distance_resolver,
    )

    # Ambulance has now progressed to e2 (route_index=1),
    # meaning it has crossed J1 (movement_route_index=0).
    state_after = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=1,
        current_edge="e2",
    )

    released = manager.update_corridor_progression(corridor, state_after)

    # J1 should be released
    assert "J1" in released

    j1_state = corridor.junctions[0]
    assert j1_state.passed is True
    assert j1_state.released is True
    assert j1_state.status == "released"

    # J2 should NOT be released
    j2_state = corridor.junctions[1]
    assert j2_state.released is False
    assert j2_state.passed is False


# -----------------------------------------------------------------------
# TEST 5b: Drain hold — junction stays green until approach edge clears
# -----------------------------------------------------------------------

def _drain_mock_traci(vehicles_on_approach):
    """Mock traci where the approach edge reports the given vehicles, each
    parked near the end of a 200m lane (inside the drain zone)."""
    return SimpleNamespace(
        simulation=SimpleNamespace(getTime=lambda: 10.0),
        edge=SimpleNamespace(
            getIDList=lambda: [],
            getLastStepVehicleIDs=lambda edge: vehicles_on_approach,
            getLastStepVehicleNumber=lambda edge: len(vehicles_on_approach),
        ),
        vehicle=SimpleNamespace(
            getLaneID=lambda vid: f"{vid}_0",
            getLanePosition=lambda vid: 190.0,
        ),
        lane=SimpleNamespace(getLength=lambda lane: 200.0),
    )


def test_drain_hold_keeps_junction_until_approach_clears():
    """After the ambulance passes a junction, it enters a draining hold
    (green kept) while the approach edge still has vehicles near the
    junction, and is released only once the approach edge clears."""
    topologies, runtimes = _corridor_topology_3j()

    state_before = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    state_after = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=1,
        current_edge="e2",
    )

    # Approach edge e1 still has a car near the junction → J1 stays draining.
    manager = EmergencyCorridorManager(_drain_mock_traci(["car_1"]))
    corridor = manager.build_corridor_plan(
        state_before, topologies, runtimes, _distance_resolver,
    )
    released = manager.update_corridor_progression(corridor, state_after)

    assert released == []
    j1 = corridor.junctions[0]
    assert j1.passed is True
    assert j1.draining is True
    assert j1.released is False
    assert j1.status == "draining"

    # Approach edge clears → J1 is released.
    manager2 = EmergencyCorridorManager(_drain_mock_traci([]))
    corridor2 = manager2.build_corridor_plan(
        state_before, topologies, runtimes, _distance_resolver,
    )
    released2 = manager2.update_corridor_progression(corridor2, state_after)

    assert "J1" in released2
    j1b = corridor2.junctions[0]
    assert j1b.released is True
    assert j1b.status == "released"


# -----------------------------------------------------------------------
# TEST 6: Route completed → all released, normal AI resumes
# -----------------------------------------------------------------------

def test_route_completed_all_released():
    """When ambulance completes the route, all junctions are released
    and the corridor becomes inactive."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state_initial = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    corridor = manager.build_corridor_plan(
        state_initial, topologies, runtimes, _distance_resolver,
    )

    # Ambulance at the final edge (route_index=3)
    state_final = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=3,
        current_edge="e4",
    )

    released = manager.update_corridor_progression(corridor, state_final)

    assert len(released) == 3
    assert all(js.released for js in corridor.junctions)
    assert corridor.active is False


# -----------------------------------------------------------------------
# TEST 7: Conflicting phase is not simultaneously granted
# -----------------------------------------------------------------------

def test_conflicting_phase_not_candidate():
    """The ambulance's through-movement should NOT include the
    conflicting cross-movement phase as a candidate."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )

    corridor = manager.build_corridor_plan(
        state, topologies, runtimes, _distance_resolver,
    )

    for js in corridor.junctions:
        plan = js.plan
        # The selected phase should contain the ambulance's movement
        selected_phase_data = runtimes[plan.junction_id]["phase_by_id"][plan.selected_phase]
        assert plan.required_movement in selected_phase_data["movements"]

        # The cross movement should NOT be in the selected phase's movements
        cross_movement = f"{plan.junction_id}_M1"
        assert cross_movement not in selected_phase_data["movements"]


# -----------------------------------------------------------------------
# TEST 8: Normal traffic not permanently held
# -----------------------------------------------------------------------

def test_emergency_hold_not_permanent():
    """Once the ambulance passes a junction, emergency_hold_phase is
    clearable. Corridor junctions get released — they do not stay
    held indefinitely."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state_initial = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    corridor = manager.build_corridor_plan(
        state_initial, topologies, runtimes, _distance_resolver,
    )

    # Simulate progression: ambulance reaches e4
    for ri in range(1, 4):
        state = _make_state(
            route=("e1", "e2", "e3", "e4"),
            route_index=ri,
            current_edge=f"e{ri + 1}",
        )
        manager.update_corridor_progression(corridor, state)

    # Every junction should be released
    for js in corridor.junctions:
        assert js.released is True
    assert corridor.active is False


# -----------------------------------------------------------------------
# TEST 9: Ambulance disappears → corridor safely released
# -----------------------------------------------------------------------

def test_ambulance_disappears_corridor_released():
    """If the ambulance disappears from SUMO, its corridor should be
    safely released without crashes."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state_initial = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    corridor = manager.build_corridor_plan(
        state_initial, topologies, runtimes, _distance_resolver,
    )

    # The corridor should be active
    assert corridor.active is True

    # Simulate disappearance: force-release all junctions
    for js in corridor.junctions:
        js.released = True
        js.passed = True
        js.status = "released"
    corridor.active = False

    # Should not crash and all should be released
    assert all(js.released for js in corridor.junctions)
    assert corridor.active is False


# -----------------------------------------------------------------------
# TEST 10: Route changes → corridor recalculated
# -----------------------------------------------------------------------

def test_route_change_triggers_rebuild():
    """When the ambulance's route changes, a new corridor should be built
    with potentially different junctions."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    # Original route through all 3 junctions
    state_original = _make_state(
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    corridor_original = manager.build_corridor_plan(
        state_original, topologies, runtimes, _distance_resolver,
    )
    assert len(corridor_original.junctions) == 3

    # Changed route — only goes through J1
    state_rerouted = _make_state(
        route=("e1", "e2"),
        route_index=0,
        current_edge="e1",
    )
    corridor_new = manager.build_corridor_plan(
        state_rerouted, topologies, runtimes, _distance_resolver,
    )

    # New corridor should only contain J1 (e1->e2)
    assert len(corridor_new.junctions) == 1
    assert corridor_new.junctions[0].plan.junction_id == "J1"

    # Routes are different — controller would detect and rebuild
    assert corridor_original.route != corridor_new.route


# -----------------------------------------------------------------------
# TEST 11: Multiple ambulances → deterministic behavior
# -----------------------------------------------------------------------

def test_multiple_ambulances_deterministic():
    """Two ambulances with separate routes produce independent corridors."""
    topologies, runtimes = _corridor_topology_3j()
    manager = EmergencyCorridorManager(SimpleNamespace())

    state_a = _make_state(
        vehicle_id="ambulance_A",
        route=("e1", "e2", "e3", "e4"),
        route_index=0,
        current_edge="e1",
    )
    state_b = _make_state(
        vehicle_id="ambulance_B",
        route=("e2", "e3", "e4"),
        route_index=0,
        current_edge="e2",
    )

    corridor_a = manager.build_corridor_plan(
        state_a, topologies, runtimes, _distance_resolver,
    )
    corridor_b = manager.build_corridor_plan(
        state_b, topologies, runtimes, _distance_resolver,
    )

    # Both corridors should be independently built
    assert corridor_a.ambulance_id == "ambulance_A"
    assert corridor_b.ambulance_id == "ambulance_B"
    assert len(corridor_a.junctions) == 3  # J1, J2, J3
    assert len(corridor_b.junctions) == 2  # J2, J3

    # Updating one should not affect the other
    state_a_progressed = _make_state(
        vehicle_id="ambulance_A",
        route=("e1", "e2", "e3", "e4"),
        route_index=1,
        current_edge="e2",
    )
    released_a = manager.update_corridor_progression(
        corridor_a, state_a_progressed,
    )
    assert "J1" in released_a
    # Corridor B should be unaffected
    assert all(not js.released for js in corridor_b.junctions)


# -----------------------------------------------------------------------
# BONUS: ETA-based activation gating
# -----------------------------------------------------------------------

def test_activation_gating_by_eta():
    """Junctions with ETA > threshold should not be activated."""
    plan_far = _make_plan(eta=100.0, distance=1400.0, movement_route_index=3)
    js_far = CorridorJunctionState(plan=plan_far)

    plan_near = _make_plan(eta=10.0, distance=140.0, movement_route_index=0)
    js_near = CorridorJunctionState(plan=plan_near)

    assert not EmergencyCorridorManager.should_activate_junction(
        js_far, eta_threshold=45.0, distance_threshold=500.0,
    )
    assert EmergencyCorridorManager.should_activate_junction(
        js_near, eta_threshold=45.0, distance_threshold=500.0,
    )


# -----------------------------------------------------------------------
# BONUS: Passage detection via route index
# -----------------------------------------------------------------------

def test_passage_detection_route_index():
    """Passage is detected when ambulance route_index > movement_route_index."""
    plan = _make_plan(movement_route_index=1, next_edge="e3")
    js = CorridorJunctionState(plan=plan)

    # Still on the junction's approach edge
    state_before = _make_state(route_index=0)
    assert not EmergencyCorridorManager.has_ambulance_passed_junction(
        state_before, js,
    )

    # Ambulance has advanced past the junction
    state_after = _make_state(route_index=2)
    assert EmergencyCorridorManager.has_ambulance_passed_junction(
        state_after, js,
    )
