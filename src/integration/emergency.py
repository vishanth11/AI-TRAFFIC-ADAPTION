"""Topology-adaptive emergency priority planning for the SUMO controller."""

from __future__ import annotations

import math
from dataclasses import dataclass


EMERGENCY_TYPE_NAMES = {
    "ambulance",
    "emergency",
    "emergency_vehicle",
    "fire",
    "firetruck",
    "police",
}


@dataclass(frozen=True)
class EmergencyVehicleState:
    vehicle_id: str
    vehicle_type: str
    current_edge: str | None
    current_lane: str | None
    current_position: float | None
    distance_to_junction: float | None
    speed_mps: float | None
    route: tuple[str, ...]
    current_junction: str | None
    next_junction: str | None
    destination: str | None
    emergency: bool
    confidence: float
    eta_seconds: float | None


@dataclass(frozen=True)
class EmergencyPlan:
    vehicle_id: str
    junction_id: str
    required_movement: str | None
    candidate_phases: tuple[str, ...]
    selected_phase: str | None
    eta_seconds: float | None
    urgency: float
    safe: bool
    current_edge: str | None
    next_edge: str | None
    distance_to_junction: float | None
    vehicle_speed_mps: float | None
    conflicting_movements: tuple[str, ...]


class EmergencyCorridorManager:
    """Create emergency phase plans without directly controlling signals."""

    def __init__(self, traci_connection, max_speed_floor=0.5):
        self.traci = traci_connection
        self.max_speed_floor = float(max_speed_floor)
        self.active_vehicle_ids = set()
        self.released_vehicle_ids = set()

    @staticmethod
    def _safe_call(function, *args, default=None):
        try:
            return function(*args)
        except Exception:
            return default

    @classmethod
    def _is_emergency(cls, vehicle_type, vehicle_class):
        values = {
            str(vehicle_type or "").lower(),
            str(vehicle_class or "").lower(),
        }
        return bool(values.intersection(EMERGENCY_TYPE_NAMES)) or any(
            "emergency" in value or "ambulance" in value or "fire" in value
            for value in values
        )

    def _junction_for_movement(self, movement, topologies):
        for junction_id, topology in topologies.items():
            if movement in topology.movements:
                return junction_id
        return None

    def discover(self, topologies):
        """Discover current emergency vehicles from SUMO type/class data."""
        states = []
        current_ids = set(self.traci.vehicle.getIDList())
        for vehicle_id in current_ids:
            vehicle_type = self._safe_call(
                self.traci.vehicle.getTypeID, vehicle_id, default=""
            )
            vehicle_class = self._safe_call(
                self.traci.vehicle.getVehicleClass, vehicle_id, default=""
            )
            if not self._is_emergency(vehicle_type, vehicle_class):
                continue

            current_edge = self._safe_call(
                self.traci.vehicle.getRoadID, vehicle_id, default=None
            )
            if current_edge and str(current_edge).startswith(":"):
                current_edge = None
            lane_id = self._safe_call(
                self.traci.vehicle.getLaneID, vehicle_id, default=None
            )
            position = self._safe_call(
                self.traci.vehicle.getLanePosition, vehicle_id, default=None
            )
            lane_length = (
                self._safe_call(self.traci.lane.getLength, lane_id, default=None)
                if lane_id else None
            )
            speed = self._safe_call(
                self.traci.vehicle.getSpeed, vehicle_id, default=None
            )
            speed = float(speed) if speed is not None else None
            route = tuple(
                self._safe_call(self.traci.vehicle.getRoute, vehicle_id, default=[])
                or []
            )
            route_index = self._safe_call(
                self.traci.vehicle.getRouteIndex, vehicle_id, default=None
            )
            next_edge = None
            if route_index is not None and route_index + 1 < len(route):
                next_edge = route[route_index + 1]
            if next_edge and str(next_edge).startswith(":"):
                next_edge = None

            current_junction = None
            required_movement = None
            for junction_id, topology in topologies.items():
                for movement_id, movement in topology.movements.items():
                    if movement.from_road == current_edge and movement.to_road == next_edge:
                        current_junction = junction_id
                        required_movement = movement_id
                        break
                if current_junction:
                    break

            distance = None
            if lane_length is not None and position is not None:
                distance = max(0.0, float(lane_length) - float(position))
            eta = None
            if distance is not None and speed is not None and speed >= self.max_speed_floor:
                eta = distance / speed
            next_junction = None
            if required_movement:
                movement = topologies[current_junction].movements[required_movement]
                for junction_id, topology in topologies.items():
                    if any(
                        candidate.from_road == movement.to_road
                        for candidate in topology.movements.values()
                    ):
                        next_junction = junction_id
                        break

            states.append(
                EmergencyVehicleState(
                    vehicle_id=str(vehicle_id),
                    vehicle_type=str(vehicle_type),
                    current_edge=current_edge,
                    current_lane=lane_id,
                    current_position=float(position) if position is not None else None,
                    distance_to_junction=distance,
                    speed_mps=speed,
                    route=route,
                    current_junction=current_junction,
                    next_junction=next_junction,
                    destination=route[-1] if route else None,
                    emergency=True,
                    confidence=1.0,
                    eta_seconds=eta,
                )
            )

        return states

    @staticmethod
    def urgency(state):
        """Normalize urgency from ETA, with safe handling for missing/zero speed."""
        if state.eta_seconds is None:
            return 0.35
        if state.eta_seconds <= 0.0:
            return 1.0
        return max(0.0, min(1.0, math.exp(-state.eta_seconds / 30.0)))

    def plan_for_junction(self, state, junction_id, topology, phases, generator):
        """Build a safe candidate plan for the vehicle's required movement."""
        movement_id = None
        for candidate_id, movement in topology.movements.items():
            if movement.from_road == state.current_edge:
                next_edge = state.route[
                    state.route.index(state.current_edge) + 1
                ] if state.current_edge in state.route and state.route.index(state.current_edge) + 1 < len(state.route) else None
                if movement.to_road == next_edge:
                    movement_id = candidate_id
                    break
        if movement_id is None:
            return EmergencyPlan(
                state.vehicle_id, junction_id, None, (), None,
                state.eta_seconds, self.urgency(state), False,
                state.current_edge, None, state.distance_to_junction,
                state.speed_mps, ()
            )

        candidate_phases = tuple(
            phase["phase_id"]
            for phase in phases
            if movement_id in phase["movements"]
            and generator.is_compatible(phase["movements"])
        )
        selected = candidate_phases[0] if candidate_phases else None
        conflicting = tuple(
            movement_id_2
            for movement_id_2 in topology.movements
            if movement_id_2 != movement_id
            and generator.graph.are_conflicting(movement_id, movement_id_2)
        )
        return EmergencyPlan(
            state.vehicle_id,
            junction_id,
            movement_id,
            candidate_phases,
            selected,
            state.eta_seconds,
            self.urgency(state),
            bool(selected),
            state.current_edge,
            topology.movements[movement_id].to_road,
            state.distance_to_junction,
            state.speed_mps,
            conflicting,
        )

    def plans(self, states, topologies, runtimes):
        """Return the most urgent valid plan for each junction."""
        result = {}
        for junction_id, runtime in runtimes.items():
            candidates = [
                state for state in states
                if state.current_junction == junction_id
            ]
            plans = [
                self.plan_for_junction(
                    state,
                    junction_id,
                    runtime["topology"],
                    runtime["phases"],
                    runtime["generator"],
                )
                for state in candidates
            ]
            plans = [plan for plan in plans if plan.required_movement]
            if plans:
                result[junction_id] = max(
                    plans,
                    key=lambda plan: (plan.urgency, -(plan.eta_seconds or 1e9)),
                )
        return result

    def release_completed(self, states):
        current = {state.vehicle_id for state in states}
        released = self.active_vehicle_ids - current
        self.released_vehicle_ids.update(released)
        self.active_vehicle_ids = current
        return released
