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

# SUMO TraCI vehicle-signal bits.
# Bit 3 = rear brake-light dots; bit 11 = blue emergency light.
EMERGENCY_REAR_MARKER_SIGNAL = 1 << 3
EMERGENCY_BLUE_SIGNAL = 1 << 11
EMERGENCY_VISUAL_SIGNALS = EMERGENCY_REAR_MARKER_SIGNAL | EMERGENCY_BLUE_SIGNAL

# Green-extension (drain) hold: after the ambulance passes a junction, keep
# its green phase until the approach edge has drained, so the queue behind the
# ambulance clears instead of being trapped mid-junction when the normal AI
# switches phases. DRAIN_MAX_SECONDS is a safety cap so a stuck car can never
# hold a junction green indefinitely.
#
# The drain condition is distance-based, not occupancy-based: a long approach
# edge can hold many queued cars yet report low occupancy, so we instead check
# that no vehicle sits within DRAIN_ZONE_METERS of the junction (the zone that
# would be trapped by a phase switch) and that the junction's internal
# connectors are empty.
DRAIN_ZONE_METERS = 120.0
DRAIN_MAX_SECONDS = 20.0


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
    route_index: int | None = None


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
    route_index: int | None = None
    movement_route_index: int | None = None
    # The approach edge the ambulance uses to enter this junction. Used for
    # the green-extension drain hold after the ambulance passes.
    from_edge: str | None = None


@dataclass
class CorridorJunctionState:
    """Route-wide priority state for one upcoming signalized junction."""

    plan: EmergencyPlan
    status: str = "prepared"
    activated: bool = False
    passed: bool = False
    released: bool = False
    # Green-extension drain hold: True while the ambulance has passed but the
    # approach edge is still draining. The junction keeps its green phase
    # until the queue clears, then is released to the normal AI.
    draining: bool = False
    drain_started: float | None = None


@dataclass
class AmbulanceCorridorState:
    """Explicit state for an ambulance's complete upcoming corridor."""

    ambulance_id: str
    route: tuple[str, ...]
    current_route_index: int | None
    junctions: list[CorridorJunctionState]
    active: bool = True
    route_generation: int = 0
    # Monotonically increasing detection order. The ambulance detected FIRST
    # (lowest order) keeps FCFS priority at shared junctions, regardless of
    # how its live ETA/urgency fluctuates while it is blocked in traffic.
    detection_order: int | None = None


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

    def _apply_visual_markers(self, vehicle_ids):
        """Apply only the requested emergency visualization signals.

        Emergency vehicles get two rear red light dots and the blue emergency
        light. Normal vehicles get signal state 0 so they remain visually plain.
        No left/right blinkers are enabled here.
        """
        set_signals = getattr(self.traci.vehicle, "setSignals", None)
        if set_signals is None:
            return

        for vehicle_id in vehicle_ids:
            vehicle_type = self._safe_call(
                self.traci.vehicle.getTypeID, vehicle_id, default=""
            )
            vehicle_class = self._safe_call(
                self.traci.vehicle.getVehicleClass, vehicle_id, default=""
            )
            is_emergency = self._is_emergency(vehicle_type, vehicle_class)
            signal_state = EMERGENCY_VISUAL_SIGNALS if is_emergency else 0
            self._safe_call(set_signals, vehicle_id, signal_state, default=None)

    def _junction_for_movement(self, movement, topologies):
        for junction_id, topology in topologies.items():
            if movement in topology.movements:
                return junction_id
        return None

    def discover(self, topologies):
        """Discover current emergency vehicles from SUMO type/class data."""
        states = []
        current_ids = set(self.traci.vehicle.getIDList())

        # Apply the visual state every simulation step.
        self._apply_visual_markers(current_ids)

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

            # Prefer the route's exact edge pair. This is more reliable than
            # getRoadID() during SUMO's transient internal connector states.
            if current_edge in route:
                route_pos = route.index(current_edge)
                if route_pos + 1 < len(route):
                    route_from = route[route_pos]
                    route_to = route[route_pos + 1]
                    for junction_id, topology in topologies.items():
                        for movement_id, movement in topology.movements.items():
                            if (
                                movement.from_road == route_from
                                and movement.to_road == route_to
                            ):
                                current_junction = junction_id
                                required_movement = movement_id
                                break
                        if current_junction:
                            break

            # Preserve the original live-edge lookup as a fallback.
            if current_junction is None:
                for junction_id, topology in topologies.items():
                    for movement_id, movement in topology.movements.items():
                        if (
                            movement.from_road == current_edge
                            and movement.to_road == next_edge
                        ):
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
                                    route_index=(
                                        int(route_index)
                                        if route_index is not None
                                        else None
                                    ),
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

    @classmethod
    def _resolve_required_movement(cls, state, topology):
        """Resolve the ambulance movement from stable route information.

        SUMO may briefly report an internal connector edge while a vehicle is
        approaching a junction. The route sequence is more stable for
        identifying the intended turn, so try the exact route pair first and
        then fall back to the live edge.
        """
        route = tuple(state.route or ())
        current_edge = state.current_edge

        # 1. Exact live-edge -> next-route-edge pair.
        if current_edge in route:
            index = route.index(current_edge)
            if index + 1 < len(route):
                next_edge = route[index + 1]
                for candidate_id, movement in topology.movements.items():
                    if (
                        movement.from_road == current_edge
                        and movement.to_road == next_edge
                    ):
                        return candidate_id, next_edge

        # 2. If getRoadID() is temporarily an internal connector, search the
        # route for a movement belonging to this junction.
        for index in range(len(route) - 1):
            from_edge = route[index]
            to_edge = route[index + 1]
            for candidate_id, movement in topology.movements.items():
                if (
                    movement.from_road == from_edge
                    and movement.to_road == to_edge
                ):
                    if current_edge is None or current_edge in (from_edge, to_edge):
                        return candidate_id, to_edge

        # 3. Final live-edge fallback.
        if current_edge:
            for candidate_id, movement in topology.movements.items():
                if movement.from_road == current_edge:
                    return candidate_id, movement.to_road

        return None, None

    def plan_for_junction(self, state, junction_id, topology, phases, generator):
        """Build a safe candidate plan for the vehicle's required movement."""
        movement_id, next_edge = self._resolve_required_movement(state, topology)

        if movement_id is None:
            return EmergencyPlan(
                state.vehicle_id, junction_id, None, (), None,
                state.eta_seconds, self.urgency(state), False,
                state.current_edge, next_edge, state.distance_to_junction,
                state.speed_mps, ()
            )

        candidate_phases = tuple(
            phase["phase_id"]
            for phase in phases
            if movement_id in phase["movements"]
            and generator.is_compatible(phase["movements"])
        )

        # Use a native SUMO phase containing the ambulance movement. This
        # preserves the existing conflict/safety model and does not invent
        # signal states.
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

    def build_corridor_plan(
        self,
        state,
        topologies,
        runtimes,
        distance_resolver,
    ):
        """Build plans for every upcoming signalized movement on the route."""
        route = tuple(state.route or ())
        if len(route) < 2:
            return AmbulanceCorridorState(
                state.vehicle_id,
                route,
                state.route_index,
                [],
            )

        route_index = state.route_index
        if route_index is None:
            route_index = 0
        route_index = max(0, min(int(route_index), len(route) - 1))

        junctions = []
        seen_movements = set()

        for movement_route_index in range(route_index, len(route) - 1):
            from_edge = route[movement_route_index]
            to_edge = route[movement_route_index + 1]
            if str(from_edge).startswith(":") or str(to_edge).startswith(":"):
                continue

            matched = None
            for junction_id, topology in topologies.items():
                for movement_id, movement in topology.movements.items():
                    if (
                        movement.from_road == from_edge
                        and movement.to_road == to_edge
                    ):
                        matched = junction_id, movement_id, topology
                        break
                if matched is not None:
                    break

            if matched is None:
                continue

            junction_id, movement_id, topology = matched
            movement_key = (junction_id, movement_id, movement_route_index)
            if movement_key in seen_movements:
                continue
            seen_movements.add(movement_key)

            runtime = runtimes.get(junction_id)
            if runtime is None:
                continue

            candidate_phases = tuple(
                phase["phase_id"]
                for phase in runtime["phases"]
                if (
                    movement_id in phase["movements"]
                    and runtime["generator"].is_compatible(
                        phase["movements"]
                    )
                )
            )
            if not candidate_phases:
                continue

            distance = distance_resolver(
                state,
                route_index,
                movement_route_index,
            )
            speed = float(state.speed_mps or 0.0)
            eta = (
                distance / speed
                if distance is not None and speed >= self.max_speed_floor
                else None
            )
            conflicting = tuple(
                other_id
                for other_id in topology.movements
                if (
                    other_id != movement_id
                    and runtime["generator"].graph.are_conflicting(
                        movement_id,
                        other_id,
                    )
                )
            )
            plan = EmergencyPlan(
                vehicle_id=state.vehicle_id,
                junction_id=junction_id,
                required_movement=movement_id,
                candidate_phases=candidate_phases,
                selected_phase=candidate_phases[0],
                eta_seconds=eta,
                urgency=(
                    max(0.0, min(1.0, math.exp(-eta / 30.0)))
                    if eta is not None
                    else 0.35
                ),
                safe=True,
                current_edge=state.current_edge,
                next_edge=to_edge,
                distance_to_junction=distance,
                vehicle_speed_mps=speed,
                conflicting_movements=conflicting,
                route_index=route_index,
                movement_route_index=movement_route_index,
                from_edge=from_edge,
            )
            junctions.append(CorridorJunctionState(plan=plan))

        return AmbulanceCorridorState(
            ambulance_id=state.vehicle_id,
            route=route,
            current_route_index=route_index,
            junctions=junctions,
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

    # ------------------------------------------------------------------
    # CORRIDOR LIFECYCLE
    # ------------------------------------------------------------------

    @staticmethod
    def has_ambulance_passed_junction(state, junction_state):
        """Determine whether the ambulance has passed a corridor junction.

        Uses the ambulance's current route_index compared to the junction's
        movement_route_index.  When the ambulance's route_index exceeds the
        junction's from-edge position, it has crossed that junction.

        Falls back to edge comparison when route indices are unavailable.
        """
        plan = junction_state.plan
        movement_route_index = plan.movement_route_index
        route_index = state.route_index

        # Already flagged.
        if junction_state.passed:
            return True

        # Route-index comparison is the most reliable approach.
        if route_index is not None and movement_route_index is not None:
            if route_index > movement_route_index:
                return True

        # Fallback: the ambulance is currently on the junction's outgoing
        # edge (or beyond).
        if state.current_edge and plan.next_edge:
            route = list(state.route or ())
            if plan.next_edge in route and state.current_edge in route:
                next_pos = route.index(plan.next_edge)
                curr_pos = route.index(state.current_edge)
                if curr_pos >= next_pos:
                    return True

        return False

    @staticmethod
    def should_activate_junction(
        junction_state,
        eta_threshold=45.0,
        distance_threshold=500.0,
    ):
        """Decide whether a prepared junction should be activated now.

        Junctions far ahead of the ambulance stay ``prepared``.  Activation
        occurs only when the ambulance's ETA or distance falls below the
        configured thresholds.
        """
        if junction_state.activated or junction_state.released:
            return junction_state.activated and not junction_state.released

        plan = junction_state.plan
        eta = plan.eta_seconds
        distance = plan.distance_to_junction

        eta_trigger = eta is not None and eta <= eta_threshold
        distance_trigger = (
            distance is not None and distance <= distance_threshold
        )
        return bool(eta_trigger or distance_trigger)

    def check_downstream_congestion(
        self,
        junction_state,
        congestion_threshold=0.8,
    ):
        """Check if the downstream edge of a junction is congested.

        Returns a float in [0, 1] representing downstream occupancy.
        A value >= *congestion_threshold* means downstream is congested and
        activation should be deferred.
        """
        plan = junction_state.plan
        downstream_edge = plan.next_edge
        if not downstream_edge or str(downstream_edge).startswith(":"):
            return 0.0

        occupancy = self._safe_call(
            self.traci.edge.getLastStepOccupancy,
            downstream_edge,
            default=0.0,
        )
        return float(occupancy or 0.0)

    def _junction_drained(self, jstate, now):
        """Whether a passed junction's approach edge has drained.

        The junction keeps its green phase until no vehicle remains within
        ``DRAIN_ZONE_METERS`` of the junction on the approach edge AND no
        vehicle remains on the junction's internal connectors.  A safety cap
        (``DRAIN_MAX_SECONDS``) guarantees the hold always ends.
        """
        plan = jstate.plan

        # Safety cap: never hold a junction green indefinitely.
        if (
            jstate.drain_started is not None
            and now - jstate.drain_started >= DRAIN_MAX_SECONDS
        ):
            return True

        edge_api = getattr(self.traci, "edge", None)
        vehicle_api = getattr(self.traci, "vehicle", None)
        lane_api = getattr(self.traci, "lane", None)

        # No vehicle may remain inside the junction itself.
        junction_id = plan.junction_id
        if edge_api is not None:
            for edge_id in self._safe_call(
                edge_api.getIDList, default=()
            ):
                if str(edge_id).startswith(f":{junction_id}_"):
                    if self._safe_call(
                        edge_api.getLastStepVehicleNumber,
                        edge_id,
                        default=0,
                    ):
                        return False

        # No vehicle may sit in the drain zone near the junction on the
        # approach edge — those are the cars a phase switch would trap.
        from_edge = plan.from_edge
        if (
            from_edge
            and not str(from_edge).startswith(":")
            and edge_api is not None
        ):
            for vehicle_id in self._safe_call(
                edge_api.getLastStepVehicleIDs,
                from_edge,
                default=(),
            ):
                lane_id = self._safe_call(
                    vehicle_api.getLaneID,
                    vehicle_id,
                    default=None,
                ) if vehicle_api is not None else None
                if lane_id is None:
                    continue
                lane_length = self._safe_call(
                    lane_api.getLength,
                    lane_id,
                    default=0.0,
                ) if lane_api is not None else 0.0
                position = self._safe_call(
                    vehicle_api.getLanePosition,
                    vehicle_id,
                    default=None,
                ) if vehicle_api is not None else None
                if position is None:
                    continue
                if lane_length - position <= DRAIN_ZONE_METERS:
                    return False

        return True

    def update_corridor_progression(self, corridor, state):
        """Update corridor junction states based on ambulance progression.

        When the ambulance passes a junction it enters a ``draining`` hold:
        the green phase stays active until the approach edge clears, so the
        queue behind the ambulance is not trapped when the normal AI resumes.
        Only then is the junction marked ``released``.  Returns a list of
        junction IDs that were newly released during this call.
        """
        newly_released = []
        simulation_api = getattr(self.traci, "simulation", None)
        now = (
            self._safe_call(simulation_api.getTime, default=0.0)
            if simulation_api is not None
            else 0.0
        )

        for jstate in corridor.junctions:
            if jstate.released:
                continue

            if not jstate.passed:
                if self.has_ambulance_passed_junction(state, jstate):
                    jstate.passed = True
                    jstate.draining = True
                    jstate.drain_started = now
                    jstate.status = "draining"

            if jstate.draining and self._junction_drained(jstate, now):
                jstate.draining = False
                jstate.released = True
                jstate.status = "released"
                newly_released.append(jstate.plan.junction_id)

        # Update the corridor's route index tracking.
        if state.route_index is not None:
            corridor.current_route_index = state.route_index

        # Mark corridor inactive when every junction has been released.
        if corridor.junctions and all(
            js.released for js in corridor.junctions
        ):
            corridor.active = False

        return newly_released
