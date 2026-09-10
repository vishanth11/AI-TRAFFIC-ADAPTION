"""
Canonical Phase 2 SUMO/TraCI adaptive signal controller.

This controller integrates the existing:
- topology discovery
- conflict graph
- demand calculation
- traffic prediction
- decision engine
- safety gate
- fallback controller
- emergency preemption

with the NEW multimodal SUMO environment.

SUMO remains the source of truth for:
- junctions
- traffic-light phases
- vehicle positions
- traffic demand
- simulation time

The AI modules remain responsible for:
- prediction
- phase scoring
- decision making
- safety validation
- emergency priority
"""

import os
import shutil
import sys
import math
import argparse
from types import SimpleNamespace

import numpy as np


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# SUMO / TRACI
# ---------------------------------------------------------------------------

import traci


# ---------------------------------------------------------------------------
# EXISTING AI / INTEGRATION MODULES
# ---------------------------------------------------------------------------

from integration.controller import TrafficController
from integration.decision_engine import DecisionEngine
from integration.emergency import (
    AmbulanceCorridorState,
    CorridorJunctionState,
    EmergencyCorridorManager,
    EmergencyPlan,
    EmergencyVehicleState,
)
from integration.prediction_adapter import TrafficPredictionAdapter
from integration.sumo_conflict_adapter import SUMOConflictAdapter
from integration.sumo_demand import SUMODemandCalculator
from integration.sumo_phase_adapter import SUMOPhaseAdapter
from integration.sumo_safety import SUMOSafetyRiskAdapter
from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.traffic_history import TrafficHistory
from integration.sumo_prediction_controller import SUMOTrafficStateBuilder


# ---------------------------------------------------------------------------
# NEW SUMO ENVIRONMENT
# ---------------------------------------------------------------------------

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "simulation",
    "configs",
    "large_grid_multimodal.sumocfg",
)

# Emergency uses the SAME new multimodal environment.
# We intentionally do not switch back to corridor_emergency.sumocfg.
EMERGENCY_SUMO_CONFIG = SUMO_CONFIG


# ---------------------------------------------------------------------------
# RUNTIME CONFIGURATION
# ---------------------------------------------------------------------------

NUM_SENSORS = 36
HISTORY_LENGTH = 10

CONTROL_INTERVAL = 10.0
PREDICTION_INTERVAL = 10

MIN_GREEN_SECONDS = 10.0

WARMUP_STEPS = 10
TOTAL_STEPS = 200


# ---------------------------------------------------------------------------
# EMERGENCY PREEMPTION
# ---------------------------------------------------------------------------

# Proactive emergency preemption.
#
# The ambulance does NOT receive SUMO's special red-light bypass.
# Instead, the AI controller opens the correct compatible green phase
# before the ambulance reaches the junction.

EMERGENCY_PREEMPT_ETA_SECONDS = 45.0
EMERGENCY_PREEMPT_DISTANCE_METERS = 500.0
DOWNSTREAM_CONGESTION_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _clamp(value):
    """
    Clamp a value into [0, 1].
    """
    return max(0.0, min(1.0, float(value)))


class NativePhaseCompatibility:
    """Compatibility checker for phases supplied directly by SUMO.

    The new multimodal network has 32 movements per junction. Exhaustively
    generating maximal compatible movement sets is combinatorial and is not
    necessary because SUMO already provides valid signal programs.
    """

    def __init__(self, conflict_graph):
        # Keep both names for compatibility with the emergency planner.
        # The normal decision path uses conflict_graph, while emergency.py
        # accesses the conflict graph through generator.graph.
        self.conflict_graph = conflict_graph
        self.graph = conflict_graph

    def is_compatible(self, movement_ids):
        """Return True when no two movements in the phase conflict."""
        movement_ids = list(dict.fromkeys(movement_ids or []))
        for i, first in enumerate(movement_ids):
            for second in movement_ids[i + 1:]:
                if self.conflict_graph.are_conflicting(first, second):
                    return False
        return True


def _phase_objects(phase_representations):
    """
    Adapt phase dictionaries to the API expected by SUMODemandCalculator.
    """

    return [
        SimpleNamespace(
            phase_id=phase["phase_id"],
            movement_ids=phase["movements"],
        )
        for phase in phase_representations
    ]


def _phase_sensor_ids(phase, sensor_builder, topology):
    """
    Map a phase's incoming roads to deployment sensor zones.

    The existing prediction model operates on 36 sensors.
    The state builder provides the deployment mapping.
    """

    roads = {
        topology.movements[movement_id].from_road
        for movement_id in phase["movements"]
        if movement_id in topology.movements
    }

    sensor_ids = []

    for sensor_id, info in sensor_builder.sensor_mapping.items():
        if (
            info["junction_id"] == phase["junction_id"]
            and info["road_id"] in roads
        ):
            sensor_ids.append(sensor_id)

    return sensor_ids


def _safe_transition_index(program, target_index):
    """
    Find the transition phase immediately before a target green phase.

    This prevents directly jumping from one green phase to another.
    SUMO is allowed to execute the yellow transition first.
    """

    phases = program["phases"]

    target_position = next(
        i
        for i, phase in enumerate(phases)
        if phase["phase_index"] == target_index
    )

    if target_position == 0:
        return target_index

    previous = phases[target_position - 1]

    if (
        "y" in previous["state"].lower()
        or "r" in previous["state"].lower()
    ):
        return previous["phase_index"]

    return target_index


def _find_sumo_executable():
    """
    Find SUMO through:
    1. SUMO_HOME
    2. PATH
    3. standard Windows installation paths
    """

    candidates = []

    if os.environ.get("SUMO_HOME"):
        candidates.append(
            os.path.join(
                os.environ["SUMO_HOME"],
                "bin",
                "sumo.exe",
            )
        )

    path_executable = shutil.which("sumo")

    if path_executable:
        candidates.append(path_executable)

    if os.environ.get("ProgramFiles(x86)"):
        candidates.append(
            os.path.join(
                os.environ["ProgramFiles(x86)"],
                "Eclipse",
                "Sumo",
                "bin",
                "sumo.exe",
            )
        )

    if os.environ.get("ProgramW6432"):
        candidates.append(
            os.path.join(
                os.environ["ProgramW6432"],
                "Eclipse",
                "Sumo",
                "bin",
                "sumo.exe",
            )
        )

    for candidate in candidates:
        if os.path.isfile(candidate):

            if not os.environ.get("SUMO_HOME"):
                sumo_bin = os.path.dirname(
                    os.path.abspath(candidate)
                )

                os.environ["SUMO_HOME"] = os.path.dirname(
                    sumo_bin
                )

            return candidate

    return "sumo"



def _find_sumo_gui_executable():
    """Find the SUMO-GUI executable."""
    candidates = []

    if os.environ.get("SUMO_HOME"):
        candidates.append(
            os.path.join(
                os.environ["SUMO_HOME"], "bin", "sumo-gui.exe"
            )
        )

    path_executable = shutil.which("sumo-gui")
    if path_executable:
        candidates.append(path_executable)

    for variable in ("ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(variable)
        if root:
            candidates.append(
                os.path.join(
                    root,
                    "Eclipse",
                    "Sumo",
                    "bin",
                    "sumo-gui.exe",
                )
            )

    for candidate in candidates:
        if os.path.isfile(candidate):
            if not os.environ.get("SUMO_HOME"):
                os.environ["SUMO_HOME"] = os.path.dirname(
                    os.path.dirname(os.path.abspath(candidate))
                )
            return candidate

    return "sumo-gui"

SUMO_CMD = [
    _find_sumo_executable(),
    "-c",
    SUMO_CONFIG,
]


# ---------------------------------------------------------------------------
# MAIN CONTROLLER
# ---------------------------------------------------------------------------

class Phase2SUMOController:
    """
    Main adaptive traffic-signal controller.

    The controller dynamically discovers all usable signalized junctions
    from the active SUMO network.

    No A0/B0/C0 junction IDs are hardcoded here.
    """

    def __init__(
        self,
        sumo_cmd=None,
        total_steps=TOTAL_STEPS,
        control_interval=CONTROL_INTERVAL,
        prediction_interval=PREDICTION_INTERVAL,
        warmup_steps=WARMUP_STEPS,
        traci_connection=None,
        step_observer=None,
        emergency_enabled=False,
        decision_observer=None,
    ):

        self.sumo_cmd = sumo_cmd or SUMO_CMD

        self.total_steps = int(total_steps)

        self.control_interval = float(
            control_interval
        )

        self.prediction_interval = max(
            1,
            int(prediction_interval),
        )

        self.warmup_steps = int(
            warmup_steps
        )

        self.traci = traci_connection or traci

        self.step_observer = step_observer

        self.emergency_enabled = bool(
            emergency_enabled
        )

        self.decision_observer = decision_observer

        self._observe_steps = False

        self.decisions = []
        self.corridors = {}
        self.corridor_events = []
        # Monotonic counter for FCFS ambulance detection order.
        self._detection_counter = 0


    # -----------------------------------------------------------------------
    # SIMULATION STEP
    # -----------------------------------------------------------------------

    def _simulation_step(self):

        self.traci.simulationStep()

        if (
            self.step_observer is not None
            and self._observe_steps
        ):
            self.step_observer()


    # -----------------------------------------------------------------------
    # BUILD RUNTIME
    # -----------------------------------------------------------------------

    def _build_runtime(self):

        topology_adapter = SUMOTopologyAdapter(
            self.traci
        )

        phase_adapter = SUMOPhaseAdapter(
            self.traci
        )

        conflict_adapter = SUMOConflictAdapter(
            phase_adapter
        )

        demand_calculator = SUMODemandCalculator()

        traffic_controller = TrafficController()

        decision_engine = DecisionEngine()


        # ---------------------------------------------------------------
        # DYNAMIC JUNCTION DISCOVERY
        # ---------------------------------------------------------------

        signalized_ids = (
            topology_adapter.discover_signalized_junctions()
        )

        if not signalized_ids:
            raise RuntimeError(
                "No signalized junctions discovered from SUMO"
            )

        topologies = {}

        for junction_id in signalized_ids:

            topology = (
                topology_adapter.discover_junction(
                    junction_id
                )
            )

            if topology is not None:
                topologies[junction_id] = topology


        print(
            "\n"
            + "=" * 70
        )

        print(
            "NEW MULTIMODAL SUMO ENVIRONMENT"
        )

        print(
            "=" * 70
        )

        print(
            f"Signalized junctions discovered: "
            f"{len(topologies)}"
        )

        print(
            f"Junction IDs: "
            f"{list(topologies.keys())}"
        )

        print(
            "=" * 70
        )

        print(
            "Using SUMO-native signal phases; maximal phase generation disabled."
        )


        # ---------------------------------------------------------------
        # CREATE PER-JUNCTION RUNTIME
        # ---------------------------------------------------------------

        runtimes = {}

        for junction_id, topology in topologies.items():

            conflict_graph = (
                conflict_adapter.build_conflict_graph(
                    junction_id,
                    topology,
                    topology_adapter.sumo_metadata,
                )
            )

            # IMPORTANT: Use SUMO's native signal program phases.
            # Do not call PhaseGenerator.generate_maximal_phases() here.
            # With 32 movements per junction, exhaustive maximal-phase
            # generation becomes computationally explosive.
            generator = NativePhaseCompatibility(
                conflict_graph
            )

            representations = (
                phase_adapter.build_phase_representation(
                    junction_id,
                    topology_adapter.sumo_metadata,
                )
            )

            if not representations:
                print(
                    f"[WARNING] {junction_id}: "
                    "no usable phase representations"
                )
                continue


            runtimes[junction_id] = {

                "topology": topology,

                "generator": generator,

                "generated": None,

                "phases": representations,

                "phase_objects":
                    _phase_objects(
                        representations
                    ),

                "phase_by_id":
                    {
                        phase["phase_id"]: phase
                        for phase in representations
                    },

                "last_control":
                    -float("inf"),

                "last_phase":
                    None,

                "emergency_hold_phase":
                    None,

                "emergency_vehicle_id":
                    None,
            }


            print(
                f"[TOPOLOGY] "
                f"{junction_id} | "
                f"roads={len(topology.roads)} | "
                f"movements={len(topology.movements)} | "
                f"green_phases={len(representations)} | "
                "native_sumo_phases=True"
            )


        if not runtimes:

            raise RuntimeError(
                "No usable green signal phases were discovered"
            )


        print(
            f"\nUsable signalized junctions: "
            f"{len(runtimes)}"
        )

        return (
            topology_adapter,
            traffic_controller,
            decision_engine,
            demand_calculator,
            runtimes,
        )


    # -----------------------------------------------------------------------
    # EXECUTE SIGNAL PHASE
    # -----------------------------------------------------------------------

    def _execute(
        self,
        junction_id,
        target_phase,
        phase_adapter,
    ):

        current = (
            phase_adapter.get_current_phase(
                junction_id
            )
        )

        current_state = (
            phase_adapter.get_phase_state(
                junction_id,
                current,
            )
        )

        program = (
            phase_adapter.get_current_program(
                junction_id
            )
        )

        transition = _safe_transition_index(
            program,
            target_phase,
        )


        print(
            f"EXECUTE "
            f"time={self.traci.simulation.getTime():.1f} "
            f"junction={junction_id} "
            f"current_phase={current} "
            f"current_state={current_state} "
            f"target_phase={target_phase} "
            f"transition_phase={transition}"
        )


        # ---------------------------------------------------------------
        # SAFE TRANSITION
        # ---------------------------------------------------------------

        if current != target_phase:

            self.traci.trafficlight.setPhase(
                junction_id,
                transition,
            )

            transition_duration = next(
                (
                    phase["duration"]
                    for phase in program["phases"]
                    if phase["phase_index"] == transition
                ),
                1.0,
            )

            max_transition_steps = max(
                1,
                int(
                    math.ceil(
                        float(
                            transition_duration
                        )
                    )
                )
                + 1,
            )


            for _ in range(
                max_transition_steps
            ):

                self._simulation_step()

                actual_phase = (
                    phase_adapter.get_current_phase(
                        junction_id
                    )
                )

                if actual_phase == target_phase:
                    break


        # ---------------------------------------------------------------
        # READBACK FROM SUMO
        # ---------------------------------------------------------------

        executed = (
            phase_adapter.get_current_phase(
                junction_id
            )
        )

        executed_state = (
            phase_adapter.get_phase_state(
                junction_id,
                executed,
            )
        )


        print(
            f"READBACK "
            f"junction={junction_id} "
            f"executed_phase={executed} "
            f"resulting_state={executed_state}"
        )


        return {
            "phase": executed,
            "state": executed_state,
            "requested": target_phase,
        }


    # -----------------------------------------------------------------------
    # EMERGENCY PREEMPTION
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # ROUTE-AHEAD EMERGENCY PLANNING
    # -----------------------------------------------------------------------

    def _edge_length(self, edge_id):
        """Return a SUMO edge length with a safe fallback."""
        if not edge_id or str(edge_id).startswith(":"):
            return 0.0

        try:
            return max(0.0, float(self.traci.edge.getLength(edge_id)))
        except Exception:
            return 0.0

    def _route_distance_to_movement(
        self,
        state,
        route_index,
        movement_route_index,
    ):
        """
        Calculate distance from the ambulance's current position to the end
        of the approach edge that contains the required movement.

        The old emergency path used only:
            current_lane_length - current_lane_position

        That becomes approximately zero when the ambulance enters the
        approach edge, which is why some preemption events appeared at
        ~1 metre. This method walks forward through the actual SUMO route.
        """
        route = list(state.route or [])
        if not route:
            return None

        if route_index < 0 or movement_route_index < route_index:
            return None

        distance = 0.0

        current_lane = state.current_lane
        current_position = state.current_position

        if route_index < len(route):
            if current_lane and current_position is not None:
                try:
                    lane_length = float(
                        self.traci.lane.getLength(current_lane)
                    )
                    distance += max(
                        0.0,
                        lane_length - float(current_position),
                    )
                except Exception:
                    distance += self._edge_length(route[route_index])
            else:
                distance += self._edge_length(route[route_index])

        for index in range(route_index + 1, movement_route_index + 1):
            distance += self._edge_length(route[index])

        return max(0.0, distance)

    def _update_corridor_plans(
        self,
        emergency_states,
        topologies,
        runtimes,
        emergency_manager,
    ):
        """
        Build or update route-wide corridor plans for every ambulance.

        For each emergency vehicle:
        1. Build a full corridor plan via EmergencyCorridorManager.
        2. If the route changed, rebuild the corridor.
        3. Update progression — mark passed junctions.
        4. Collect per-junction activation decisions.

        Returns a dict of {junction_id: EmergencyPlan} for junctions that
        should be preempted this step.
        """
        now = self.traci.simulation.getTime()
        active_plans = {}

        # Track which ambulance IDs are still present.
        current_ambulance_ids = set()

        for state in emergency_states:
            current_ambulance_ids.add(state.vehicle_id)

            # Fetch live route index from TraCI.
            try:
                live_route_index = int(
                    self.traci.vehicle.getRouteIndex(state.vehicle_id)
                )
            except Exception:
                live_route_index = state.route_index or 0

            # Create a state with the live route index for corridor building.
            live_state = EmergencyVehicleState(
                vehicle_id=state.vehicle_id,
                vehicle_type=state.vehicle_type,
                current_edge=state.current_edge,
                current_lane=state.current_lane,
                current_position=state.current_position,
                distance_to_junction=state.distance_to_junction,
                speed_mps=state.speed_mps,
                route=state.route,
                current_junction=state.current_junction,
                next_junction=state.next_junction,
                destination=state.destination,
                emergency=state.emergency,
                confidence=state.confidence,
                eta_seconds=state.eta_seconds,
                route_index=live_route_index,
            )

            existing = self.corridors.get(state.vehicle_id)

            # Detect route changes — rebuild if the route is different.
            rebuild = False
            if existing is None:
                rebuild = True
            elif tuple(state.route or ()) != existing.route:
                rebuild = True
                self.corridor_events.append({
                    "time": now,
                    "event": "corridor_route_changed",
                    "ambulance": state.vehicle_id,
                })
                print(
                    f"[CORRIDOR_ROUTE_CHANGED] "
                    f"time={now:.1f} "
                    f"vehicle={state.vehicle_id} "
                    "action=REBUILD_CORRIDOR"
                )

            if rebuild:
                corridor = emergency_manager.build_corridor_plan(
                    live_state,
                    topologies,
                    runtimes,
                    self._route_distance_to_movement,
                )
                # FCFS: the first ambulance detected keeps priority. Preserve
                # the detection order across route-change rebuilds so a
                # corridor rebuild never demotes an earlier ambulance.
                if existing is not None and existing.detection_order is not None:
                    corridor.detection_order = existing.detection_order
                else:
                    corridor.detection_order = self._detection_counter
                    self._detection_counter += 1
                self.corridors[state.vehicle_id] = corridor

                if corridor.junctions:
                    junction_summary = "; ".join(
                        f"{js.plan.junction_id} "
                        f"ETA={js.plan.eta_seconds:.1f}s "
                        f"movement={js.plan.required_movement} "
                        f"phase={js.plan.selected_phase}"
                        if js.plan.eta_seconds is not None
                        else f"{js.plan.junction_id} "
                        f"movement={js.plan.required_movement} "
                        f"phase={js.plan.selected_phase}"
                        for js in corridor.junctions
                    )
                    print(
                        f"[GREEN_CORRIDOR_CREATED] "
                        f"time={now:.1f} "
                        f"vehicle={state.vehicle_id} "
                        f"junctions={len(corridor.junctions)} "
                        f"plan=[{junction_summary}]"
                    )
                    self.corridor_events.append({
                        "time": now,
                        "event": "corridor_created",
                        "ambulance": state.vehicle_id,
                        "junction_count": len(corridor.junctions),
                        "junctions": [
                            js.plan.junction_id
                            for js in corridor.junctions
                        ],
                    })
            else:
                corridor = existing
                # Re-compute ETAs with current speed/position.
                updated_corridor = emergency_manager.build_corridor_plan(
                    live_state,
                    topologies,
                    runtimes,
                    self._route_distance_to_movement,
                )
                # Merge updated plans into existing corridor, preserving
                # activation/passage state.
                updated_map = {
                    js.plan.junction_id: js.plan
                    for js in updated_corridor.junctions
                }
                for jstate in corridor.junctions:
                    if jstate.released:
                        continue
                    new_plan = updated_map.get(jstate.plan.junction_id)
                    if new_plan is not None:
                        jstate.plan = new_plan

            # Update progression — mark passed junctions.
            newly_released = emergency_manager.update_corridor_progression(
                corridor, live_state,
            )
            for released_junction_id in newly_released:
                print(
                    f"[CORRIDOR_JUNCTION_PASSED] "
                    f"time={now:.1f} "
                    f"vehicle={state.vehicle_id} "
                    f"junction={released_junction_id} "
                    "action=RELEASED"
                )
                self.corridor_events.append({
                    "time": now,
                    "event": "junction_passed",
                    "ambulance": state.vehicle_id,
                    "junction": released_junction_id,
                })

            if not corridor.active:
                print(
                    f"[GREEN_CORRIDOR_COMPLETED] "
                    f"time={now:.1f} "
                    f"vehicle={state.vehicle_id} "
                    "all_junctions_passed=True"
                )
                self.corridor_events.append({
                    "time": now,
                    "event": "corridor_completed",
                    "ambulance": state.vehicle_id,
                })

            # Collect junctions that should be activated this step.
            for jstate in corridor.junctions:
                if jstate.released:
                    continue

                if not emergency_manager.should_activate_junction(
                    jstate,
                    eta_threshold=EMERGENCY_PREEMPT_ETA_SECONDS,
                    distance_threshold=EMERGENCY_PREEMPT_DISTANCE_METERS,
                ):
                    continue

                # Downstream congestion check.  Skipped for draining
                # junctions: they are holding green to clear the queue behind
                # the ambulance, not to admit it, so a congested exit edge must
                # not cut the drain short (that would trap cars mid-junction).
                if not jstate.draining:
                    downstream_occ = emergency_manager.check_downstream_congestion(
                        jstate,
                        congestion_threshold=DOWNSTREAM_CONGESTION_THRESHOLD,
                    )
                    if downstream_occ >= DOWNSTREAM_CONGESTION_THRESHOLD:
                        print(
                            f"[CORRIDOR_DOWNSTREAM_CONGESTED] "
                            f"time={now:.1f} "
                            f"vehicle={state.vehicle_id} "
                            f"junction={jstate.plan.junction_id} "
                            f"downstream_occupancy={downstream_occ:.2f} "
                            "action=DEFER_ACTIVATION"
                        )
                        continue

                junction_id = jstate.plan.junction_id
                plan = jstate.plan

                # FCFS: if multiple ambulances want the same junction, the
                # ambulance detected FIRST keeps priority. Live urgency/ETA
                # is only a deterministic tie-breaker for ambulances detected
                # in the same step. This prevents a later ambulance from
                # preempting an earlier one whose ETA inflated because it is
                # blocked in traffic.
                current_plan = active_plans.get(junction_id)
                take_plan = current_plan is None
                if not take_plan:
                    this_corridor = self.corridors.get(plan.vehicle_id)
                    current_corridor = self.corridors.get(current_plan.vehicle_id)
                    this_order = (
                        this_corridor.detection_order
                        if this_corridor is not None
                        else None
                    )
                    current_order = (
                        current_corridor.detection_order
                        if current_corridor is not None
                        else None
                    )
                    if this_order is not None and current_order is not None:
                        if this_order < current_order:
                            take_plan = True
                        elif this_order == current_order:
                            # Same detection step: deterministic tie-break on
                            # urgency, then ETA.
                            take_plan = (
                                plan.urgency > current_plan.urgency
                                or (
                                    plan.urgency == current_plan.urgency
                                    and (
                                        plan.eta_seconds is not None
                                        and (
                                            current_plan.eta_seconds is None
                                            or plan.eta_seconds
                                            < current_plan.eta_seconds
                                        )
                                    )
                                )
                            )
                    else:
                        # Missing detection order: fall back to urgency.
                        take_plan = (
                            plan.urgency > current_plan.urgency
                            or (
                                plan.urgency == current_plan.urgency
                                and (
                                    plan.eta_seconds is not None
                                    and (
                                        current_plan.eta_seconds is None
                                        or plan.eta_seconds
                                        < current_plan.eta_seconds
                                    )
                                )
                            )
                        )
                if take_plan:
                    active_plans[junction_id] = plan

                if not jstate.activated:
                    jstate.activated = True
                    jstate.status = "activated"
                    print(
                        f"[CORRIDOR_JUNCTION_ACTIVATED] "
                        f"time={now:.1f} "
                        f"vehicle={state.vehicle_id} "
                        f"junction={junction_id} "
                        f"distance={plan.distance_to_junction} "
                        f"ETA={plan.eta_seconds} "
                        f"movement={plan.required_movement} "
                        f"phase={plan.selected_phase}"
                    )
                    self.corridor_events.append({
                        "time": now,
                        "event": "junction_activated",
                        "ambulance": state.vehicle_id,
                        "junction": junction_id,
                    })

        # Clean up corridors for ambulances that disappeared.
        disappeared = set(self.corridors.keys()) - current_ambulance_ids
        for ambulance_id in disappeared:
            corridor = self.corridors.pop(ambulance_id, None)
            if corridor is not None:
                print(
                    f"[CORRIDOR_AMBULANCE_DISAPPEARED] "
                    f"time={now:.1f} "
                    f"vehicle={ambulance_id} "
                    "action=RELEASE_ALL_JUNCTIONS"
                )
                self.corridor_events.append({
                    "time": now,
                    "event": "ambulance_disappeared",
                    "ambulance": ambulance_id,
                })
                # Mark all junctions released so the main loop clears holds.
                for jstate in corridor.junctions:
                    jstate.released = True
                    jstate.passed = True
                    jstate.status = "released"
                corridor.active = False

        return active_plans

    @staticmethod
    def _emergency_requires_preemption(
        emergency_plan
    ):
        """
        Determine whether an approaching emergency vehicle
        requires proactive signal preemption.
        """

        if (
            emergency_plan is None
            or not emergency_plan.selected_phase
        ):
            return False


        eta = (
            emergency_plan.eta_seconds
        )

        distance = (
            emergency_plan.distance_to_junction
        )


        eta_trigger = (
            eta is not None
            and eta <=
            EMERGENCY_PREEMPT_ETA_SECONDS
        )


        distance_trigger = (
            distance is not None
            and distance <=
            EMERGENCY_PREEMPT_DISTANCE_METERS
        )


        return bool(
            eta_trigger
            or distance_trigger
        )


    # -----------------------------------------------------------------------
    # DECISION LOGGER
    # -----------------------------------------------------------------------

    def _record_decision(self, record):

        self.decisions.append(
            record
        )

        if self.decision_observer is not None:

            self.decision_observer(
                record
            )


    # -----------------------------------------------------------------------
    # MAIN RUN LOOP
    # -----------------------------------------------------------------------

    def run(self):

        command = list(
            self.sumo_cmd
        )


        if (
            self.emergency_enabled
            and command == SUMO_CMD
        ):

            command = [
                command[0],
                "-c",
                EMERGENCY_SUMO_CONFIG,
            ]


        print(
            "\n"
            + "=" * 70
        )

        print(
            "STARTING AI TRAFFIC CONTROL SYSTEM"
        )

        print(
            "=" * 70
        )

        print(
            f"SUMO command: {command}"
        )

        print(
            f"SUMO config: {SUMO_CONFIG}"
        )

        print(
            f"Emergency mode: "
            f"{self.emergency_enabled}"
        )

        print(
            "=" * 70
        )


        self.traci.start(
            command
        )


        try:

            (
                topology_adapter,
                traffic_controller,
                decision_engine,
                demand_calculator,
                runtimes,
            ) = self._build_runtime()


            phase_adapter = SUMOPhaseAdapter(
                self.traci
            )


            # -----------------------------------------------------------
            # STATE BUILDER
            # -----------------------------------------------------------

            #
            # IMPORTANT:
            # target_junctions comes from SUMO discovery.
            #
            # Therefore the controller does not assume:
            # A0 / B0 / C0
            #
            # It will receive the actual signalized junctions
            # discovered from the new environment.
            #

            sensor_builder = (
                SUMOTrafficStateBuilder(
                    num_sensors=NUM_SENSORS,
                    target_junctions=list(
                        runtimes.keys()
                    ),
                )
            )


            history = TrafficHistory(
                HISTORY_LENGTH,
                NUM_SENSORS,
            )


            safety_adapter = (
                SUMOSafetyRiskAdapter(
                    self.traci
                )
            )


            emergency_manager = (
                EmergencyCorridorManager(
                    self.traci
                )
                if self.emergency_enabled
                else None
            )


            # -----------------------------------------------------------
            # PREDICTION MODEL
            # -----------------------------------------------------------

            predictor = None

            try:

                predictor = (
                    TrafficPredictionAdapter()
                )

                print(
                    "Prediction model: AVAILABLE"
                )

            except Exception as exc:

                print(
                    f"PREDICTION unavailable: "
                    f"{exc}"
                )

                print(
                    "Using safe zero-prediction baseline."
                )


            # -----------------------------------------------------------
            # WARMUP
            # -----------------------------------------------------------

            print(
                f"\nWarmup steps: "
                f"{self.warmup_steps}"
            )

            for _ in range(
                self.warmup_steps
            ):

                self._simulation_step()


            self._observe_steps = True


            predictions = np.zeros(
                NUM_SENSORS,
                dtype=np.float32,
            )


            safety_assessment = None

            last_safety_assessment = (
                -float("inf")
            )


            # ===========================================================
            # MAIN SIMULATION LOOP
            # ===========================================================

            for step in range(
                self.total_steps
            ):

                self._simulation_step()


                now = (
                    self.traci.simulation.getTime()
                )


                # -------------------------------------------------------
                # STATE
                # -------------------------------------------------------

                current_state = (
                    sensor_builder.build_state()
                )

                history.add_state(
                    current_state
                )


                # -------------------------------------------------------
                # SAFETY ASSESSMENT
                # -------------------------------------------------------

                if (
                    now
                    - last_safety_assessment
                    >= self.control_interval
                ):

                    safety_assessment = (
                        safety_adapter.assess()
                    )

                    last_safety_assessment = now


                # -------------------------------------------------------
                # EMERGENCY DISCOVERY + CORRIDOR PLANNING
                # -------------------------------------------------------

                emergency_states = []

                emergency_plans = {}


                if emergency_manager is not None:

                    topology_map = {
                        key: value["topology"]
                        for key, value
                        in runtimes.items()
                    }


                    emergency_states = (
                        emergency_manager.discover(
                            topology_map
                        )
                    )


                    released_emergency_ids = (
                        emergency_manager.release_completed(
                            emergency_states
                        )
                    )


                    for released_vehicle_id in sorted(
                        released_emergency_ids
                    ):

                        print(
                            f"[EMERGENCY_RELEASE] "
                            f"time={now:.1f} "
                            f"vehicle={released_vehicle_id} "
                            "emergency_active=False "
                            "normal_ai_resumed=True"
                        )


                    emergency_plans = (
                        self._update_corridor_plans(
                            emergency_states,
                            topology_map,
                            runtimes,
                            emergency_manager,
                        )
                    )


                # -------------------------------------------------------
                # TRAFFIC PREDICTION
                # -------------------------------------------------------

                if (
                    predictor is not None
                    and step
                    % self.prediction_interval
                    == 0
                ):

                    try:

                        predictions = (
                            predictor.predict(
                                history.get_padded_history()
                            )
                        )

                        predictions = np.asarray(
                            predictions,
                            dtype=np.float32,
                        ).reshape(
                            NUM_SENSORS
                        )

                    except Exception as exc:

                        print(
                            f"[PREDICTION ERROR] "
                            f"{exc}"
                        )

                        predictions = np.zeros(
                            NUM_SENSORS,
                            dtype=np.float32,
                        )


                # =======================================================
                # PER-JUNCTION DECISION
                # =======================================================

                for (
                    junction_id,
                    runtime
                ) in runtimes.items():


                    emergency_plan = (
                        emergency_plans.get(
                            junction_id
                        )
                    )


                    emergency_preempt = (
                        self._emergency_requires_preemption(
                            emergency_plan
                        )
                    )


                    # ---------------------------------------------------
                    # CONTROL RATE LIMIT
                    # ---------------------------------------------------

                    if (
                        not emergency_preempt
                        and
                        now
                        - runtime["last_control"]
                        < self.control_interval
                    ):

                        continue


                    # ===================================================
                    # EMERGENCY PREEMPTION
                    # ===================================================

                    if emergency_preempt:

                        target_phase_id = (
                            emergency_plan.selected_phase
                        )


                        target_phase = (
                            runtime[
                                "phase_by_id"
                            ].get(
                                target_phase_id
                            )
                        )


                        if target_phase is None:

                            print(
                                f"[EMERGENCY_REJECT] "
                                f"time={now:.1f} "
                                f"junction={junction_id} "
                                f"vehicle={emergency_plan.vehicle_id} "
                                "reason=selected_phase_not_available"
                            )

                            continue


                        target_index = (
                            target_phase[
                                "phase_index"
                            ]
                        )


                        current_phase = (
                            phase_adapter.get_current_phase(
                                junction_id
                            )
                        )


                        if (
                            current_phase
                            != target_index
                        ):

                            print(
                                f"[EMERGENCY_PREEMPT] "
                                f"time={now:.1f} "
                                f"junction={junction_id} "
                                f"vehicle={emergency_plan.vehicle_id} "
                                f"distance={emergency_plan.distance_to_junction} "
                                f"ETA={emergency_plan.eta_seconds} "
                                f"required_movement={emergency_plan.required_movement} "
                                f"open_phase={target_phase_id} "
                                "action=OPEN_EARLY"
                            )


                            executed = self._execute(
                                junction_id,
                                target_index,
                                phase_adapter,
                            )


                            runtime[
                                "emergency_hold_phase"
                            ] = target_phase_id


                            runtime[
                                "emergency_vehicle_id"
                            ] = (
                                emergency_plan.vehicle_id
                            )


                            runtime[
                                "last_control"
                            ] = (
                                self.traci.simulation.getTime()
                            )


                            runtime[
                                "last_phase"
                            ] = target_phase_id


                            self._record_decision(
                                {
                                    "time": now,
                                    "junction": junction_id,
                                    "selected": target_phase_id,
                                    "method": "emergency_preemption",
                                    "safety": {
                                        "approved": True,
                                        "mode":
                                            "proactive_emergency_preemption",
                                    },
                                    "emergency_active": True,
                                    "emergency_vehicle_id":
                                        emergency_plan.vehicle_id,
                                    "emergency_eta":
                                        emergency_plan.eta_seconds,
                                    "emergency_priority":
                                        emergency_plan.urgency,
                                    "emergency_distance":
                                        emergency_plan.distance_to_junction,
                                    "executed": executed,
                                }
                            )


                        else:

                            runtime[
                                "emergency_hold_phase"
                            ] = target_phase_id


                            runtime[
                                "emergency_vehicle_id"
                            ] = (
                                emergency_plan.vehicle_id
                            )


                            print(
                                f"[EMERGENCY_HOLD] "
                                f"time={now:.1f} "
                                f"junction={junction_id} "
                                f"vehicle={emergency_plan.vehicle_id} "
                                f"distance={emergency_plan.distance_to_junction} "
                                f"ETA={emergency_plan.eta_seconds} "
                                f"phase={target_phase_id} "
                                "action=KEEP_GREEN"
                            )


                            runtime[
                                "last_control"
                            ] = now


                            runtime[
                                "last_phase"
                            ] = target_phase_id


                        continue


                    # ===================================================
                    # RELEASE EMERGENCY HOLD (corridor-aware)
                    # ===================================================

                    if (
                        runtime[
                            "emergency_hold_phase"
                        ]
                        is not None
                    ):
                        # Check if this junction has been released by
                        # corridor progression.
                        held_vehicle = runtime["emergency_vehicle_id"]
                        junction_released = True

                        if held_vehicle and held_vehicle in self.corridors:
                            corridor = self.corridors[held_vehicle]
                            for jstate in corridor.junctions:
                                if jstate.plan.junction_id == junction_id:
                                    if not jstate.released:
                                        junction_released = False
                                    break

                        if junction_released:
                            print(
                                f"[EMERGENCY_JUNCTION_RELEASE] "
                                f"time={now:.1f} "
                                f"junction={junction_id} "
                                f"vehicle={runtime['emergency_vehicle_id']} "
                                "action=RESUME_NORMAL_AI"
                            )

                            self.corridor_events.append({
                                "time": now,
                                "event": "junction_normal_restored",
                                "ambulance": held_vehicle,
                                "junction": junction_id,
                            })

                            runtime[
                                "emergency_hold_phase"
                            ] = None


                            runtime[
                                "emergency_vehicle_id"
                            ] = None


                    # ===================================================
                    # NORMAL AI DECISION
                    # ===================================================

                    topology = runtime[
                        "topology"
                    ]


                    # ---------------------------------------------------
                    # MOVEMENT DEMAND
                    # ---------------------------------------------------

                    movement_demand = (
                        demand_calculator.calculate_movement_demand(
                            topology,
                            topology_adapter.sumo_metadata,
                        )
                    )


                    phase_demand = (
                        demand_calculator.calculate_phase_demand(
                            runtime[
                                "phase_objects"
                            ],
                            movement_demand,
                        )
                    )


                    normalized_demand = (
                        demand_calculator.normalize_phase_demand(
                            phase_demand
                        )
                    )


                    # ---------------------------------------------------
                    # PREDICTION PER PHASE
                    # ---------------------------------------------------

                    prediction_values = {}


                    for phase in runtime[
                        "phases"
                    ]:

                        ids = _phase_sensor_ids(
                            phase,
                            sensor_builder,
                            topology,
                        )


                        if ids:

                            prediction_values[
                                phase["phase_id"]
                            ] = (
                                decision_engine.calculate_phase_prediction(
                                    predictions,
                                    ids,
                                )
                            )

                        else:

                            prediction_values[
                                phase["phase_id"]
                            ] = 0.0


                    max_prediction = max(
                        prediction_values.values(),
                        default=0.0,
                    )


                    # ---------------------------------------------------
                    # PHASE RISK
                    # ---------------------------------------------------

                    phase_risks = {

                        phase["phase_id"]:
                            safety_adapter.phase_risk(
                                phase,
                                topology,
                                safety_assessment,
                            )

                        for phase
                        in runtime["phases"]
                    }


                    # ---------------------------------------------------
                    # PHASE SCORE
                    # ---------------------------------------------------

                    phase_scores = {}


                    for phase in runtime[
                        "phases"
                    ]:

                        phase_id = (
                            phase["phase_id"]
                        )


                        prediction_component = (

                            prediction_values[
                                phase_id
                            ]
                            / max_prediction

                            if max_prediction > 0
                            else 0.0
                        )


                        phase_scores[
                            phase_id
                        ] = (
                            decision_engine.calculate_phase_score(

                                demand=_clamp(
                                    normalized_demand.get(
                                        phase_id,
                                        0.0,
                                    )
                                ),

                                prediction=_clamp(
                                    prediction_component
                                ),

                                priority=0.5,

                                fairness=0.5,

                                risk=_clamp(
                                    1.0
                                    - phase_risks[
                                        phase_id
                                    ]
                                ),
                            )
                        )


                    # ---------------------------------------------------
                    # SAFETY CHECKS
                    # ---------------------------------------------------

                    checks = {}


                    current_phase = (
                        phase_adapter.get_current_phase(
                            junction_id
                        )
                    )


                    current_type = (
                        phase_adapter.get_phase_type(
                            junction_id,
                            current_phase,
                        )
                    )


                    for phase in runtime[
                        "phases"
                    ]:

                        phase_id = (
                            phase["phase_id"]
                        )


                        compatible = (
                            runtime[
                                "generator"
                            ].is_compatible(
                                phase["movements"]
                            )
                        )


                        timing_valid = (

                            phase["phase_index"]
                            == current_phase

                            or current_type
                            != "green"

                            or self.control_interval
                            >= MIN_GREEN_SECONDS
                        )


                        checks[
                            phase_id
                        ] = {

                            "topology_valid":
                                bool(
                                    topology.roads
                                    and topology.movements
                                ),

                            "conflict_free":
                                compatible,

                            "timing_valid":
                                timing_valid,

                            "pedestrian_clear":
                                True,

                            "downstream_available":
                                True,

                            "emergency_safe":
                                True,

                            # Safety confidence is deliberately kept separate
                            # from the phase utility score. A phase score can be
                            # low simply because traffic demand is low; that
                            # must NOT make an otherwise legal SUMO phase unsafe.
                            # SUMO topology/signal state is authoritative, so
                            # the safety gate receives a state-validity
                            # confidence rather than the AI utility score.
                            "confidence": _clamp(
                                1.0
                                if topology.roads
                                and topology.movements
                                and phase.get("phase_index") is not None
                                else 0.0
                            ),

                            "safety_risk":
                                phase_risks[
                                    phase_id
                                ],

                            "accident_detected":
                                (
                                    safety_assessment[
                                        "accident_status"
                                    ]
                                    == "confirmed"
                                ),

                            "safety_confidence":
                                safety_assessment[
                                    "safety_confidence"
                                ],
                        }


                    # ---------------------------------------------------
                    # CURRENT PHASE INDEX
                    # ---------------------------------------------------

                    phase_ids = list(
                        phase_scores
                    )


                    current_index = next(
                        (
                            i
                            for i, phase
                            in enumerate(
                                runtime[
                                    "phases"
                                ]
                            )
                            if phase[
                                "phase_index"
                            ]
                            == current_phase
                        ),
                        0,
                    )


                    # ---------------------------------------------------
                    # DECISION ENGINE + SAFETY
                    # ---------------------------------------------------

                    result = (
                        traffic_controller.decide_and_validate(

                            phase_scores,

                            checks,

                            phase_densities=
                                phase_demand,

                            phases=
                                phase_ids,

                            current_index=
                                current_index,
                        )
                    )


                    selected = (
                        result[
                            "selected_phase"
                        ]
                    )


                    # ---------------------------------------------------
                    # LOG
                    # ---------------------------------------------------

                    print(
                        f"DECISION "
                        f"time={now:.1f} "
                        f"junction={junction_id} "
                        f"current_phase={current_phase} "
                        f"candidates={phase_ids} "
                        f"movement_demand={movement_demand} "
                        f"phase_demand={phase_demand} "
                        f"prediction_available={predictor is not None} "
                        f"prediction_contribution={prediction_values} "
                        f"risk_score={phase_risks} "
                        f"near_miss_risk={safety_assessment['near_miss_risk']} "
                        f"accident_status={safety_assessment['accident_status']} "
                        f"accident_confidence={safety_assessment['accident_confidence']} "
                        f"safety_confidence={safety_assessment['safety_confidence']} "
                        "emergency_active=False "
                        f"scores={phase_scores} "
                        f"selected={selected} "
                        f"safety="
                        f"{'APPROVED' if result['execute'] else 'REJECTED'} "
                        f"reasons="
                        f"{result['safety'].get('reasons', [])} "
                        f"method={result['method']}"
                    )


                    # ---------------------------------------------------
                    # EXECUTE APPROVED DECISION
                    # ---------------------------------------------------

                    if (
                        result["execute"]
                        and selected is not None
                    ):

                        selected_phase = (
                            runtime[
                                "phase_by_id"
                            ].get(
                                selected
                            )
                        )


                        if selected_phase is None:

                            print(
                                f"[EXECUTION_ERROR] "
                                f"junction={junction_id} "
                                f"selected={selected} "
                                "reason=phase_not_found"
                            )

                            continue


                        executed = self._execute(
                            junction_id,

                            selected_phase[
                                "phase_index"
                            ],

                            phase_adapter,
                        )


                        runtime[
                            "last_control"
                        ] = (
                            self.traci.simulation.getTime()
                        )


                        runtime[
                            "last_phase"
                        ] = selected


                        self._record_decision(
                            {

                                "time":
                                    now,

                                "junction":
                                    junction_id,

                                "selected":
                                    selected,

                                "method":
                                    result[
                                        "method"
                                    ],

                                "safety":
                                    result[
                                        "safety"
                                    ],

                                "emergency_active":
                                    False,

                                "emergency_vehicle_id":
                                    None,

                                "emergency_eta":
                                    None,

                                "emergency_priority":
                                    None,

                                "executed":
                                    executed,
                            }
                        )


            # -----------------------------------------------------------
            # FINISHED
            # -----------------------------------------------------------

            print(
                "\n"
                + "=" * 70
            )

            print(
                "AI TRAFFIC CONTROL RUN COMPLETE"
            )

            print(
                "=" * 70
            )

            print(
                f"Simulation time: "
                f"{self.traci.simulation.getTime():.1f}s"
            )

            print(
                f"Signalized junctions: "
                f"{len(runtimes)}"
            )

            print(
                f"Decisions recorded: "
                f"{len(self.decisions)}"
            )

            print(
                "=" * 70
            )


        finally:

            self.traci.close()

            print(
                "SUMO closed safely"
            )


# ---------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI Traffic Adaptive Signal Controller "
            "for the multimodal SUMO environment."
        )
    )


    parser.add_argument(
        "--emergency",
        action="store_true",
        help="Enable emergency vehicle preemption.",
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
        help="Number of simulation steps.",
    )


    parser.add_argument(
        "--control-interval",
        type=float,
        default=CONTROL_INTERVAL,
        help="AI control interval in seconds.",
    )


    parser.add_argument(
        "--prediction-interval",
        type=int,
        default=PREDICTION_INTERVAL,
        help="Prediction update interval.",
    )


    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_STEPS,
        help="SUMO warmup steps.",
    )


    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO-GUI instead of headless SUMO.",
    )

    args = parser.parse_args()

    sumo_binary = (
        _find_sumo_gui_executable()
        if args.gui
        else _find_sumo_executable()
    )

    sumo_cmd = [
        sumo_binary,
        "-c",
        SUMO_CONFIG,
    ]

    controller = Phase2SUMOController(
        sumo_cmd=sumo_cmd,

        total_steps=args.steps,

        control_interval=
            args.control_interval,

        prediction_interval=
            args.prediction_interval,

        warmup_steps=
            args.warmup,

        emergency_enabled=
            args.emergency,
    )


    controller.run()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()