"""Canonical Phase 2 SUMO/TraCI adaptive signal controller.

The controller deliberately keeps SUMO-specific concerns here. Generic
 decision, safety, fallback, demand, topology and prediction modules remain
unchanged and are composed by this runtime.
"""

import os
import shutil
import sys
import math
import argparse
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import traci

from integration.controller import TrafficController
from integration.decision_engine import DecisionEngine
from integration.emergency import EmergencyCorridorManager
from integration.phase_generator import PhaseGenerator
from integration.prediction_adapter import TrafficPredictionAdapter
from integration.sumo_conflict_adapter import SUMOConflictAdapter
from integration.sumo_demand import SUMODemandCalculator
from integration.sumo_phase_adapter import SUMOPhaseAdapter
from integration.sumo_safety import SUMOSafetyRiskAdapter
from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.traffic_history import TrafficHistory
from integration.sumo_prediction_controller import SUMOTrafficStateBuilder


SUMO_CONFIG = os.path.join(PROJECT_ROOT, "simulation", "configs", "corridor.sumocfg")
EMERGENCY_SUMO_CONFIG = os.path.join(
    PROJECT_ROOT, "simulation", "configs", "corridor_emergency.sumocfg"
)


def _find_sumo_executable():
    """Find SUMO through configuration, PATH, or standard Windows install."""
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe"))
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
                sumo_bin = os.path.dirname(os.path.abspath(candidate))
                os.environ["SUMO_HOME"] = os.path.dirname(sumo_bin)
            return candidate
    return "sumo"


SUMO_CMD = [_find_sumo_executable(), "-c", SUMO_CONFIG]

NUM_SENSORS = 36
HISTORY_LENGTH = 10
CONTROL_INTERVAL = 10.0
PREDICTION_INTERVAL = 10
MIN_GREEN_SECONDS = 10.0
WARMUP_STEPS = 10
TOTAL_STEPS = 200


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _phase_objects(phase_representations):
    """Adapt existing dict representations to SUMODemandCalculator's API."""
    return [
        SimpleNamespace(
            phase_id=phase["phase_id"],
            movement_ids=phase["movements"],
        )
        for phase in phase_representations
    ]


def _phase_sensor_ids(phase, sensor_builder, topology):
    """Map a phase's incoming roads to the deployment sensor zones."""
    roads = {topology.movements[m].from_road for m in phase["movements"]}
    return [
        sensor_id
        for sensor_id, info in sensor_builder.sensor_mapping.items()
        if info["junction_id"] == phase["junction_id"]
        and info["road_id"] in roads
    ]


def _safe_transition_index(program, target_index):
    """Return the transition immediately before a target green phase."""
    phases = program["phases"]
    target_position = next(
        i for i, phase in enumerate(phases)
        if phase["phase_index"] == target_index
    )
    if target_position == 0:
        return target_index
    previous = phases[target_position - 1]
    if "y" in previous["state"].lower() or "r" in previous["state"].lower():
        return previous["phase_index"]
    return target_index


class Phase2SUMOController:
    """Run adaptive decisions independently for each discovered corridor signal."""

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
        self.control_interval = float(control_interval)
        self.prediction_interval = max(1, int(prediction_interval))
        self.warmup_steps = int(warmup_steps)
        self.traci = traci_connection or traci
        self.step_observer = step_observer
        self.emergency_enabled = bool(emergency_enabled)
        self.decision_observer = decision_observer
        self._observe_steps = False
        self.decisions = []

    def _simulation_step(self):
        self.traci.simulationStep()
        if self.step_observer is not None and self._observe_steps:
            self.step_observer()

    def _build_runtime(self):
        topology_adapter = SUMOTopologyAdapter(self.traci)
        phase_adapter = SUMOPhaseAdapter(self.traci)
        conflict_adapter = SUMOConflictAdapter(phase_adapter)
        demand_calculator = SUMODemandCalculator()
        traffic_controller = TrafficController()
        decision_engine = DecisionEngine()

        signalized_ids = topology_adapter.discover_signalized_junctions()
        topologies = {
            junction_id: topology_adapter.discover_junction(junction_id)
            for junction_id in signalized_ids
        }
        print(f"Signalized corridor junctions: {signalized_ids}")

        runtimes = {}
        for junction_id, topology in topologies.items():
            conflict_graph = conflict_adapter.build_conflict_graph(
                junction_id, topology, topology_adapter.sumo_metadata
            )
            generator = PhaseGenerator(conflict_graph)
            generated = generator.generate_maximal_phases()
            representations = phase_adapter.build_phase_representation(
                junction_id, topology_adapter.sumo_metadata
            )
            if not representations:
                continue
            runtimes[junction_id] = {
                "topology": topology,
                "generator": generator,
                "generated": generated,
                "phases": representations,
                "phase_objects": _phase_objects(representations),
                "phase_by_id": {p["phase_id"]: p for p in representations},
                "last_control": -float("inf"),
                "last_phase": None,
            }
            print(
                f"Topology: {junction_id} roads={len(topology.roads)} "
                f"movements={len(topology.movements)} "
                f"generated_maximal_phases={len(generated)}"
            )

        if not runtimes:
            raise RuntimeError("No usable green signal phases were discovered")

        return (
            topology_adapter,
            traffic_controller,
            decision_engine,
            demand_calculator,
            runtimes,
        )

    def _execute(self, junction_id, target_phase, phase_adapter):
        current = phase_adapter.get_current_phase(junction_id)
        current_state = phase_adapter.get_phase_state(junction_id, current)
        program = phase_adapter.get_current_program(junction_id)
        transition = _safe_transition_index(program, target_phase)

        print(
            f"EXECUTE time={self.traci.simulation.getTime():.1f} "
            f"junction={junction_id} current_phase={current} "
            f"current_state={current_state} target_phase={target_phase} "
            f"transition_phase={transition}"
        )

        # Do not restart an already active green. Otherwise request the
        # preceding yellow/all-red phase so SUMO owns the safe transition.
        if current != target_phase:
            self.traci.trafficlight.setPhase(junction_id, transition)

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
                int(math.ceil(float(transition_duration))) + 1,
            )
            for _ in range(max_transition_steps):
                self._simulation_step()
                if phase_adapter.get_current_phase(junction_id) == target_phase:
                    break

        executed = phase_adapter.get_current_phase(junction_id)
        executed_state = phase_adapter.get_phase_state(junction_id, executed)
        print(
            f"READBACK junction={junction_id} executed_phase={executed} "
            f"resulting_state={executed_state}"
        )
        return {"phase": executed, "state": executed_state, "requested": target_phase}

    def run(self):
        command = list(self.sumo_cmd)
        if self.emergency_enabled and command == SUMO_CMD:
            command = [command[0], "-c", EMERGENCY_SUMO_CONFIG]
        print(f"Starting SUMO: {command[2] if len(command) > 2 else command}")
        self.traci.start(command)
        try:
            (
                topology_adapter,
                traffic_controller,
                decision_engine,
                demand_calculator,
                runtimes,
            ) = self._build_runtime()
            phase_adapter = SUMOPhaseAdapter(self.traci)
            sensor_builder = SUMOTrafficStateBuilder(
                num_sensors=NUM_SENSORS,
                target_junctions=list(runtimes),
            )
            history = TrafficHistory(HISTORY_LENGTH, NUM_SENSORS)
            safety_adapter = SUMOSafetyRiskAdapter(self.traci)
            emergency_manager = (
                EmergencyCorridorManager(self.traci)
                if self.emergency_enabled else None
            )
            predictor = None
            try:
                predictor = TrafficPredictionAdapter()
            except Exception as exc:
                print(f"PREDICTION unavailable: {exc}; using safe baseline")

            for _ in range(self.warmup_steps):
                self._simulation_step()

            self._observe_steps = True

            predictions = np.zeros(NUM_SENSORS, dtype=np.float32)
            safety_assessment = None
            last_safety_assessment = -float("inf")
            for step in range(self.total_steps):
                self._simulation_step()
                now = self.traci.simulation.getTime()
                history.add_state(sensor_builder.build_state())
                if now - last_safety_assessment >= self.control_interval:
                    safety_assessment = safety_adapter.assess()
                    last_safety_assessment = now

                emergency_states = []
                emergency_plans = {}
                if emergency_manager is not None:
                    topology_map = {
                        key: value["topology"]
                        for key, value in runtimes.items()
                    }
                    emergency_states = emergency_manager.discover(topology_map)
                    released_emergency_ids = emergency_manager.release_completed(
                        emergency_states
                    )
                    for released_vehicle_id in sorted(released_emergency_ids):
                        print(
                            f"[EMERGENCY_RELEASE] time={now:.1f} "
                            f"vehicle={released_vehicle_id} "
                            "emergency_active=False normal_ai_resumed=True"
                        )
                    emergency_plans = emergency_manager.plans(
                        emergency_states,
                        topology_map,
                        runtimes,
                    )

                if predictor is not None and step % self.prediction_interval == 0:
                    predictions = predictor.predict(history.get_padded_history())
                    predictions = np.asarray(predictions, dtype=np.float32).reshape(NUM_SENSORS)

                for junction_id, runtime in runtimes.items():
                    if now - runtime["last_control"] < self.control_interval:
                        continue

                    topology = runtime["topology"]
                    movement_demand = demand_calculator.calculate_movement_demand(
                        topology, topology_adapter.sumo_metadata
                    )
                    phase_demand = demand_calculator.calculate_phase_demand(
                        runtime["phase_objects"], movement_demand
                    )
                    normalized_demand = demand_calculator.normalize_phase_demand(phase_demand)

                    prediction_values = {}
                    for phase in runtime["phases"]:
                        ids = _phase_sensor_ids(phase, sensor_builder, topology)
                        prediction_values[phase["phase_id"]] = (
                            decision_engine.calculate_phase_prediction(predictions, ids)
                            if ids else 0.0
                        )
                    max_prediction = max(prediction_values.values(), default=0.0)
                    phase_risks = {
                        phase["phase_id"]: safety_adapter.phase_risk(
                            phase, topology, safety_assessment
                        )
                        for phase in runtime["phases"]
                    }

                    phase_scores = {}
                    emergency_plan = emergency_plans.get(junction_id)
                    for phase in runtime["phases"]:
                        phase_id = phase["phase_id"]
                        emergency_priority = (
                            emergency_plan.urgency
                            if emergency_plan is not None
                            and phase_id == emergency_plan.selected_phase
                            else 0.0
                        )
                        phase_scores[phase_id] = decision_engine.calculate_phase_score(
                            demand=_clamp(normalized_demand.get(phase_id, 0.0)),
                            prediction=_clamp(
                                prediction_values[phase_id] / max_prediction
                                if max_prediction > 0 else 0.0
                            ),
                            # Phase 2 baseline: no priority/fairness model yet.
                            priority=(
                                emergency_priority
                                if emergency_plan is not None else 0.5
                            ),
                            fairness=0.5,
                            # DecisionEngine's risk input is a safety factor:
                            # high observed risk lowers the phase score.
                            risk=_clamp(1.0 - phase_risks[phase_id]),
                        )

                    checks = {}
                    current_phase = phase_adapter.get_current_phase(junction_id)
                    current_type = phase_adapter.get_phase_type(junction_id, current_phase)
                    for phase in runtime["phases"]:
                        phase_id = phase["phase_id"]
                        compatible = runtime["generator"].is_compatible(phase["movements"])
                        timing_valid = (
                            phase["phase_index"] == current_phase
                            or current_type != "green"
                            or self.control_interval >= MIN_GREEN_SECONDS
                        )
                        checks[phase_id] = {
                            "topology_valid": bool(topology.roads and topology.movements),
                            "conflict_free": compatible,
                            "timing_valid": timing_valid,
                            # Phase-2 assumptions: these inputs are not sensed yet.
                            "pedestrian_clear": True,
                            "downstream_available": True,
                            "emergency_safe": True,
                            "confidence": _clamp(phase_scores[phase_id]),
                            "safety_risk": phase_risks[phase_id],
                            "accident_detected": (
                                safety_assessment["accident_status"] == "confirmed"
                            ),
                            "safety_confidence": safety_assessment["safety_confidence"],
                        }

                    if emergency_plan is not None and emergency_plan.selected_phase:
                        phase_scores[emergency_plan.selected_phase] = max(
                            phase_scores.values()
                        ) + (0.01 * emergency_plan.urgency)

                    phase_ids = list(phase_scores)
                    current_index = next(
                        (
                            i for i, p in enumerate(runtime["phases"])
                            if p["phase_index"] == current_phase
                        ),
                        0,
                    )
                    result = traffic_controller.decide_and_validate(
                        phase_scores,
                        checks,
                        phase_densities=phase_demand,
                        phases=phase_ids,
                        current_index=current_index,
                    )
                    selected = result["selected_phase"]
                    print(
                        f"DECISION time={now:.1f} junction={junction_id} "
                        f"current_phase={current_phase} candidates={phase_ids} "
                        f"movement_demand={movement_demand} phase_demand={phase_demand} "
                        f"prediction_available={predictor is not None} "
                        f"prediction_contribution={prediction_values} "
                        f"risk_score={phase_risks} "
                        f"near_miss_risk={safety_assessment['near_miss_risk']} "
                        f"accident_status={safety_assessment['accident_status']} "
                        f"accident_confidence={safety_assessment['accident_confidence']} "
                        f"safety_confidence={safety_assessment['safety_confidence']} "
                        f"emergency_active={emergency_plan is not None} "
                        f"scores={phase_scores} selected={selected} "
                        f"safety={'APPROVED' if result['execute'] else 'REJECTED'} "
                        f"reasons={result['safety'].get('reasons', [])} method={result['method']}"
                    )
                    if emergency_plan is not None:
                        print(
                            f"[EMERGENCY] time={now:.1f} "
                            f"vehicle={emergency_plan.vehicle_id} "
                            f"junction={junction_id} "
                            f"current_edge={emergency_plan.current_edge} "
                            f"next_edge={emergency_plan.next_edge} "
                            f"required_movement={emergency_plan.required_movement} "
                            f"distance={emergency_plan.distance_to_junction} "
                            f"speed={emergency_plan.vehicle_speed_mps} "
                            f"ETA={emergency_plan.eta_seconds} "
                            f"urgency={emergency_plan.urgency:.3f} "
                            f"candidate_phases={emergency_plan.candidate_phases} "
                            f"selected_phase={selected} "
                            f"safe={emergency_plan.safe} "
                            f"fallback={result['method']}"
                        )
                    if result["execute"] and selected is not None:
                        executed = self._execute(
                            junction_id,
                            runtime["phase_by_id"][selected]["phase_index"],
                            phase_adapter,
                        )
                        runtime["last_control"] = self.traci.simulation.getTime()
                        runtime["last_phase"] = selected
                        decision_record = {
                            "time": now,
                            "junction": junction_id,
                            "selected": selected,
                            "method": result["method"],
                            "safety": result["safety"],
                            "emergency_active": emergency_plan is not None,
                            "emergency_vehicle_id": (
                                emergency_plan.vehicle_id
                                if emergency_plan is not None else None
                            ),
                            "emergency_eta": (
                                emergency_plan.eta_seconds
                                if emergency_plan is not None else None
                            ),
                            "emergency_priority": (
                                emergency_plan.urgency
                                if emergency_plan is not None else None
                            ),
                            "executed": executed,
                        }
                        self.decisions.append(decision_record)
                        if self.decision_observer is not None:
                            self.decision_observer(decision_record)
        finally:
            self.traci.close()
            print("SUMO closed safely")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emergency", action="store_true")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    Phase2SUMOController(
        total_steps=args.steps,
        emergency_enabled=args.emergency,
    ).run()


if __name__ == "__main__":
    main()
