"""Reproducible Phase 2.1 SUMO controller evaluation.

The four evaluated controllers are intentionally kept separate:

* fixed_time: SUMO's configured signal program, with no intervention;
* density_based: highest current movement demand phase;
* predictive_adaptive: highest phase-specific GNN prediction;
* integrated_ai: the verified Phase2SUMOController.

Every run uses a fresh SUMO process, the same network/configuration, route
file, simulation window, and explicit seed. Metrics are collected directly
from TraCI and are never substituted when a run fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import traci

from integration.decision_engine import DecisionEngine
from integration.fallback_controller import FallbackController
from integration.phase_generator import PhaseGenerator
from integration.prediction_adapter import TrafficPredictionAdapter
from integration.safety_gate import SafetyGate
from integration.sumo_conflict_adapter import SUMOConflictAdapter
from integration.sumo_decision_controller import (
    HISTORY_LENGTH,
    NUM_SENSORS,
    Phase2SUMOController,
    _phase_objects,
    _phase_sensor_ids,
)
from integration.sumo_demand import SUMODemandCalculator
from integration.sumo_phase_adapter import SUMOPhaseAdapter
from integration.sumo_topology_adapter import SUMOTopologyAdapter
from integration.traffic_history import TrafficHistory
from integration.sumo_prediction_controller import SUMOTrafficStateBuilder


CONTROLLERS = (
    "fixed_time",
    "density_based",
    "predictive_adaptive",
    "integrated_ai",
)
METRICS = (
    "average_waiting_time",
    "total_waiting_time",
    "maximum_waiting_time",
    "average_queue_length",
    "maximum_queue_length",
    "throughput",
    "average_travel_time",
    "total_travel_time",
    "number_of_stops",
    "fuel_consumption_mg",
    "co2_emissions_mg",
)


class MetricAccumulator:
    """Collect raw TraCI measurements and calculate documented metrics."""

    def __init__(self, traci_module=traci):
        self.traci = traci_module
        self.departure_time = {}
        self.last_waiting = {}
        self.last_speed = {}
        self.completed_waiting = []
        self.completed_travel = []
        self.queue_samples = []
        self.total_stops = 0
        self.total_fuel_mg = 0.0
        self.total_co2_mg = 0.0
        self.total_distance_m = 0.0
        self.samples = 0
        self.max_queue = 0
        self.max_waiting = 0.0

    def record_sample(self, queue_length, fuel_mg=0.0, co2_mg=0.0, stops=0, distance_m=0.0):
        """Record one simulation-timestep observation for unit tests/runtime."""
        queue_length = int(queue_length)
        self.queue_samples.append(queue_length)
        self.max_queue = max(self.max_queue, queue_length)
        self.total_fuel_mg += float(fuel_mg)
        self.total_co2_mg += float(co2_mg)
        self.total_stops += int(stops)
        self.total_distance_m += float(distance_m)
        self.samples += 1

    def record_vehicle_completion(self, travel_time, waiting_time):
        self.completed_travel.append(float(travel_time))
        self.completed_waiting.append(float(waiting_time))
        self.max_waiting = max(self.max_waiting, float(waiting_time))

    @staticmethod
    def _vehicle_value(getter, vehicle_id, default=0.0):
        try:
            return float(getter(vehicle_id))
        except Exception:
            return float(default)

    def sample(self):
        """Sample active vehicles and arrivals after one TraCI step."""
        now = float(self.traci.simulation.getTime())
        try:
            delta_t = float(self.traci.simulation.getDeltaT()) / 1000.0
        except Exception:
            delta_t = 1.0

        departed = list(self.traci.simulation.getDepartedIDList())
        for vehicle_id in departed:
            self.departure_time.setdefault(vehicle_id, now)

        active_ids = list(self.traci.vehicle.getIDList())
        queue_length = 0
        fuel = 0.0
        co2 = 0.0
        distance = 0.0
        stops = 0

        for vehicle_id in active_ids:
            self.departure_time.setdefault(vehicle_id, now)
            speed = self._vehicle_value(self.traci.vehicle.getSpeed, vehicle_id)
            waiting = self._vehicle_value(
                self.traci.vehicle.getAccumulatedWaitingTime,
                vehicle_id,
            )
            self.last_waiting[vehicle_id] = waiting
            self.max_waiting = max(self.max_waiting, waiting)
            if speed <= 0.1:
                queue_length += 1
            if vehicle_id in self.last_speed and self.last_speed[vehicle_id] > 0.1 and speed <= 0.1:
                stops += 1
            self.last_speed[vehicle_id] = speed
            fuel += self._vehicle_value(self.traci.vehicle.getFuelConsumption, vehicle_id) * delta_t
            co2 += self._vehicle_value(self.traci.vehicle.getCO2Emission, vehicle_id) * delta_t
            distance += max(0.0, speed * delta_t)

        self.record_sample(queue_length, fuel, co2, stops, distance)

        for vehicle_id in self.traci.simulation.getArrivedIDList():
            depart = self.departure_time.pop(vehicle_id, now)
            self.record_vehicle_completion(
                travel_time=max(0.0, now - depart),
                waiting_time=self.last_waiting.pop(vehicle_id, 0.0),
            )
            self.last_speed.pop(vehicle_id, None)

    def finalize(self):
        completed = len(self.completed_travel)
        return {
            "average_waiting_time": (
                sum(self.completed_waiting) / completed if completed else None
            ),
            "total_waiting_time": sum(self.completed_waiting),
            "maximum_waiting_time": self.max_waiting,
            "average_queue_length": (
                sum(self.queue_samples) / len(self.queue_samples)
                if self.queue_samples else None
            ),
            "maximum_queue_length": self.max_queue,
            "throughput": completed,
            "average_travel_time": (
                sum(self.completed_travel) / completed if completed else None
            ),
            "total_travel_time": sum(self.completed_travel),
            "number_of_stops": self.total_stops,
            "fuel_consumption_mg": self.total_fuel_mg,
            "co2_emissions_mg": self.total_co2_mg,
            "total_distance_m": self.total_distance_m,
            "completed_vehicles": completed,
            "measurement_samples": self.samples,
        }


def _sumo_command(seed, config):
    executable = Phase2SUMOController().sumo_cmd[0]
    return [executable, "-c", str(config), "--seed", str(int(seed))]


def _runtime_metadata(connection):
    topology_adapter = SUMOTopologyAdapter(connection)
    phase_adapter = SUMOPhaseAdapter(connection)
    conflict_adapter = SUMOConflictAdapter(phase_adapter)
    runtimes = {}
    for junction_id in topology_adapter.discover_signalized_junctions():
        topology = topology_adapter.discover_junction(junction_id)
        graph = conflict_adapter.build_conflict_graph(
            junction_id, topology, topology_adapter.sumo_metadata
        )
        representations = phase_adapter.build_phase_representation(
            junction_id, topology_adapter.sumo_metadata
        )
        runtimes[junction_id] = {
            "topology": topology,
            "generator": PhaseGenerator(graph),
            "phases": representations,
            "phase_objects": _phase_objects(representations),
            "phase_by_id": {p["phase_id"]: p for p in representations},
            "last_control": -float("inf"),
        }
    return topology_adapter, phase_adapter, runtimes


def _safe_baseline_phase(junction_id, phase_id, runtime, phase_adapter, now, last_control):
    phase = runtime["phase_by_id"][phase_id]
    current = phase_adapter.get_current_phase(junction_id)
    current_type = phase_adapter.get_phase_type(junction_id, current)
    timing_valid = (
        phase["phase_index"] == current
        or current_type != "green"
        or now - last_control >= 10.0
    )
    safety = SafetyGate(min_green_seconds=10.0).validate(
        phase_id,
        topology_valid=bool(runtime["topology"].roads and runtime["topology"].movements),
        conflict_free=runtime["generator"].is_compatible(phase["movements"]),
        timing_valid=timing_valid,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=1.0,
    )
    return safety


def _select_density_phase(runtime, demand, current_phase):
    values = {
        phase.phase_id: sum(demand.get(movement_id, 0) for movement_id in phase.movement_ids)
        for phase in runtime["phase_objects"]
    }
    selected = max(values, key=values.get)
    return selected, values


def _select_predictive_phase(runtime, predictions, sensor_builder):
    values = {}
    for phase in runtime["phases"]:
        sensor_ids = _phase_sensor_ids(phase, sensor_builder, runtime["topology"])
        values[phase["phase_id"]] = (
            float(np.mean(np.maximum(predictions[sensor_ids], 0.0)))
            if sensor_ids else 0.0
        )
    return max(values, key=values.get), values


def _run_baseline(controller_name, seed, config, steps, warmup, control_interval):
    metrics = MetricAccumulator()
    traci.start(_sumo_command(seed, config))
    try:
        topology_adapter, phase_adapter, runtimes = _runtime_metadata(traci)
        if controller_name == "predictive_adaptive":
            sensor_builder = SUMOTrafficStateBuilder(
                num_sensors=NUM_SENSORS,
                target_junctions=list(runtimes),
            )
            history = TrafficHistory(HISTORY_LENGTH, NUM_SENSORS)
            predictor = TrafficPredictionAdapter()
            predictions = np.zeros(NUM_SENSORS, dtype=np.float32)
        else:
            sensor_builder = None
            history = None
            predictor = None
            predictions = None

        executor = Phase2SUMOController(
            traci_connection=traci,
            step_observer=metrics.sample,
        )
        executor._observe_steps = True
        last_control = {junction_id: -float("inf") for junction_id in runtimes}

        for _ in range(int(warmup)):
            traci.simulationStep()

        for step in range(int(steps)):
            traci.simulationStep()
            metrics.sample()
            now = float(traci.simulation.getTime())
            if history is not None:
                history.add_state(sensor_builder.build_state())
            if controller_name == "fixed_time":
                continue
            if predictor is not None and step % 10 == 0:
                predictions = np.asarray(
                    predictor.predict(history.get_padded_history()),
                    dtype=np.float32,
                )
            for junction_id, runtime in runtimes.items():
                if now - last_control[junction_id] < control_interval:
                    continue
                demand = SUMODemandCalculator().calculate_movement_demand(
                    runtime["topology"], topology_adapter.sumo_metadata
                )
                if controller_name == "density_based":
                    selected, values = _select_density_phase(
                        runtime, demand, phase_adapter.get_current_phase(junction_id)
                    )
                else:
                    selected, values = _select_predictive_phase(
                        runtime, predictions, sensor_builder
                    )
                safety = _safe_baseline_phase(
                    junction_id,
                    selected,
                    runtime,
                    phase_adapter,
                    now,
                    last_control[junction_id],
                )
                if safety["approved"]:
                    executor._execute(
                        junction_id,
                        runtime["phase_by_id"][selected]["phase_index"],
                        phase_adapter,
                    )
                    last_control[junction_id] = float(traci.simulation.getTime())
        return metrics.finalize()
    finally:
        traci.close()


def _run_integrated(seed, config, steps, warmup, control_interval):
    metrics = MetricAccumulator()
    controller = Phase2SUMOController(
        sumo_cmd=_sumo_command(seed, config),
        total_steps=steps,
        warmup_steps=warmup,
        control_interval=control_interval,
        step_observer=metrics.sample,
    )
    controller.run()
    return metrics.finalize()


def run_one(controller_name, seed, config, steps=200, warmup=10, control_interval=10.0):
    """Run exactly one controller/seed pair and preserve failures."""
    started = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        if controller_name == "integrated_ai":
            metrics = _run_integrated(seed, config, steps, warmup, control_interval)
        else:
            metrics = _run_baseline(
                controller_name, seed, config, steps, warmup, control_interval
            )
        return {
            "controller": controller_name,
            "seed": int(seed),
            "status": "success",
            "started_at": started,
            **metrics,
        }
    except Exception as exc:
        try:
            traci.close()
        except Exception:
            pass
        return {
            "controller": controller_name,
            "seed": int(seed),
            "status": "failed",
            "started_at": started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def aggregate_results(results):
    summary = []
    grouped = defaultdict(list)
    for result in results:
        if result.get("status") == "success":
            grouped[result["controller"]].append(result)
    for controller in CONTROLLERS:
        runs = grouped.get(controller, [])
        row = {"controller": controller, "successful_runs": len(runs)}
        for metric in METRICS:
            values = [r[metric] for r in runs if r.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(row)
    return summary


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path, summary):
    labels = {
        "fixed_time": "Fixed-Time",
        "density_based": "Density-Based",
        "predictive_adaptive": "Predictive Adaptive",
        "integrated_ai": "Integrated AI",
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase 2.1 Controller Evaluation\n\n")
        handle.write("| Metric | Fixed-Time | Density-Based | Predictive Adaptive | Integrated AI |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        by_controller = {row["controller"]: row for row in summary}
        for metric in METRICS:
            values = []
            for controller in CONTROLLERS:
                row = by_controller[controller]
                value = row.get(f"{metric}_mean")
                deviation = row.get(f"{metric}_std")
                if value is None:
                    values.append("unavailable")
                elif deviation:
                    values.append(f"{value:.3f} ± {deviation:.3f}")
                else:
                    values.append(f"{value:.3f}")
            label = metric.replace("_", " ").title()
            handle.write(f"| {label} | " + " | ".join(values) + " |\n")
        handle.write("\nMetrics are raw SUMO/TraCI measurements; no unavailable values were fabricated.\n")


def _write_plots(output_dir, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plots_unavailable.txt").write_text(str(exc), encoding="utf-8")
        return False
    labels = ["Fixed-Time", "Density-Based", "Predictive", "Integrated AI"]
    for metric in (
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_travel_time",
        "fuel_consumption_mg",
        "co2_emissions_mg",
    ):
        values = [
            row.get(f"{metric}_mean") or 0.0
            for row in summary
        ]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(labels, values)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("SUMO units")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}.png", dpi=140)
        plt.close(figure)
    return True


def run_evaluation(output_dir, seeds=(1,), config=None, steps=200, warmup=10, control_interval=10.0, make_plots=True):
    config = Path(config or PROJECT_ROOT / "simulation" / "configs" / "corridor.sumocfg")
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summaries"
    plots_dir = output_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        for controller_name in CONTROLLERS:
            print(f"\n=== {controller_name} seed={seed} ===")
            result = run_one(controller_name, seed, config, steps, warmup, control_interval)
            results.append(result)
            if result["status"] == "failed":
                print(f"FAILED: {result['error']}")
            else:
                print(f"completed: throughput={result['throughput']}")
            (raw_dir / f"{controller_name}_seed{seed}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

    summary = aggregate_results(results)
    _write_csv(output_dir / "raw_results.csv", results)
    _write_csv(summary_dir / "summary.csv", summary)
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_dir / "summary.md", summary)
    plots_created = _write_plots(plots_dir, summary) if make_plots else False
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "config": str(config),
            "seeds": list(seeds),
            "controllers": list(CONTROLLERS),
            "steps": steps,
            "warmup_steps": warmup,
            "control_interval": control_interval,
            "plots_created": plots_created,
            "metric_definitions": {
                "average_waiting_time": "completed-vehicle accumulated waiting time mean",
                "total_waiting_time": "completed-vehicle accumulated waiting time sum",
                "maximum_waiting_time": "maximum observed accumulated waiting time",
                "average_queue_length": "time-average active vehicles at speed <= 0.1 m/s",
                "maximum_queue_length": "maximum active vehicles at speed <= 0.1 m/s",
                "throughput": "number of arrived vehicles",
                "average_travel_time": "mean arrival time minus departure time",
                "total_travel_time": "sum of completed-vehicle travel times",
                "number_of_stops": "speed transition from >0.1 to <=0.1 m/s",
                "fuel_consumption_mg": "time-integrated SUMO fuel rate",
                "co2_emissions_mg": "time-integrated SUMO CO2 rate",
            },
        }, indent=2), encoding="utf-8"
    )
    return results, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Phase 2.1 SUMO evaluation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--control-interval", type=float, default=10.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or str(
        PROJECT_ROOT / "results" / "evaluation" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    results, summary = run_evaluation(
        output_dir,
        seeds=args.seeds,
        steps=args.steps,
        warmup=args.warmup,
        control_interval=args.control_interval,
        make_plots=not args.no_plots,
    )
    print(f"\nResults: {output_dir}")
    print(f"Runs: {len(results)}; successful: {sum(r['status'] == 'success' for r in results)}")
    print((Path(output_dir) / "summaries" / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
