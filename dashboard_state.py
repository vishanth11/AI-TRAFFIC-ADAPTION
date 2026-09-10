"""Thread-safe dashboard state — updated by the existing runtime, read by the API."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DashboardState:
    """Single shared state object between the traffic runtime and the FastAPI layer."""

    def __init__(self, max_logs: int = 200, max_metric_history: int = 300):
        self._lock = threading.Lock()

        # System
        self.system_status: str = "stopped"          # running | stopped | error
        self.simulation_time: float = 0.0
        self.simulation_mode: str = "normal"          # normal | emergency_demo
        self.prediction_available: bool = False
        self.started_at: str | None = None
        self.updated_at: str = _utc_now()

        # Junctions — dict[junction_id, dict]
        self.junctions: dict[str, dict] = {}

        # Vehicles
        self.vehicle_count: int = 0

        # Latest AI decisions — list of dicts (most recent first, capped at 100)
        self.decisions: deque = deque(maxlen=100)

        # Safety assessment (latest)
        self.safety_assessment: dict = {}

        # Emergency vehicles — list of dicts
        self.emergency_vehicles: list[dict] = []

        # Risk
        self.risk_level: str = "UNKNOWN"
        self.risk_score: float = 0.0
        self.near_miss_risk: float = 0.0
        self.safety_confidence: float = 0.0
        self.accident_status: str = "unavailable_no_camera_frame_or_model"
        self.pair_risks: list[dict] = []

        # Metrics (live)
        self.metrics: dict = {}

        # Metric history for charts — list of {time, value}
        self.waiting_time_history: deque = deque(maxlen=max_metric_history)
        self.queue_history: deque = deque(maxlen=max_metric_history)
        self.throughput_history: deque = deque(maxlen=max_metric_history)

        # Structured runtime logs
        self.logs: deque = deque(maxlen=max_logs)

        # Comparison results (from evaluation runs)
        self.comparison_results: list[dict] = []

    # ------------------------------------------------------------------
    # Update helpers — called from the runtime thread
    # ------------------------------------------------------------------

    def update_system(self, status: str, sim_time: float, mode: str = "normal",
                      prediction_available: bool = False):
        with self._lock:
            self.system_status = status
            self.simulation_time = sim_time
            self.simulation_mode = mode
            self.prediction_available = prediction_available
            if status == "running" and self.started_at is None:
                self.started_at = _utc_now()
            self.updated_at = _utc_now()

    def update_junction(self, junction_id: str, data: dict):
        with self._lock:
            self.junctions[junction_id] = {**data, "junction_id": junction_id}
            self.updated_at = _utc_now()

    def update_vehicles(self, count: int):
        with self._lock:
            self.vehicle_count = count
            self.updated_at = _utc_now()

    def add_decision(self, record: dict):
        with self._lock:
            self.decisions.appendleft(record)
            self.updated_at = _utc_now()

    def update_safety(self, assessment: dict):
        with self._lock:
            self.safety_assessment = assessment
            self.risk_score = float(assessment.get("risk_score", 0.0))
            self.near_miss_risk = float(assessment.get("near_miss_risk", 0.0))
            self.safety_confidence = float(assessment.get("safety_confidence", 0.0))
            self.accident_status = assessment.get(
                "accident_status", "unavailable_no_camera_frame_or_model"
            )
            self.pair_risks = assessment.get("pair_risks", [])
            # Derive risk level from score
            if self.risk_score >= 0.75:
                self.risk_level = "CRITICAL"
            elif self.risk_score >= 0.50:
                self.risk_level = "HIGH"
            elif self.risk_score >= 0.25:
                self.risk_level = "MEDIUM"
            else:
                self.risk_level = "LOW"
            self.updated_at = _utc_now()

    def update_emergencies(self, vehicles: list[dict]):
        with self._lock:
            self.emergency_vehicles = vehicles
            self.updated_at = _utc_now()

    def update_metrics(self, metrics: dict, sim_time: float):
        with self._lock:
            self.metrics = metrics
            t = sim_time
            if metrics.get("average_waiting_time") is not None:
                self.waiting_time_history.append(
                    {"time": t, "value": metrics["average_waiting_time"]}
                )
            if metrics.get("average_queue_length") is not None:
                self.queue_history.append(
                    {"time": t, "value": metrics["average_queue_length"]}
                )
            if metrics.get("throughput") is not None:
                self.throughput_history.append(
                    {"time": t, "value": metrics["throughput"]}
                )
            self.updated_at = _utc_now()

    def add_log(self, event: str, **fields):
        with self._lock:
            entry = {
                "timestamp": _utc_now(),
                "simulation_time": self.simulation_time,
                "event": event,
                **fields,
            }
            self.logs.appendleft(entry)

    def set_comparison_results(self, results: list[dict]):
        with self._lock:
            self.comparison_results = results

    # ------------------------------------------------------------------
    # Snapshot helpers — called from the API thread
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "system_status": self.system_status,
                "simulation_time": self.simulation_time,
                "simulation_mode": self.simulation_mode,
                "prediction_available": self.prediction_available,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "junction_count": len(self.junctions),
                "vehicle_count": self.vehicle_count,
                "emergency_count": len(self.emergency_vehicles),
                "risk_level": self.risk_level,
                "risk_score": self.risk_score,
                "near_miss_risk": self.near_miss_risk,
                "safety_confidence": self.safety_confidence,
                "accident_status": self.accident_status,
                "current_ai_method": self._latest_method(),
            }

    def _latest_method(self) -> str | None:
        if not self.decisions:
            return None
        return self.decisions[0].get("method")

    def junctions_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.junctions.values())

    def junction_snapshot(self, junction_id: str) -> dict | None:
        with self._lock:
            return self.junctions.get(junction_id)

    def decisions_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.decisions)

    def safety_snapshot(self) -> dict:
        with self._lock:
            return {
                "assessment": dict(self.safety_assessment),
                "risk_level": self.risk_level,
                "risk_score": self.risk_score,
                "near_miss_risk": self.near_miss_risk,
                "safety_confidence": self.safety_confidence,
                "accident_status": self.accident_status,
                "pair_risks": list(self.pair_risks),
            }

    def emergencies_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.emergency_vehicles)

    def metrics_snapshot(self) -> dict:
        with self._lock:
            return {
                "current": dict(self.metrics),
                "waiting_time_history": list(self.waiting_time_history),
                "queue_history": list(self.queue_history),
                "throughput_history": list(self.throughput_history),
            }

    def logs_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.logs)

    def comparison_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.comparison_results)


# Module-level singleton used by both the runtime and the API
state = DashboardState()
