"""SUMO telemetry adapter for Phase 2.2 safety intelligence.

This adapter reuses the project's calibrated near-miss feature/model and
SafetyIntelligence risk fusion. SUMO does not provide camera frames, so the
image-based accident detector is only loaded/status-checked here; it is never
fed a fabricated image. Accident inference remains explicitly unavailable
until a real image provider is connected.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np

try:
    from integration.near_miss_feature_adapter import NearMissFeatureAdapter
except ModuleNotFoundError:
    from src.integration.near_miss_feature_adapter import NearMissFeatureAdapter
from src.safety_intelligence import SafetyIntelligence


class SUMOSafetyRiskAdapter:
    """Build conservative, phase-specific safety risk from SUMO telemetry."""

    def __init__(
        self,
        traci_connection,
        safety_intelligence=None,
        max_track_points=3,
        max_pairs=4,
    ):
        self.traci = traci_connection
        self.safety = safety_intelligence or SafetyIntelligence()
        self.feature_adapter = NearMissFeatureAdapter()
        self.max_track_points = max(3, int(max_track_points))
        self.max_pairs = max(1, int(max_pairs))
        self.tracks = defaultdict(list)
        self._models_checked = False
        self._near_miss_error = None
        self._accident_error = None
        self._near_miss_available = False
        self._accident_available = False

    def _check_models(self):
        if self._models_checked:
            return
        self._models_checked = True
        try:
            self.safety._ensure_near_miss()
            self._near_miss_available = True
        except Exception as exc:
            self._near_miss_error = f"{type(exc).__name__}: {exc}"
        # The calibrated accident detector accepts images, not SUMO
        # telemetry. Do not load it with fabricated input; report the real
        # degraded state until a camera/image provider is connected.
        self._accident_error = "image input unavailable in SUMO telemetry mode"

    @staticmethod
    def _safe_call(function, *args, default=None):
        try:
            return function(*args)
        except Exception:
            return default

    def _snapshot(self, vehicle_id):
        position = self._safe_call(
            self.traci.vehicle.getPosition, vehicle_id, default=None
        )
        if position is None or len(position) < 2:
            return None
        lane_id = self._safe_call(self.traci.vehicle.getLaneID, vehicle_id, default="")
        edge_id = self._safe_call(self.traci.lane.getEdgeID, lane_id, default="") if lane_id else ""
        route = self._safe_call(self.traci.vehicle.getRoute, vehicle_id, default=[]) or []
        route_index = self._safe_call(self.traci.vehicle.getRouteIndex, vehicle_id, default=None)
        next_edge = None
        if route_index is not None and route_index + 1 < len(route):
            next_edge = route[route_index + 1]
        speed_mps = float(self._safe_call(self.traci.vehicle.getSpeed, vehicle_id, default=0.0) or 0.0)
        point = [float(position[0]), float(position[1])]
        self.tracks[vehicle_id].append(point)
        self.tracks[vehicle_id] = self.tracks[vehicle_id][-self.max_track_points:]
        vehicle_type = self._safe_call(self.traci.vehicle.getTypeID, vehicle_id, default="car") or "car"
        return {
            "vehicle_id": vehicle_id,
            "type": str(vehicle_type),
            "position": {"x": point[0], "y": point[1]},
            # Existing near-miss adapter names this field kph.
            "speed": speed_mps * 3.6,
            "speed_mps": speed_mps,
            "trajectory": list(self.tracks[vehicle_id]),
            "tracking_status": "active",
            "edge_id": edge_id,
            "next_edge": next_edge,
        }

    def _snapshots(self):
        snapshots = []
        active_ids = set(self.traci.vehicle.getIDList())
        for vehicle_id in list(active_ids):
            snapshot = self._snapshot(vehicle_id)
            if snapshot is not None:
                snapshots.append(snapshot)
        for vehicle_id in list(self.tracks):
            if vehicle_id not in active_ids:
                del self.tracks[vehicle_id]
        return snapshots

    @staticmethod
    def _ttc_seconds(vehicle1, vehicle2):
        """Return closing TTC, or None when relative motion is insufficient."""
        if len(vehicle1.get("trajectory", [])) < 2 or len(vehicle2.get("trajectory", [])) < 2:
            return None
        p1 = np.asarray(vehicle1["trajectory"][-1], dtype=float)
        p2 = np.asarray(vehicle2["trajectory"][-1], dtype=float)
        p1_prev = np.asarray(vehicle1["trajectory"][-2], dtype=float)
        p2_prev = np.asarray(vehicle2["trajectory"][-2], dtype=float)
        relative_position = p2 - p1
        relative_velocity = (p2 - p2_prev) - (p1 - p1_prev)
        denominator = float(np.dot(relative_velocity, relative_velocity))
        if denominator <= 1e-12:
            return None
        closing_projection = float(np.dot(relative_position, relative_velocity))
        if closing_projection >= 0.0:
            return None
        ttc = -closing_projection / denominator
        if not math.isfinite(ttc) or ttc < 0.0:
            return None
        return float(ttc)

    def assess(self):
        """Assess current SUMO traffic without inventing missing evidence."""
        self._check_models()
        vehicles = self._snapshots()
        pair_risks = []
        errors = []
        candidate_pairs = []
        for index, vehicle1 in enumerate(vehicles):
            for vehicle2 in vehicles[index + 1:]:
                distance = math.dist(
                    (vehicle1["position"]["x"], vehicle1["position"]["y"]),
                    (vehicle2["position"]["x"], vehicle2["position"]["y"]),
                )
                candidate_pairs.append((distance, vehicle1, vehicle2))
        candidate_pairs.sort(key=lambda item: item[0])
        for _, vehicle1, vehicle2 in candidate_pairs[:self.max_pairs]:
                ttc = self._ttc_seconds(vehicle1, vehicle2)
                try:
                    features = self.feature_adapter.create_features(
                        vehicle1, vehicle2, len(vehicles)
                    )
                    # TTC is telemetry-derived and is logged separately. It is
                    # not inserted into the trained feature schema.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        assessment = self.safety.assess_event(
                            near_miss_features=features,
                            accident_probability=None,
                            severity=None,
                            event_age_hours=None,
                        )
                    near_miss = assessment.get("near_miss_probability")
                    risk_score = float(assessment.get("risk_score", 0.0))
                    pair_risks.append({
                        "vehicles": (vehicle1["vehicle_id"], vehicle2["vehicle_id"]),
                        "roads": {vehicle1.get("edge_id"), vehicle2.get("edge_id")},
                        "near_miss_probability": near_miss,
                        "risk_score": risk_score,
                        "ttc_seconds": ttc,
                        # PET cannot be computed reliably from these SUMO
                        # point tracks without a validated conflict point.
                        "pet_seconds": None,
                    })
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

        highest = max(pair_risks, key=lambda item: item["risk_score"], default=None)
        max_risk = min(1.0, highest["risk_score"] / 100.0) if highest else 0.0
        max_near_miss = max(
            (float(item["near_miss_probability"]) for item in pair_risks
             if item["near_miss_probability"] is not None),
            default=0.0,
        )
        if self._near_miss_available:
            safety_confidence = 0.5 if not self._accident_available else 0.75
        else:
            safety_confidence = 0.0
        accident_status = (
            "loaded_no_camera_frame"
            if self._accident_available
            else "unavailable_no_camera_frame_or_model"
        )
        return {
            "risk_score": max_risk,
            "near_miss_risk": min(1.0, max_near_miss),
            "accident_probability": None,
            "accident_status": accident_status,
            "accident_confidence": None,
            "safety_confidence": safety_confidence,
            "pair_risks": pair_risks,
            "ttc_available": any(item["ttc_seconds"] is not None for item in pair_risks),
            "pet_available": False,
            "model_errors": errors + [error for error in (self._near_miss_error, self._accident_error) if error],
            "degraded": bool(errors or self._near_miss_error or self._accident_error or not self._accident_available),
        }

    @staticmethod
    def phase_risk(phase, topology, assessment):
        """Assign pair risk to phases whose movements involve the pair roads."""
        phase_roads = set()
        for movement_id in phase.get("movements", []):
            movement = topology.movements[movement_id]
            phase_roads.update((movement.from_road, movement.to_road))
        relevant = [
            pair["risk_score"] / 100.0
            for pair in assessment.get("pair_risks", [])
            if phase_roads.intersection(pair["roads"])
        ]
        return min(1.0, max(relevant, default=0.0))
