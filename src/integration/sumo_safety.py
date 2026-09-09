"""
SUMO telemetry adapter for Phase 2.2 safety intelligence.

This adapter reuses the project's calibrated near-miss feature/model and
SafetyIntelligence risk fusion.

SUMO does not provide camera frames, so the image-based accident detector
is never fed fabricated input.

In SUMO telemetry-only mode:

    safety_confidence=None

means that image-based safety confidence is unavailable.

This is different from a measured confidence of 0.0.

Telemetry-derived safety signals such as near-miss probability, TTC,
and phase-specific risk remain active.
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

        self.max_track_points = max(
            3,
            int(max_track_points)
        )

        self.max_pairs = max(
            1,
            int(max_pairs)
        )

        self.tracks = defaultdict(list)

        self._models_checked = False
        self._near_miss_error = None
        self._accident_error = None

        self._near_miss_available = False
        self._accident_available = False

    # =========================================================
    # MODEL AVAILABILITY
    # =========================================================

    def _check_models(self):
        """
        Check which safety models are actually available.

        Never fabricate accident-detector input.

        SUMO provides telemetry but does not provide camera frames,
        therefore the image-based accident detector is considered
        unavailable in telemetry-only mode.
        """

        if self._models_checked:
            return

        self._models_checked = True

        # -----------------------------------------------------
        # Near-miss model
        # -----------------------------------------------------

        try:
            self.safety._ensure_near_miss()
            self._near_miss_available = True

        except Exception as exc:
            self._near_miss_error = (
                f"{type(exc).__name__}: {exc}"
            )

        # -----------------------------------------------------
        # Accident detector
        # -----------------------------------------------------
        #
        # The calibrated accident detector expects images.
        #
        # SUMO does not provide camera frames.
        #
        # DO NOT fabricate an image just to obtain a confidence.
        # -----------------------------------------------------

        self._accident_error = (
            "image input unavailable in SUMO telemetry mode"
        )

        self._accident_available = False

    # =========================================================
    # SAFE TRACI CALL
    # =========================================================

    @staticmethod
    def _safe_call(function, *args, default=None):

        try:
            return function(*args)

        except Exception:
            return default

    # =========================================================
    # VEHICLE SNAPSHOT
    # =========================================================

    def _snapshot(self, vehicle_id):

        position = self._safe_call(
            self.traci.vehicle.getPosition,
            vehicle_id,
            default=None
        )

        if position is None or len(position) < 2:
            return None

        lane_id = self._safe_call(
            self.traci.vehicle.getLaneID,
            vehicle_id,
            default=""
        )

        edge_id = (
            self._safe_call(
                self.traci.lane.getEdgeID,
                lane_id,
                default=""
            )
            if lane_id
            else ""
        )

        route = self._safe_call(
            self.traci.vehicle.getRoute,
            vehicle_id,
            default=[]
        ) or []

        route_index = self._safe_call(
            self.traci.vehicle.getRouteIndex,
            vehicle_id,
            default=None
        )

        next_edge = None

        if (
            route_index is not None
            and route_index + 1 < len(route)
        ):
            next_edge = route[route_index + 1]

        speed_mps = float(
            self._safe_call(
                self.traci.vehicle.getSpeed,
                vehicle_id,
                default=0.0
            )
            or 0.0
        )

        point = [
            float(position[0]),
            float(position[1])
        ]

        self.tracks[vehicle_id].append(point)

        self.tracks[vehicle_id] = (
            self.tracks[vehicle_id]
            [-self.max_track_points:]
        )

        vehicle_type = (
            self._safe_call(
                self.traci.vehicle.getTypeID,
                vehicle_id,
                default="car"
            )
            or "car"
        )

        return {
            "vehicle_id": vehicle_id,
            "type": str(vehicle_type),
            "position": {
                "x": point[0],
                "y": point[1]
            },

            # Existing near-miss adapter expects kph.
            "speed": speed_mps * 3.6,

            "speed_mps": speed_mps,

            "trajectory": list(
                self.tracks[vehicle_id]
            ),

            "tracking_status": "active",

            "edge_id": edge_id,

            "next_edge": next_edge,
        }

    # =========================================================
    # ALL VEHICLE SNAPSHOTS
    # =========================================================

    def _snapshots(self):

        snapshots = []

        active_ids = set(
            self.traci.vehicle.getIDList()
        )

        for vehicle_id in list(active_ids):

            snapshot = self._snapshot(
                vehicle_id
            )

            if snapshot is not None:
                snapshots.append(snapshot)

        # Remove vehicles that left SUMO.
        for vehicle_id in list(self.tracks):

            if vehicle_id not in active_ids:
                del self.tracks[vehicle_id]

        return snapshots

    # =========================================================
    # TIME TO COLLISION
    # =========================================================

    @staticmethod
    def _ttc_seconds(vehicle1, vehicle2):
        """
        Return closing TTC.

        Returns None when relative motion is insufficient to
        calculate a meaningful closing time.
        """

        if (
            len(vehicle1.get("trajectory", [])) < 2
            or len(vehicle2.get("trajectory", [])) < 2
        ):
            return None

        p1 = np.asarray(
            vehicle1["trajectory"][-1],
            dtype=float
        )

        p2 = np.asarray(
            vehicle2["trajectory"][-1],
            dtype=float
        )

        p1_prev = np.asarray(
            vehicle1["trajectory"][-2],
            dtype=float
        )

        p2_prev = np.asarray(
            vehicle2["trajectory"][-2],
            dtype=float
        )

        relative_position = p2 - p1

        relative_velocity = (
            (p2 - p2_prev)
            - (p1 - p1_prev)
        )

        denominator = float(
            np.dot(
                relative_velocity,
                relative_velocity
            )
        )

        if denominator <= 1e-12:
            return None

        closing_projection = float(
            np.dot(
                relative_position,
                relative_velocity
            )
        )

        if closing_projection >= 0.0:
            return None

        ttc = (
            -closing_projection
            / denominator
        )

        if (
            not math.isfinite(ttc)
            or ttc < 0.0
        ):
            return None

        return float(ttc)

    # =========================================================
    # SAFETY ASSESSMENT
    # =========================================================

    def assess(self):
        """
        Assess current SUMO traffic without inventing missing evidence.
        """

        self._check_models()

        vehicles = self._snapshots()

        pair_risks = []
        errors = []
        candidate_pairs = []

        # -----------------------------------------------------
        # Candidate vehicle pairs
        # -----------------------------------------------------

        for index, vehicle1 in enumerate(vehicles):

            for vehicle2 in vehicles[index + 1:]:

                distance = math.dist(
                    (
                        vehicle1["position"]["x"],
                        vehicle1["position"]["y"]
                    ),
                    (
                        vehicle2["position"]["x"],
                        vehicle2["position"]["y"]
                    )
                )

                candidate_pairs.append(
                    (
                        distance,
                        vehicle1,
                        vehicle2
                    )
                )

        candidate_pairs.sort(
            key=lambda item: item[0]
        )

        # -----------------------------------------------------
        # Analyze closest pairs
        # -----------------------------------------------------

        for (
            _,
            vehicle1,
            vehicle2
        ) in candidate_pairs[:self.max_pairs]:

            ttc = self._ttc_seconds(
                vehicle1,
                vehicle2
            )

            try:

                features = (
                    self.feature_adapter.create_features(
                        vehicle1,
                        vehicle2,
                        len(vehicles)
                    )
                )

                # TTC is telemetry-derived and is logged
                # separately. It is not inserted into the
                # trained feature schema.

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore",
                        UserWarning
                    )

                    assessment = (
                        self.safety.assess_event(
                            near_miss_features=features,
                            accident_probability=None,
                            severity=None,
                            event_age_hours=None,
                        )
                    )

                near_miss = assessment.get(
                    "near_miss_probability"
                )

                risk_score = float(
                    assessment.get(
                        "risk_score",
                        0.0
                    )
                )

                pair_risks.append(
                    {
                        "vehicles": (
                            vehicle1["vehicle_id"],
                            vehicle2["vehicle_id"]
                        ),

                        "roads": {
                            vehicle1.get("edge_id"),
                            vehicle2.get("edge_id")
                        },

                        "near_miss_probability":
                            near_miss,

                        "risk_score":
                            risk_score,

                        "ttc_seconds":
                            ttc,

                        # PET cannot be reliably computed from
                        # these point tracks without a validated
                        # conflict point.
                        "pet_seconds":
                            None,
                    }
                )

            except Exception as exc:

                errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

        # -----------------------------------------------------
        # Aggregate risk
        # -----------------------------------------------------

        highest = max(
            pair_risks,
            key=lambda item: item["risk_score"],
            default=None
        )

        max_risk = (
            min(
                1.0,
                highest["risk_score"] / 100.0
            )
            if highest
            else 0.0
        )

        max_near_miss = max(
            (
                float(
                    item["near_miss_probability"]
                )
                for item in pair_risks
                if item["near_miss_probability"]
                is not None
            ),
            default=0.0
        )

        # -----------------------------------------------------
        # Safety confidence
        # -----------------------------------------------------
        #
        # IMPORTANT:
        #
        # SUMO has no camera frame.
        #
        # Therefore image-based safety confidence is NOT
        # available.
        #
        # None means unavailable.
        #
        # We must NOT convert "unavailable" into 0.0 because
        # 0.0 means an actual measured confidence of zero.
        # -----------------------------------------------------

        if self._accident_available:

            if self._near_miss_available:
                safety_confidence = 0.75
            else:
                safety_confidence = 0.50

        else:

            safety_confidence = None

        # -----------------------------------------------------
        # Accident status
        # -----------------------------------------------------

        if self._accident_available:

            accident_status = (
                "loaded_no_camera_frame"
            )

        else:

            accident_status = (
                "unavailable_no_camera_frame_or_model"
            )

        # -----------------------------------------------------
        # Final assessment
        # -----------------------------------------------------

        return {
            "risk_score":
                max_risk,

            "near_miss_risk":
                min(
                    1.0,
                    max_near_miss
                ),

            "accident_probability":
                None,

            "accident_status":
                accident_status,

            "accident_confidence":
                None,

            "safety_confidence":
                safety_confidence,

            "pair_risks":
                pair_risks,

            "ttc_available":
                any(
                    item["ttc_seconds"] is not None
                    for item in pair_risks
                ),

            "pet_available":
                False,

            "model_errors":
                errors
                + [
                    error
                    for error in (
                        self._near_miss_error,
                        self._accident_error
                    )
                    if error
                ],

            "degraded":
                bool(
                    errors
                    or self._near_miss_error
                    or self._accident_error
                    or not self._accident_available
                ),
        }

    # =========================================================
    # PHASE-SPECIFIC RISK
    # =========================================================

    @staticmethod
    def phase_risk(
        phase,
        topology,
        assessment
    ):
        """
        Assign pair risk to phases whose movements involve
        the pair's roads.
        """

        phase_roads = set()

        for movement_id in phase.get(
            "movements",
            []
        ):

            movement = topology.movements[
                movement_id
            ]

            phase_roads.update(
                (
                    movement.from_road,
                    movement.to_road
                )
            )

        relevant = [
            pair["risk_score"] / 100.0

            for pair in assessment.get(
                "pair_risks",
                []
            )

            if phase_roads.intersection(
                pair["roads"]
            )
        ]

        return min(
            1.0,
            max(
                relevant,
                default=0.0
            )
        )