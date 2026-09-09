class SafetyGate:
    """
    Validates a proposed signal phase before execution.

    The Safety Gate does not choose the phase.
    It only decides whether the proposed phase is safe to execute.

    Safety confidence is optional because some execution environments,
    such as SUMO-only telemetry mode, do not provide camera/image
    evidence. In that case, safety_confidence=None means that the
    image-based confidence source is unavailable; it does NOT mean
    that the phase is unsafe.

    Other safety checks remain mandatory:
        - topology validity
        - conflict freedom
        - timing validity
        - pedestrian clearance
        - downstream availability
        - emergency safety
        - AI decision confidence
        - risk threshold
        - accident detection
    """

    def __init__(
        self,
        min_green_seconds=10,
        yellow_seconds=3,
        all_red_seconds=1,
        confidence_threshold=0.50,
        risk_threshold=0.75,
        accident_probability_threshold=0.75
    ):
        self.min_green_seconds = min_green_seconds
        self.yellow_seconds = yellow_seconds
        self.all_red_seconds = all_red_seconds
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.accident_probability_threshold = accident_probability_threshold

    def validate(
        self,
        phase,
        topology_valid,
        conflict_free,
        timing_valid,
        pedestrian_clear,
        downstream_available,
        emergency_safe,
        confidence,
        safety_risk=0.0,
        accident_detected=False,
        safety_confidence=None
    ):
        """
        Validate a proposed signal phase.

        safety_confidence semantics:

            None
                Safety confidence source is unavailable.
                This is valid for SUMO-only telemetry mode.

            numeric value
                A real safety-confidence measurement is available.
                It must satisfy confidence_threshold.

        Returns:
            {
                "approved": bool,
                "phase": phase,
                "checks": {...},
                "reasons": [...]
            }
        """

        # ---------------------------------------------------------
        # Normalize numeric inputs safely
        # ---------------------------------------------------------

        try:
            decision_confidence = float(confidence)
        except (TypeError, ValueError):
            decision_confidence = 0.0

        try:
            risk_value = float(safety_risk)
        except (TypeError, ValueError):
            risk_value = 1.0

        # ---------------------------------------------------------
        # Core safety checks
        # ---------------------------------------------------------

        checks = {
            "topology_valid": bool(topology_valid),

            "conflict_free": bool(conflict_free),

            "timing_valid": bool(timing_valid),

            "pedestrian_clear": bool(pedestrian_clear),

            "downstream_available": bool(
                downstream_available
            ),

            "emergency_safe": bool(emergency_safe),

            # This is the confidence of the proposed AI decision.
            "confidence_valid": (
                decision_confidence >= self.confidence_threshold
            ),

            # Risk remains mandatory even when camera evidence
            # is unavailable.
            "risk_valid": (
                0.0 <= risk_value <= self.risk_threshold
            ),

            # A confirmed accident always blocks the phase.
            "accident_safe": (
                not bool(accident_detected)
            ),
        }

        # ---------------------------------------------------------
        # Optional camera/safety confidence
        # ---------------------------------------------------------
        #
        # None means:
        #
        #   "No image-based safety confidence is available."
        #
        # This is expected in SUMO telemetry-only mode.
        #
        # A real numeric value is still validated normally.
        # ---------------------------------------------------------

        if safety_confidence is not None:

            try:
                safety_confidence_value = float(
                    safety_confidence
                )
            except (TypeError, ValueError):
                safety_confidence_value = -1.0

            checks["safety_confidence_valid"] = (
                safety_confidence_value
                >= self.confidence_threshold
            )

        # ---------------------------------------------------------
        # Collect rejection reasons
        # ---------------------------------------------------------

        reasons = []

        for name, passed in checks.items():

            if not passed:
                reasons.append(name)

        # ---------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------

        approved = len(reasons) == 0

        return {
            "approved": approved,
            "phase": phase,
            "checks": checks,
            "reasons": reasons
        }