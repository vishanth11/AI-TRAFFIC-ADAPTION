class SafetyGate:
    """
    Validates a proposed signal phase before execution.

    The Safety Gate does not choose the phase.
    It only decides whether the proposed phase is safe to execute.
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

        Returns:
            {
                "approved": bool,
                "phase": phase,
                "checks": {...},
                "reasons": [...]
            }
        """

        checks = {
            "topology_valid": bool(topology_valid),
            "conflict_free": bool(conflict_free),
            "timing_valid": bool(timing_valid),
            "pedestrian_clear": bool(pedestrian_clear),
            "downstream_available": bool(downstream_available),
            "emergency_safe": bool(emergency_safe),
            "confidence_valid": (
                confidence >= self.confidence_threshold
            ),
            "risk_valid": (
                0.0 <= safety_risk <= self.risk_threshold
            ),
            "accident_safe": not bool(accident_detected),
        }

        if safety_confidence is not None:
            checks["safety_confidence_valid"] = (
                safety_confidence >= self.confidence_threshold
            )

        reasons = []

        for name, passed in checks.items():
            if not passed:
                reasons.append(name)

        approved = len(reasons) == 0

        return {
            "approved": approved,
            "phase": phase,
            "checks": checks,
            "reasons": reasons
        }