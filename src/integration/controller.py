from src.integration.decision_engine import DecisionEngine
from src.integration.safety_gate import SafetyGate
from src.integration.fallback_controller import FallbackController


class TrafficController:

    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.safety_gate = SafetyGate()
        self.fallback_controller = FallbackController()

    def _validate_phase(self, phase, checks):
        """
        Run the Safety Gate for one phase.
        """

        return self.safety_gate.validate(
            phase=phase,
            topology_valid=checks["topology_valid"],
            conflict_free=checks["conflict_free"],
            timing_valid=checks["timing_valid"],
            pedestrian_clear=checks["pedestrian_clear"],
            downstream_available=checks["downstream_available"],
            emergency_safe=checks["emergency_safe"],
            confidence=checks["confidence"],
            safety_risk=checks.get("safety_risk", 0.0),
            accident_detected=checks.get("accident_detected", False),
            safety_confidence=checks.get("safety_confidence")
        )

    def decide_and_validate(
        self,
        phase_scores,
        phase_checks,
        phase_densities=None,
        phases=None,
        current_index=0
    ):
        """
        Complete fault-tolerant traffic control pipeline.

        AI
        ↓
        Safety Gate
        ↓
        Rule-Based
        ↓
        Density-Based
        ↓
        Fixed-Time
        """

        if not phase_scores:
            raise ValueError("No phase scores provided")

        # --------------------------------
        # STEP 1: AI DECISION
        # --------------------------------

        decision = self.decision_engine.select_best_phase(
            phase_scores
        )

        selected_phase = decision["selected_phase"]

        if selected_phase not in phase_checks:
            raise ValueError(
                f"No safety checks found for {selected_phase}"
            )

        # --------------------------------
        # STEP 2: SAFETY CHECK AI PHASE
        # --------------------------------

        safety_result = self._validate_phase(
            selected_phase,
            phase_checks[selected_phase]
        )

        # --------------------------------
        # STEP 3: AI APPROVED
        # --------------------------------

        if safety_result["approved"]:

            return {
                "selected_phase": selected_phase,
                "decision_score": decision["score"],
                "all_phase_scores": decision["all_scores"],
                "method": "ai_adaptive",
                "safety": safety_result,
                "execute": True
            }

        # --------------------------------
        # AI REJECTED
        # --------------------------------

        unsafe_phases = {selected_phase}

        # --------------------------------
        # STEP 4: RULE-BASED FALLBACK
        # --------------------------------

        fallback_result = self.fallback_controller.rule_based(
            phase_scores,
            unsafe_phases=unsafe_phases
        )

        if fallback_result is not None:

            fallback_phase = fallback_result["selected_phase"]

            fallback_safety = self._validate_phase(
                fallback_phase,
                phase_checks[fallback_phase]
            )

            if fallback_safety["approved"]:

                return {
                    "selected_phase": fallback_phase,
                    "decision_score": decision["score"],
                    "all_phase_scores": decision["all_scores"],
                    "method": "rule_based_fallback",
                    "safety": fallback_safety,
                    "fallback": fallback_result,
                    "execute": True
                }

            unsafe_phases.add(fallback_phase)

        # --------------------------------
        # STEP 5: DENSITY FALLBACK
        # --------------------------------

        if phase_densities:

            density_result = self.fallback_controller.density_based(
                phase_densities,
                unsafe_phases=unsafe_phases
            )

            if density_result is not None:

                density_phase = density_result["selected_phase"]

                density_safety = self._validate_phase(
                    density_phase,
                    phase_checks[density_phase]
                )

                if density_safety["approved"]:

                    return {
                        "selected_phase": density_phase,
                        "decision_score": decision["score"],
                        "all_phase_scores": decision["all_scores"],
                        "method": "density_based_fallback",
                        "safety": density_safety,
                        "fallback": density_result,
                        "execute": True
                    }

                unsafe_phases.add(density_phase)

        # --------------------------------
        # STEP 6: SAFE FIXED-TIME
        # --------------------------------

        if phases is None:
            phases = list(phase_scores.keys())

        for offset in range(len(phases)):

            index = (current_index + offset) % len(phases)
            fixed_phase = phases[index]

            if fixed_phase in unsafe_phases:
                continue

            fixed_safety = self._validate_phase(
                fixed_phase,
                phase_checks[fixed_phase]
            )

            if fixed_safety["approved"]:

                fixed_result = {
                    "method": "fixed_time",
                    "selected_phase": fixed_phase
                }

                return {
                    "selected_phase": fixed_phase,
                    "decision_score": decision["score"],
                    "all_phase_scores": decision["all_scores"],
                    "method": "safe_fixed_time",
                    "safety": fixed_safety,
                    "fallback": fixed_result,
                    "execute": True
                }

        # --------------------------------
        # STEP 7: NOTHING IS SAFE
        # --------------------------------

        return {
            "selected_phase": None,
            "decision_score": decision["score"],
            "all_phase_scores": decision["all_scores"],
            "method": "emergency_safe_stop",
            "safety": safety_result,
            "execute": False
        }