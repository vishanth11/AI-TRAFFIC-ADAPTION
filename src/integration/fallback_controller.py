class FallbackController:

    def rule_based(self, phase_scores, unsafe_phases=None):
        """
        Select the highest-scoring phase that is not unsafe.
        """

        if not phase_scores:
            raise ValueError("No phase scores provided")

        if unsafe_phases is None:
            unsafe_phases = set()

        safe_phases = {
            phase: score
            for phase, score in phase_scores.items()
            if phase not in unsafe_phases
        }

        if not safe_phases:
            return None

        selected_phase = max(
            safe_phases,
            key=safe_phases.get
        )

        return {
            "method": "rule_based",
            "selected_phase": selected_phase,
            "score": safe_phases[selected_phase]
        }

    def density_based(self, phase_densities, unsafe_phases=None):
        """
        Select the highest-density safe phase.
        """

        if not phase_densities:
            raise ValueError("No phase densities provided")

        if unsafe_phases is None:
            unsafe_phases = set()

        safe_phases = {
            phase: density
            for phase, density in phase_densities.items()
            if phase not in unsafe_phases
        }

        if not safe_phases:
            return None

        selected_phase = max(
            safe_phases,
            key=safe_phases.get
        )

        return {
            "method": "density_based",
            "selected_phase": selected_phase,
            "density": safe_phases[selected_phase]
        }

    def fixed_time(self, phases, current_index=0):
        """
        Final safe fixed-time fallback.
        """

        if not phases:
            raise ValueError("No phases provided")

        index = current_index % len(phases)
        selected_phase = phases[index]

        return {
            "method": "fixed_time",
            "selected_phase": selected_phase
        }