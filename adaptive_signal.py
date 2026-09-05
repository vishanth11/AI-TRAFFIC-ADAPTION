class AdaptiveSignalController:

    def __init__(
        self,
        min_green=20,
        max_green=60,
        congestion_threshold=0.50,
        extension_seconds=10
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.congestion_threshold = congestion_threshold
        self.extension_seconds = extension_seconds

        self.last_phase = {}
        self.extended_duration = {}

        self.approach_phase = {
            "A0": {"west": 2},
            "B0": {"west": 2},
            "C0": {"west": 2}
        }

    def decide(self, junction_state, traci):

        junction_id = junction_state["junction_id"]

        tls_data = junction_state["traffic_lights"].get(
            junction_id
        )

        if tls_data is None:
            return {
                "junction_id": junction_id,
                "current_phase": None,
                "current_duration": None,
                "most_congested_approach":
                    junction_state["most_congested_approach"],
                "congestion_score":
                    junction_state["highest_congestion_score"],
                "action": "HOLD",
                "new_duration": None,
                "target_phase": None,
                "serves_congested_approach": False,
                "reason": "Traffic light not found"
            }

        current_phase = tls_data["current_phase"]
        current_duration = tls_data["phase_duration"]

        congestion = junction_state[
            "highest_congestion_score"
        ]

        most_congested = junction_state[
            "most_congested_approach"
        ]

        previous_phase = self.last_phase.get(
            junction_id
        )

        if previous_phase != current_phase:

            self.extended_duration.pop(
                junction_id,
                None
            )

            self.last_phase[junction_id] = current_phase

        effective_duration = self.extended_duration.get(
            junction_id,
            current_duration
        )

        # Default decision
        decision = {
            "junction_id": junction_id,
            "current_phase": current_phase,
            "current_duration": current_duration,
            "most_congested_approach": most_congested,
            "congestion_score": congestion,
            "action": "HOLD",
            "new_duration": current_duration,
            "target_phase": current_phase,
            "serves_congested_approach": False,
            "reason": "Congestion is below threshold"
        }

        # No adaptive action needed
        if congestion < self.congestion_threshold:

            decision["serves_congested_approach"] = None

            return decision

        target_phase = self._get_target_phase(
            junction_id,
            most_congested
        )

        if target_phase is None:

            decision["reason"] = (
                f"No configured phase mapping for "
                f"{most_congested} approach"
            )

            return decision

        decision["target_phase"] = target_phase

        if current_phase == target_phase:

            decision["serves_congested_approach"] = True

            if effective_duration < self.max_green:

                new_duration = min(
                    effective_duration
                    + self.extension_seconds,
                    self.max_green
                )

                decision["action"] = "EXTEND_GREEN"
                decision["new_duration"] = new_duration

                decision["reason"] = (
                    f"High congestion on "
                    f"{most_congested}; "
                    f"phase {current_phase} serves "
                    f"this approach"
                )

            else:

                decision["reason"] = (
                    "Maximum green duration reached"
                )

            return decision

        decision["serves_congested_approach"] = False

        decision["action"] = "SWITCH_PHASE"

        decision["reason"] = (
            f"High congestion on "
            f"{most_congested}; "
            f"switching from phase "
            f"{current_phase} to phase "
            f"{target_phase}"
        )

        return decision

    def _get_target_phase(
        self,
        junction_id,
        approach
    ):

        junction_mapping = self.approach_phase.get(
            junction_id,
            {}
        )

        return junction_mapping.get(
            approach
        )

    def apply_decision(self, traci, decision):

        junction_id = decision["junction_id"]

        if decision["action"] == "EXTEND_GREEN":

            new_duration = decision["new_duration"]

            traci.trafficlight.setPhaseDuration(
                junction_id,
                new_duration
            )

            self.extended_duration[
                junction_id
            ] = new_duration

            return True

        if decision["action"] == "SWITCH_PHASE":

            target_phase = decision["target_phase"]

            traci.trafficlight.setPhase(
                junction_id,
                target_phase
            )

            self.extended_duration.pop(
                junction_id,
                None
            )

            self.last_phase[
                junction_id
            ] = target_phase

            return True

        return False