import traci


class SUMOPhaseAdapter:
    """
    Converts SUMO traffic-light phases into a generic
    representation that can be used by the decision engine.

    SUMO-specific link indices are kept here instead of
    being placed inside the generic topology classes.
    """

    def __init__(self, traci_connection=None):
        self.traci = traci_connection or traci

    # ============================================================
    # GET ALL SIGNAL PROGRAMS
    # ============================================================

    def get_programs(self, junction_id):
        """
        Return all signal programs available at a junction.
        """

        programs = (
            self.traci.trafficlight.getAllProgramLogics(
                junction_id
            )
        )

        result = []

        for program in programs:

            program_data = {
                "program_id": program.programID,
                "type": program.type,
                "current_phase": (
                    self.traci.trafficlight.getPhase(
                        junction_id
                    )
                ),
                "phases": []
            }

            for phase_index, phase in enumerate(
                program.phases
            ):

                program_data["phases"].append({
                    "phase_index": phase_index,
                    "duration": phase.duration,
                    "min_duration": phase.minDur,
                    "max_duration": phase.maxDur,
                    "state": phase.state
                })

            result.append(program_data)

        return result

    # ============================================================
    # GET CURRENT PROGRAM
    # ============================================================

    def get_current_program(self, junction_id):
        """
        Get the currently active SUMO signal program.
        """

        program_id = (
            self.traci.trafficlight.getProgram(
                junction_id
            )
        )

        programs = self.get_programs(junction_id)

        for program in programs:

            if program["program_id"] == program_id:
                return program

        return None

    # ============================================================
    # GET CURRENT PHASE
    # ============================================================

    def get_current_phase(self, junction_id):
        """
        Return the current phase index.
        """

        return self.traci.trafficlight.getPhase(
            junction_id
        )

    # ============================================================
    # GET PHASE STATE
    # ============================================================

    def get_phase_state(
        self,
        junction_id,
        phase_index
    ):
        """
        Return the signal state string for a phase.

        Example:

            GGggrrrrGGggrrrr
        """

        program = self.get_current_program(
            junction_id
        )

        if program is None:
            return None

        for phase in program["phases"]:

            if phase["phase_index"] == phase_index:

                return phase["state"]

        return None

    # ============================================================
    # GET GREEN LINKS
    # ============================================================

    def get_green_links(
        self,
        junction_id,
        phase_index
    ):
        """
        Return link indices that are green during
        the specified phase.

        SUMO signal characters:

            G = green
            g = permissive green
            y = yellow
            r = red

        Both G and g are considered active green.
        """

        state = self.get_phase_state(
            junction_id,
            phase_index
        )

        if state is None:
            return []

        green_links = []

        for link_index, signal in enumerate(state):

            if signal in ("G", "g"):

                green_links.append(
                    link_index
                )

        return green_links

    # ============================================================
    # GET PHASE TYPE
    # ============================================================

    def get_phase_type(
        self,
        junction_id,
        phase_index
    ):
        """
        Classify a phase as:

            green
            yellow
            red
            mixed
        """

        state = self.get_phase_state(
            junction_id,
            phase_index
        )

        if state is None:
            return "unknown"

        has_green = any(
            signal in ("G", "g")
            for signal in state
        )

        has_yellow = any(
            signal == "y"
            for signal in state
        )

        has_red = any(
            signal == "r"
            for signal in state
        )

        if has_green:
            return "green"

        if has_yellow:
            return "yellow"

        if has_red:
            return "red"

        return "mixed"

    # ============================================================
    # GET DECISION PHASES
    # ============================================================

    def get_decision_phases(self, junction_id):
        """
        Return phases that can be considered normal
        green decision phases.

        Yellow phases are deliberately excluded.
        """

        program = self.get_current_program(
            junction_id
        )

        if program is None:
            return []

        decision_phases = []

        for phase in program["phases"]:

            phase_index = phase["phase_index"]

            phase_type = self.get_phase_type(
                junction_id,
                phase_index
            )

            if phase_type == "green":

                decision_phases.append({
                    "phase_index": phase_index,
                    "duration": phase["duration"],
                    "state": phase["state"],
                    "green_links": (
                        self.get_green_links(
                            junction_id,
                            phase_index
                        )
                    )
                })

        return decision_phases

    # ============================================================
    # CONVERT PHASE TO MOVEMENTS
    # ============================================================

    def phase_to_movements(
        self,
        junction_id,
        phase_index,
        movement_metadata
    ):
        """
        Convert SUMO green link indices into movement IDs.

        movement_metadata should be the metadata produced
        by SUMOTopologyAdapter.

        Example:

            link 0
                ↓
            A0_M0

            link 1
                ↓
            A0_M1
        """

        green_links = self.get_green_links(
            junction_id,
            phase_index
        )

        movements = []

        for movement_id, metadata in (
            movement_metadata.items()
        ):

            if (
                metadata["junction_id"] == junction_id
                and metadata["link_index"]
                in green_links
            ):

                movements.append(
                    movement_id
                )

        return movements

    # ============================================================
    # BUILD PHASE REPRESENTATION
    # ============================================================

    def build_phase_representation(
        self,
        junction_id,
        movement_metadata
    ):
        """
        Build a generic representation of all usable
        signal phases at a junction.
        """

        decision_phases = (
            self.get_decision_phases(
                junction_id
            )
        )

        phases = []

        for phase in decision_phases:

            phase_index = phase[
                "phase_index"
            ]

            movement_ids = (
                self.phase_to_movements(
                    junction_id,
                    phase_index,
                    movement_metadata
                )
            )

            phases.append({

                "phase_id": (
                    f"{junction_id}_P"
                    f"{phase_index}"
                ),

                "junction_id": junction_id,

                "phase_index": phase_index,

                "duration": phase["duration"],

                "state": phase["state"],

                "green_links": phase[
                    "green_links"
                ],

                "movements": movement_ids
            })

        return phases

    # ============================================================
    # PRINT DEBUG INFORMATION
    # ============================================================

    def print_junction_phases(
        self,
        junction_id,
        movement_metadata
    ):
        """
        Print a readable representation for debugging.
        """

        print()
        print("=" * 70)
        print(
            f"PHASES FOR JUNCTION: {junction_id}"
        )
        print("=" * 70)

        phases = (
            self.build_phase_representation(
                junction_id,
                movement_metadata
            )
        )

        for phase in phases:

            print()

            print(
                f"Phase: {phase['phase_id']}"
            )

            print(
                f"  SUMO index: "
                f"{phase['phase_index']}"
            )

            print(
                f"  Duration: "
                f"{phase['duration']} seconds"
            )

            print(
                f"  State: "
                f"{phase['state']}"
            )

            print(
                f"  Green links: "
                f"{phase['green_links']}"
            )

            print(
                f"  Movements: "
                f"{phase['movements']}"
            )


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    SUMO_CMD = [
        "sumo",
        "-c",
        "simulation/configs/corridor.sumocfg"
    ]

    try:

        traci.start(SUMO_CMD)

        from integration.sumo_topology_adapter import (
            SUMOTopologyAdapter
        )

        topology_adapter = (
            SUMOTopologyAdapter()
        )

        phase_adapter = (
            SUMOPhaseAdapter()
        )

        # Discover topology first.
        topologies = (
            topology_adapter
            .discover_all_junctions()
        )

        # Display phases for the main
        # corridor junctions.
        for junction_id in [
            "A0",
            "B0",
            "C0"
        ]:

            phase_adapter.print_junction_phases(
                junction_id,
                topology_adapter.sumo_metadata
            )

    finally:

        traci.close()