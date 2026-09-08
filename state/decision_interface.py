class DecisionInterface:

    def build_junction_decision(self, junction_state):

        return {
            "junction_id": junction_state["junction_id"],
            "vehicle_count": junction_state["vehicle_count"],
            "most_congested_approach": (
                junction_state["most_congested_approach"]
            ),
            "congestion_score": (
                junction_state["highest_congestion_score"]
            ),
            "traffic_lights": junction_state[
                "traffic_lights"
            ]
        }

    def build_corridor_decision(self, corridor_state):

        junctions = {}

        for junction_id, state in corridor_state[
            "junctions"
        ].items():

            junctions[junction_id] = (
                self.build_junction_decision(state)
            )

        return {
            "corridor_id": corridor_state["corridor_id"],
            "junction_count": corridor_state["junction_count"],
            "total_vehicles": corridor_state["total_vehicles"],
            "worst_junction": corridor_state["worst_junction"],
            "highest_congestion_score": (
                corridor_state["highest_congestion_score"]
            ),
            "junctions": junctions
        }