class CorridorStateEngine:

    def __init__(self, junction_engines):
        self.junction_engines = junction_engines

    def calculate_corridor_state(self, traci):

        junction_states = {}

        total_vehicles = 0
        worst_junction = None
        highest_congestion = -1.0

        for junction_name, engine in self.junction_engines.items():

            state = engine.calculate_state(traci)

            junction_states[junction_name] = state

            total_vehicles += state["vehicle_count"]

            congestion = state["highest_congestion_score"]

            if congestion > highest_congestion:
                highest_congestion = congestion
                worst_junction = junction_name

        corridor_state = {
            "corridor_id": "CORRIDOR_1",
            "junction_count": len(junction_states),
            "total_vehicles": total_vehicles,
            "junctions": junction_states,
            "worst_junction": worst_junction,
            "highest_congestion_score": highest_congestion
        }

        return corridor_state