class TrafficStateEngine:

    def __init__(
        self,
        junction_id,
        approaches,
        traffic_lights=None,
        approach_length=200.0
    ):
        self.junction_id = junction_id
        self.approaches = approaches
        self.traffic_lights = traffic_lights or []
        self.approach_length = approach_length

    def calculate_state(self, traci):

        vehicles = traci.vehicle.getIDList()
        approaches = self._initialize_approaches()

        for vehicle_id in vehicles:

            road_id = traci.vehicle.getRoadID(vehicle_id)
            direction = self._find_approach(road_id)

            if direction is None:
                continue

            speed = traci.vehicle.getSpeed(vehicle_id)
            waiting = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

            approaches[direction]["vehicle_count"] += 1
            approaches[direction]["speeds"].append(speed)
            approaches[direction]["waiting_times"].append(waiting)

            if speed < 1.0:
                approaches[direction]["queue_length"] += 1

        for direction in approaches:
            self._calculate_approach_metrics(approaches[direction])

        for direction in approaches:
            self._calculate_congestion(approaches[direction])

        traffic_lights = self._get_traffic_light_state(traci)

        most_congested = self._find_most_congested(approaches)

        highest_score = approaches[most_congested]["congestion_score"]

        return {
            "junction_id": self.junction_id,
            "vehicle_count": sum(
                data["vehicle_count"]
                for data in approaches.values()
            ),
            "approaches": approaches,
            "traffic_lights": traffic_lights,
            "most_congested_approach": most_congested,
            "highest_congestion_score": highest_score
        }

    def _initialize_approaches(self):

        approaches = {}

        for direction, edge_id in self.approaches.items():

            approaches[direction] = {
                "edge_id": edge_id,
                "vehicle_count": 0,
                "queue_length": 0,
                "speeds": [],
                "waiting_times": []
            }

        return approaches

    def _find_approach(self, road_id):

        for direction, edge_id in self.approaches.items():

            if road_id == edge_id:
                return direction

        return None

    def _calculate_approach_metrics(self, data):

        speeds = data["speeds"]
        waiting_times = data["waiting_times"]

        if speeds:
            data["average_speed"] = sum(speeds) / len(speeds)
        else:
            data["average_speed"] = 0.0

        if waiting_times:
            data["average_waiting_time"] = (
                sum(waiting_times) / len(waiting_times)
            )
        else:
            data["average_waiting_time"] = 0.0

        vehicle_count = data["vehicle_count"]

        if self.approach_length > 0:
            data["density"] = (
                vehicle_count / self.approach_length
            )
        else:
            data["density"] = 0.0

        del data["speeds"]
        del data["waiting_times"]

    def _calculate_congestion(self, data):

        queue_score = min(
            data["queue_length"] / 10.0,
            1.0
        )

        waiting_score = min(
            data["average_waiting_time"] / 60.0,
            1.0
        )

        density_score = min(
            data["density"] / 0.05,
            1.0
        )

        vehicle_score = min(
            data["vehicle_count"] / 20.0,
            1.0
        )

        data["congestion_score"] = (
            0.40 * queue_score
            + 0.30 * waiting_score
            + 0.20 * density_score
            + 0.10 * vehicle_score
        )

    def _get_traffic_light_state(self, traci):

        traffic_lights = {}

        for tls_id in self.traffic_lights:

            current_phase = traci.trafficlight.getPhase(tls_id)

            phase_duration = (
                traci.trafficlight.getPhaseDuration(tls_id)
            )

            signal_state = (
                traci.trafficlight.getRedYellowGreenState(tls_id)
            )

            traffic_lights[tls_id] = {
                "current_phase": current_phase,
                "phase_duration": phase_duration,
                "signal_state": signal_state
            }

        return traffic_lights

    def _find_most_congested(self, approaches):

        most_congested = None
        highest_score = -1.0

        for direction, data in approaches.items():

            score = data["congestion_score"]

            if score > highest_score:
                highest_score = score
                most_congested = direction

        return most_congested
