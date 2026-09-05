from src.integration.near_miss_feature_adapter import (
    NearMissFeatureAdapter
)


class NearMissProcessor:

    def __init__(self):
        self.adapter = NearMissFeatureAdapter()

    def process_frame(self, detections):

        vehicles = []

        for detection in detections:

            if not detection.get("tracking_status") == "active":
                continue

            vehicles.append(detection)

        results = []

        for i in range(len(vehicles)):

            for j in range(i + 1, len(vehicles)):

                vehicle1 = vehicles[i]
                vehicle2 = vehicles[j]

                features = self.adapter.create_features(
                    vehicle1,
                    vehicle2,
                    len(vehicles)
                )

                results.append({
                    "vehicle_1": vehicle1["vehicle_id"],
                    "vehicle_2": vehicle2["vehicle_id"],
                    "features": features
                })

        return results