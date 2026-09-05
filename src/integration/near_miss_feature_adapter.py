import math
from typing import Dict, Any, List


class NearMissFeatureAdapter:

    CLASS_MAPPING = {
        "car": "Car",
        "person": "Ped",
        "motorcycle": "VRU",
        "bus": "HCV",
        "truck": "HCV",
    }

    def __init__(self):
        pass

    def map_class(self, vehicle_type: str) -> str:
        vehicle_type = vehicle_type.lower()

        if vehicle_type in self.CLASS_MAPPING:
            return self.CLASS_MAPPING[vehicle_type]

        return "Car"

    def get_position(self, vehicle: Dict[str, Any]):
        position = vehicle.get("position", {})

        x = position.get("x", 0.0)
        y = position.get("y", 0.0)

        return float(x), float(y)

    def calculate_pixel_distance(self, vehicle1, vehicle2):

        x1, y1 = self.get_position(vehicle1)
        x2, y2 = self.get_position(vehicle2)

        dx = x2 - x1
        dy = y2 - y1

        distance = math.sqrt(dx * dx + dy * dy)

        return distance

    def calculate_angle(self, vehicle1, vehicle2):

        trajectory1 = vehicle1.get("trajectory", [])
        trajectory2 = vehicle2.get("trajectory", [])

        if len(trajectory1) < 2 or len(trajectory2) < 2:
            return 0.0

        p1_old = trajectory1[-2]
        p1_new = trajectory1[-1]

        p2_old = trajectory2[-2]
        p2_new = trajectory2[-1]

        v1x = p1_new[0] - p1_old[0]
        v1y = p1_new[1] - p1_old[1]

        v2x = p2_new[0] - p2_old[0]
        v2y = p2_new[1] - p2_old[1]

        mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
        mag2 = math.sqrt(v2x ** 2 + v2y ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        dot = v1x * v2x + v1y * v2y

        cosine = dot / (mag1 * mag2)

        cosine = max(-1.0, min(1.0, cosine))

        angle = math.degrees(math.acos(cosine))

        return angle

    def calculate_acceleration(self, vehicle):

        trajectory = vehicle.get("trajectory", [])

        if len(trajectory) < 3:
            return 0.0

        p1 = trajectory[-3]
        p2 = trajectory[-2]
        p3 = trajectory[-1]

        d1 = math.sqrt(
            (p2[0] - p1[0]) ** 2 +
            (p2[1] - p1[1]) ** 2
        )

        d2 = math.sqrt(
            (p3[0] - p2[0]) ** 2 +
            (p3[1] - p2[1]) ** 2
        )

        # Current Member 2 pipeline uses 30 FPS
        fps = 30.0
        pixels_per_meter = 15.0

        dt = 1.0 / fps

        v1 = (d1 / pixels_per_meter) / dt
        v2 = (d2 / pixels_per_meter) / dt

        acceleration = (v2 - v1) / dt

        return acceleration

    def create_features(
        self,
        vehicle1: Dict[str, Any],
        vehicle2: Dict[str, Any],
        object_count: int
    ):

        speed1 = float(vehicle1.get("speed", 0.0))
        speed2 = float(vehicle2.get("speed", 0.0))

        target_dist_px = self.calculate_pixel_distance(
            vehicle1,
            vehicle2
        )

        angle = self.calculate_angle(
            vehicle1,
            vehicle2
        )

        acceleration1 = self.calculate_acceleration(
            vehicle1
        )

        acceleration2 = self.calculate_acceleration(
            vehicle2
        )

        features = {

            "speed_object_1_kph": speed1,

            "speed_object_2_kph": speed2,

            "angle_degrees": angle,

            "acceleration_obj1_mps2": acceleration1,

            "acceleration_obj2_mps2": acceleration2,

            # IMPORTANT:
            # Exact project definition has not yet been recovered.
            "min_dist_dual_check_m": None,

            "object_count_at_frame1": object_count,

            "target_dist_px": target_dist_px,

            "FE_log_mesafe": math.log(
                1.0 + target_dist_px
            ),

            "FE_dist_squared": target_dist_px ** 2,

            "class_object_1": self.map_class(
                vehicle1.get("type", "car")
            ),

            "class_object_2": self.map_class(
                vehicle2.get("type", "car")
            ),
        }

        return features