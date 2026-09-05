import joblib
import pandas as pd


class NearMissModel:

    FEATURE_NAMES = [
        "speed_object_1_kph",
        "speed_object_2_kph",
        "angle_degrees",
        "acceleration_obj1_mps2",
        "acceleration_obj2_mps2",
        "min_dist_dual_check_m",
        "object_count_at_frame1",
        "target_dist_px",
        "FE_log_mesafe",
        "FE_dist_squared",
        "class_object_1",
        "class_object_2"
    ]

    def __init__(
        self,
        model_path="models/near_miss/random_forest_D_calibrated.joblib"
    ):
        print("Loading Near-Miss Model D...")

        self.model = joblib.load(model_path)

        print("Model D loaded successfully.")

    def predict_probability(self, features):

        row = {}

        for feature in self.FEATURE_NAMES:
            row[feature] = features.get(feature)

        dataframe = pd.DataFrame([row])

        probability = self.model.predict_proba(dataframe)[0][1]

        return float(probability)