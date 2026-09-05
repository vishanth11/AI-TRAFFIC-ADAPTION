from src.integration.near_miss_feature_adapter import (
    NearMissFeatureAdapter
)

from src.integration.near_miss_model import (
    NearMissModel
)


adapter = NearMissFeatureAdapter()

model = NearMissModel()


vehicle1 = {
    "vehicle_id": 1,
    "type": "car",
    "position": {
        "x": 100,
        "y": 100
    },
    "speed": 30.0,
    "trajectory": [
        [90, 100],
        [95, 100],
        [100, 100]
    ]
}


vehicle2 = {
    "vehicle_id": 2,
    "type": "motorcycle",
    "position": {
        "x": 130,
        "y": 120
    },
    "speed": 25.0,
    "trajectory": [
        [130, 100],
        [130, 110],
        [130, 120]
    ]
}


features = adapter.create_features(
    vehicle1,
    vehicle2,
    object_count=2
)


print("\nFeatures:")
for key, value in features.items():
    print(key, "=", value)


probability = model.predict_probability(features)


print("\n==============================")
print("Near-Miss Probability:", probability)
print("==============================")