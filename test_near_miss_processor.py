from src.integration.near_miss_processor import NearMissProcessor


processor = NearMissProcessor()


detections = [
    {
        "vehicle_id": 1,
        "type": "car",
        "confidence": 0.92,
        "position": {
            "x": 100,
            "y": 100
        },
        "speed": 30.0,
        "direction": "right",
        "trajectory": [
            [90, 100],
            [95, 100],
            [100, 100]
        ],
        "tracking_status": "active",
        "emergency": False
    },

    {
        "vehicle_id": 2,
        "type": "motorcycle",
        "confidence": 0.88,
        "position": {
            "x": 130,
            "y": 120
        },
        "speed": 25.0,
        "direction": "down",
        "trajectory": [
            [130, 100],
            [130, 110],
            [130, 120]
        ],
        "tracking_status": "active",
        "emergency": False
    }
]


results = processor.process_frame(detections)


print("Number of vehicle pairs:", len(results))

for result in results:

    print("\nVehicle pair:")
    print(
        result["vehicle_1"],
        "<->",
        result["vehicle_2"]
    )

    print("\nFeatures:")

    for key, value in result["features"].items():
        print(key, "=", value)