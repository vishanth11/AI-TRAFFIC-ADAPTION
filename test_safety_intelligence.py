from src.integration.near_miss_feature_adapter import NearMissFeatureAdapter
from safety_intelligence import SafetyIntelligence


def test_safety_intelligence():

    adapter = NearMissFeatureAdapter()

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

    near_miss_features = adapter.create_features(
        vehicle1,
        vehicle2,
        object_count=2
    )

    safety = SafetyIntelligence()

    result = safety.assess_event(
        near_miss_features=near_miss_features,
        accident_probability=0.3,
        severity=0.5,
        event_age_hours=0
    )

    print("\n==============================")
    print("SAFETY INTELLIGENCE RESULT")
    print("==============================")

    print("Near-Miss Probability:",
          result["near_miss_probability"])

    print("Accident Probability:",
          result["accident_probability"])

    print("Risk Score:",
          result["risk_score"])

    print("Risk Level:",
          result["risk_level"])

    print("==============================")

    assert result["near_miss_probability"] is not None
    assert result["accident_probability"] == 0.3
    assert isinstance(result["risk_score"], float)
    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]