from src.integration.near_miss_feature_adapter import NearMissFeatureAdapter
from src.safety_intelligence import SafetyIntelligence
from src.integration.decision_engine import DecisionEngine


def test_safety_to_decision_integration():

    print("\n" + "=" * 70)
    print("Testing Safety Intelligence → Decision Engine")
    print("=" * 70)

    # ==========================================================
    # 1. CREATE TWO VEHICLES
    # ==========================================================

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

    # ==========================================================
    # 2. GENERATE NEAR-MISS FEATURES
    # ==========================================================

    adapter = NearMissFeatureAdapter()

    features = adapter.create_features(
        vehicle1,
        vehicle2,
        object_count=2
    )

    print("\nNear-Miss Features:")
    print(features)

    # ==========================================================
    # 3. RUN SAFETY INTELLIGENCE
    # ==========================================================

    safety = SafetyIntelligence()

    safety_result = safety.assess_event(
        near_miss_features=features,
        accident_probability=0.30,
        severity=0.50,
        event_age_hours=0
    )

    print("\nSafety Intelligence:")
    print(
        "Near-Miss Probability:",
        safety_result["near_miss_probability"]
    )

    print(
        "Accident Probability:",
        safety_result["accident_probability"]
    )

    print(
        "Risk Score:",
        safety_result["risk_score"]
    )

    print(
        "Risk Level:",
        safety_result["risk_level"]
    )

    # ==========================================================
    # 4. VALIDATE SAFETY RESULT
    # ==========================================================

    assert 0 <= safety_result["near_miss_probability"] <= 1

    assert 0 <= safety_result["accident_probability"] <= 1

    assert 0 <= safety_result["risk_score"] <= 100

    assert safety_result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    print("\nSafety Intelligence: PASS")

    # ==========================================================
    # 5. CREATE DECISION ENGINE
    # ==========================================================

    engine = DecisionEngine()

    # ==========================================================
    # 6. USE SAFETY RISK IN PHASE SCORING
    # ==========================================================
    #
    # IMPORTANT:
    #
    # The current Decision Engine expects a [0,1] value.
    #
    # Member 4 risk score is 0-100.
    #
    # Therefore normalize it here.
    #
    # This is only an integration test.
    # We will later define exactly how risk affects
    # signal selection.
    #

    normalized_risk = (
        safety_result["risk_score"] / 100.0
    )

    print(
        "\nNormalized Risk:",
        normalized_risk
    )

    assert 0 <= normalized_risk <= 1

    # ==========================================================
    # 7. CALCULATE TWO EXAMPLE PHASE SCORES
    # ==========================================================

    phase_scores = {}

    # Phase 1:
    # High predicted demand but exposed to the detected risk.

    phase_scores["Phase_1"] = engine.calculate_phase_score(
        demand=0.80,
        prediction=0.70,
        priority=0.50,
        fairness=0.60,
        risk=normalized_risk
    )

    # Phase 2:
    # Slightly lower traffic demand but lower safety risk.

    phase_scores["Phase_2"] = engine.calculate_phase_score(
        demand=0.60,
        prediction=0.50,
        priority=0.50,
        fairness=0.60,
        risk=0.10
    )

    print("\nPhase Scores:")

    for phase, score in phase_scores.items():
        print(
            f"{phase}: {score:.6f}"
        )

    # ==========================================================
    # 8. SELECT BEST PHASE
    # ==========================================================

    result = engine.select_best_phase(
        phase_scores
    )

    print("\nDecision:")
    print(
        "Selected Phase:",
        result["selected_phase"]
    )

    print(
        "Score:",
        result["score"]
    )

    # ==========================================================
    # 9. VALIDATE FINAL DECISION
    # ==========================================================

    assert result["selected_phase"] in phase_scores

    assert 0 <= result["score"] <= 1

    assert len(
        result["all_scores"]
    ) == 2

    print("\nDecision Engine: PASS")

    print("\n" + "=" * 70)
    print("SAFETY → DECISION INTEGRATION PASSED")
    print("=" * 70)