from src.integration.controller import TrafficController


def test_controller():

    controller = TrafficController()

    phase_scores = {
        "Phase_1": 0.80,
        "Phase_2": 0.60
    }

    phase_checks = {
        "Phase_1": {
            "topology_valid": True,
            "conflict_free": True,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        },
        "Phase_2": {
            "topology_valid": True,
            "conflict_free": True,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        }
    }

    result = controller.decide_and_validate(
        phase_scores,
        phase_checks
    )

    print("\nAI Controller Result:")
    print(result)

    assert result["selected_phase"] == "Phase_1"
    assert result["method"] == "ai_adaptive"
    assert result["execute"] is True


def test_rule_based_fallback():

    controller = TrafficController()

    phase_scores = {
        "Phase_1": 0.80,
        "Phase_2": 0.60
    }

    phase_checks = {
        "Phase_1": {
            "topology_valid": True,
            "conflict_free": False,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        },
        "Phase_2": {
            "topology_valid": True,
            "conflict_free": True,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        }
    }

    result = controller.decide_and_validate(
        phase_scores,
        phase_checks
    )

    print("\nRule-Based Fallback Result:")
    print(result)

    assert result["selected_phase"] == "Phase_2"
    assert result["method"] == "rule_based_fallback"
    assert result["safety"]["approved"] is True
    assert result["execute"] is True


def test_density_fallback():

    controller = TrafficController()

    phase_scores = {
        "Phase_1": 0.90,
        "Phase_2": 0.80
    }

    phase_densities = {
        "Phase_1": 30,
        "Phase_2": 50
    }

    phase_checks = {
        "Phase_1": {
            "topology_valid": True,
            "conflict_free": False,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        },
        "Phase_2": {
            "topology_valid": True,
            "conflict_free": False,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        },
        "Phase_3": {
            "topology_valid": True,
            "conflict_free": True,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        }
    }

    phase_scores["Phase_3"] = 0.70
    phase_densities["Phase_3"] = 20

    result = controller.decide_and_validate(
        phase_scores,
        phase_checks,
        phase_densities=phase_densities,
        phases=list(phase_scores.keys())
    )

    print("\nDensity Fallback Result:")
    print(result)

    assert result["selected_phase"] == "Phase_3"
    assert result["method"] == "density_based_fallback"
    assert result["execute"] is True


def test_no_safe_phase():

    controller = TrafficController()

    phase_scores = {
        "Phase_1": 0.80,
        "Phase_2": 0.60
    }

    phase_checks = {
        "Phase_1": {
            "topology_valid": True,
            "conflict_free": False,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        },
        "Phase_2": {
            "topology_valid": True,
            "conflict_free": False,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.90
        }
    }

    result = controller.decide_and_validate(
        phase_scores,
        phase_checks
    )

    print("\nNo Safe Phase Result:")
    print(result)

    assert result["method"] == "emergency_safe_stop"
    assert result["selected_phase"] is None
    assert result["execute"] is False

    print("\nFAULT-TOLERANT CONTROLLER TEST PASSED")