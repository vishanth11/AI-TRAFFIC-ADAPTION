from src.integration.fallback_controller import FallbackController


def test_fallback_controller():

    fallback = FallbackController()

    phase_scores = {
        "Phase_1": 0.4,
        "Phase_2": 0.8,
        "Phase_3": 0.5
    }

    result = fallback.rule_based(phase_scores)

    assert result["method"] == "rule_based"
    assert result["selected_phase"] == "Phase_2"

    phase_densities = {
        "Phase_1": 10,
        "Phase_2": 25,
        "Phase_3": 15
    }

    result = fallback.density_based(phase_densities)

    assert result["method"] == "density_based"
    assert result["selected_phase"] == "Phase_2"

    phases = [
        "Phase_1",
        "Phase_2",
        "Phase_3"
    ]

    result = fallback.fixed_time(
        phases,
        current_index=1
    )

    assert result["method"] == "fixed_time"
    assert result["selected_phase"] == "Phase_2"

    print("\nFALLBACK CONTROLLER TEST PASSED")