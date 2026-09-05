from src.integration.safety_gate import SafetyGate


def test_safety_gate():

    gate = SafetyGate()

    # Test 1: Safe phase
    safe_result = gate.validate(
        phase="Phase_1",
        topology_valid=True,
        conflict_free=True,
        timing_valid=True,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=0.90
    )

    print("\nSafe decision:")
    print(safe_result)

    assert safe_result["approved"] is True
    assert safe_result["reasons"] == []

    # Test 2: Conflicting phase
    unsafe_result = gate.validate(
        phase="Phase_2",
        topology_valid=True,
        conflict_free=False,
        timing_valid=True,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=0.90
    )

    print("\nUnsafe decision:")
    print(unsafe_result)

    assert unsafe_result["approved"] is False
    assert "conflict_free" in unsafe_result["reasons"]

    # Test 3: Low confidence
    low_confidence_result = gate.validate(
        phase="Phase_3",
        topology_valid=True,
        conflict_free=True,
        timing_valid=True,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=0.30
    )

    print("\nLow confidence decision:")
    print(low_confidence_result)

    assert low_confidence_result["approved"] is False
    assert "confidence_valid" in low_confidence_result["reasons"]

    print("\nSAFETY GATE TEST PASSED")
