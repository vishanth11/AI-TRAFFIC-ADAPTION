from types import SimpleNamespace

from src.integration.controller import TrafficController
from src.integration.decision_engine import DecisionEngine
from src.integration.safety_gate import SafetyGate
from src.integration.sumo_safety import SUMOSafetyRiskAdapter


def _checks(phases, unsafe=(), **extra):
    unsafe = set(unsafe)
    result = {}
    for phase in phases:
        result[phase] = {
            "topology_valid": True,
            "conflict_free": phase not in unsafe,
            "timing_valid": True,
            "pedestrian_clear": True,
            "downstream_available": True,
            "emergency_safe": True,
            "confidence": 0.95,
            **extra,
        }
    return result


def test_high_risk_phase_is_rejected_by_safety_gate():
    result = SafetyGate().validate(
        phase="phase_a",
        topology_valid=True,
        conflict_free=True,
        timing_valid=True,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=0.95,
        safety_risk=0.90,
    )

    assert result["approved"] is False
    assert "risk_valid" in result["reasons"]


def test_confirmed_accident_is_rejected_by_safety_gate():
    result = SafetyGate().validate(
        phase="phase_a",
        topology_valid=True,
        conflict_free=True,
        timing_valid=True,
        pedestrian_clear=True,
        downstream_available=True,
        emergency_safe=True,
        confidence=0.95,
        accident_detected=True,
    )

    assert result["approved"] is False
    assert "accident_safe" in result["reasons"]


def test_fallback_chain_reaches_rule_density_fixed_and_stop():
    controller = TrafficController()
    phases = ["phase_a", "phase_b", "phase_c", "phase_d"]
    scores = {"phase_a": 0.9, "phase_b": 0.8, "phase_c": 0.7, "phase_d": 0.6}
    densities = {"phase_a": 10, "phase_b": 9, "phase_c": 8, "phase_d": 7}

    result = controller.decide_and_validate(
        scores, _checks(phases, unsafe={"phase_a"}), densities, phases
    )
    assert result["method"] == "rule_based_fallback"
    assert result["selected_phase"] == "phase_b"

    result = controller.decide_and_validate(
        scores, _checks(phases, unsafe={"phase_a", "phase_b"}), densities, phases
    )
    assert result["method"] == "density_based_fallback"
    assert result["selected_phase"] == "phase_c"

    result = controller.decide_and_validate(
        scores, _checks(phases, unsafe={"phase_a", "phase_b", "phase_c"}), densities, phases
    )
    assert result["method"] == "safe_fixed_time"
    assert result["selected_phase"] == "phase_d"

    result = controller.decide_and_validate(
        scores, _checks(phases, unsafe=set(phases)), densities, phases
    )
    assert result["method"] == "emergency_safe_stop"
    assert result["execute"] is False


def test_risk_safety_factor_reduces_decision_score():
    engine = DecisionEngine()
    safe = engine.calculate_phase_score(0.7, 0.7, 0.5, 0.5, 1.0)
    risky = engine.calculate_phase_score(0.7, 0.7, 0.5, 0.5, 0.0)
    assert safe > risky


def test_safety_adapter_degrades_when_models_are_unavailable():
    class EmptySimulation:
        def getTime(self):
            return 0.0

        def getDeltaT(self):
            return 1000.0

        def getDepartedIDList(self):
            return []

        def getArrivedIDList(self):
            return []

    class EmptyVehicle:
        def getIDList(self):
            return []

    fake_traci = SimpleNamespace(
        simulation=EmptySimulation(),
        vehicle=EmptyVehicle(),
    )

    class BrokenSafety:
        def _ensure_near_miss(self):
            raise RuntimeError("near miss unavailable")

        def _ensure_accident(self):
            raise RuntimeError("accident unavailable")

    adapter = SUMOSafetyRiskAdapter(fake_traci, safety_intelligence=BrokenSafety())
    result = adapter.assess()

    assert result["degraded"] is True
    assert result["safety_confidence"] == 0.0
    assert result["accident_probability"] is None
    assert result["pet_available"] is False
