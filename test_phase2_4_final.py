from pathlib import Path

from run_system import ROOT
from src.integration.sumo_decision_controller import (
    EMERGENCY_SUMO_CONFIG,
    SUMO_CONFIG,
    Phase2SUMOController,
)


def test_canonical_runtime_and_scenarios_exist():
    assert Path(SUMO_CONFIG).exists()
    assert Path(EMERGENCY_SUMO_CONFIG).exists()
    assert Phase2SUMOController.__module__.endswith(
        "integration.sumo_decision_controller"
    )
    assert ROOT.exists()


def test_final_output_contract_is_project_relative():
    assert not str(ROOT).startswith("E:")
    assert (ROOT / "src" / "integration" / "emergency.py").exists()
    assert (ROOT / "docs" / "FINAL_SYSTEM.md").exists()
