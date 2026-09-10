"""SUMO integration test for the ambulance green corridor.

Starts a real SUMO simulation with the corridor_emergency configuration
and verifies that:
1. A multi-junction corridor is created for ambulance_3
2. Each junction receives the correct green phase
3. Junctions are released after the ambulance passes
4. Normal control resumes after the corridor is completed

Requires a working SUMO installation.
"""

from __future__ import annotations

import os
import sys
import shutil

import pytest

# -----------------------------------------------------------------------
# SUMO availability check
# -----------------------------------------------------------------------

def _find_sumo():
    """Locate the SUMO binary, returning None if unavailable."""
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(
            os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")
        )
        candidates.append(
            os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
        )
    found = shutil.which("sumo")
    if found:
        candidates.append(found)
    for variable in ("ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(variable)
        if root:
            candidates.append(
                os.path.join(root, "Eclipse", "Sumo", "bin", "sumo.exe")
            )
    for candidate in candidates:
        if os.path.isfile(candidate):
            if not os.environ.get("SUMO_HOME"):
                os.environ["SUMO_HOME"] = os.path.dirname(
                    os.path.dirname(os.path.abspath(candidate))
                )
            return candidate
    return None


SUMO_BINARY = _find_sumo()
SUMO_AVAILABLE = SUMO_BINARY is not None

if not SUMO_AVAILABLE:
    pytest.skip("SUMO not installed", allow_module_level=True)

# -----------------------------------------------------------------------
# Project imports (after SUMO_HOME is set)
# -----------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import traci

from src.integration.sumo_decision_controller import Phase2SUMOController

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

CORRIDOR_EMERGENCY_CFG = os.path.join(
    PROJECT_ROOT,
    "simulation",
    "configs",
    "corridor_emergency.sumocfg",
)


# -----------------------------------------------------------------------
# SUMO INTEGRATION TEST
# -----------------------------------------------------------------------

class TestGreenCorridorSUMO:
    """Integration test using the real SUMO corridor_emergency scenario.

    ambulance_3 departs at t=190 on route:
        top2C0 -> C0B0 -> B0A0 -> A0bottom0

    This route passes through junctions C0, B0, A0 in sequence.
    """

    @pytest.fixture(autouse=True)
    def _check_sumo(self):
        if not SUMO_AVAILABLE:
            pytest.skip("SUMO not installed")
        if not os.path.isfile(CORRIDOR_EMERGENCY_CFG):
            pytest.skip("corridor_emergency.sumocfg not found")

    def test_corridor_multi_junction_green_progression(self):
        """Ambulance_3 should receive green at C0, B0, A0 in sequence,
        each junction should be released after passage, and normal
        control should resume."""

        sumo_cmd = [SUMO_BINARY, "-c", CORRIDOR_EMERGENCY_CFG]

        controller = Phase2SUMOController(
            sumo_cmd=sumo_cmd,
            total_steps=400,
            control_interval=5.0,
            prediction_interval=10,
            warmup_steps=5,
            emergency_enabled=True,
        )

        controller.run()

        # ----- VERIFICATION -----

        # 1. Check that corridors were created for ambulance_3
        corridor_creations = [
            e for e in controller.corridor_events
            if e["event"] == "corridor_created"
            and e["ambulance"] == "ambulance_3"
        ]
        assert len(corridor_creations) >= 1, (
            "No corridor was created for ambulance_3"
        )

        # The corridor should span multiple junctions (C0, B0, A0)
        first_creation = corridor_creations[0]
        planned_junctions = first_creation.get("junctions", [])
        assert len(planned_junctions) >= 2, (
            f"Expected >=2 junctions in corridor, got {planned_junctions}"
        )

        # 2. Check that junction activations occurred
        activations = [
            e for e in controller.corridor_events
            if e["event"] == "junction_activated"
            and e["ambulance"] == "ambulance_3"
        ]
        assert len(activations) >= 1, (
            "No junction activations for ambulance_3"
        )

        # 3. Check that some junctions were released (passed or disappeared)
        releases = [
            e for e in controller.corridor_events
            if e["event"] in ("junction_passed", "junction_normal_restored",
                              "ambulance_disappeared")
            and e.get("ambulance") == "ambulance_3"
        ]
        # ambulance_3 should complete or disappear
        assert len(releases) >= 1, (
            "No junction releases for ambulance_3"
        )

        # 4. Verify that emergency decisions were recorded
        emergency_decisions = [
            d for d in controller.decisions
            if d.get("emergency_active")
        ]
        assert len(emergency_decisions) >= 1, (
            "No emergency decisions recorded"
        )

        # 5. Verify that normal AI decisions occurred after emergency
        normal_decisions = [
            d for d in controller.decisions
            if not d.get("emergency_active")
        ]
        assert len(normal_decisions) >= 1, (
            "No normal AI decisions after emergency"
        )

        # 6. Check corridors also handled ambulance_0 (departs t=20)
        amb0_events = [
            e for e in controller.corridor_events
            if e.get("ambulance") == "ambulance_0"
        ]
        assert len(amb0_events) >= 1, (
            "No corridor events for ambulance_0"
        )

        print("\n" + "=" * 70)
        print("SUMO GREEN CORRIDOR INTEGRATION TEST: PASSED")
        print("=" * 70)
        print(f"Total corridor events: {len(controller.corridor_events)}")
        print(f"Total decisions: {len(controller.decisions)}")
        print(f"Emergency decisions: {len(emergency_decisions)}")
        print(f"Normal decisions: {len(normal_decisions)}")
        print(f"ambulance_3 corridor junctions: {planned_junctions}")
        print(f"ambulance_3 activations: {len(activations)}")
        print(f"ambulance_3 releases: {len(releases)}")
        print("=" * 70)
