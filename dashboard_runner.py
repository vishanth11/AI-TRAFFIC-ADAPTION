"""Dashboard runtime bridge.

Wraps Phase2SUMOController with observers that populate DashboardState.
The existing controller is NOT modified.

Usage:
    python dashboard_runner.py [--emergency] [--steps N] [--gui]
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn

from dashboard_state import state
from dashboard_api import app
from integration.sumo_decision_controller import Phase2SUMOController, SUMO_CMD, EMERGENCY_SUMO_CONFIG


def _build_step_observer(traci_ref: list, predictor_ref: list):
    """Returns a step observer that samples TraCI state into DashboardState."""

    def observer():
        traci = traci_ref[0]
        if traci is None:
            return
        try:
            sim_time = float(traci.simulation.getTime())
            vehicle_ids = list(traci.vehicle.getIDList())
            state.update_vehicles(len(vehicle_ids))
            state.update_system(
                "running",
                sim_time,
                prediction_available=predictor_ref[0] is not None,
            )
        except Exception:
            pass

    return observer


def _build_decision_observer(traci_ref: list):
    """Returns a decision observer that records decisions and junction state."""

    def observer(record: dict):
        state.add_decision(record)
        state.add_log("decision", **record)

        traci = traci_ref[0]
        if traci is None:
            return
        try:
            junction_id = record.get("junction")
            if junction_id:
                sim_time = float(traci.simulation.getTime())
                # Collect per-junction data
                try:
                    phase_index = traci.trafficlight.getPhase(junction_id)
                    phase_state = traci.trafficlight.getRedYellowGreenState(junction_id)
                    next_switch = traci.trafficlight.getNextSwitch(junction_id)
                    remaining = max(0.0, next_switch - sim_time)
                except Exception:
                    phase_index = None
                    phase_state = None
                    remaining = None

                safety = record.get("safety", {})
                state.update_junction(junction_id, {
                    "current_phase_index": phase_index,
                    "current_phase_state": phase_state,
                    "remaining_time": remaining,
                    "last_decision": record.get("selected") or record.get("selected_phase"),
                    "decision_method": record.get("method"),
                    "safety_approved": safety.get("approved"),
                    "safety_checks": safety.get("checks"),
                    "emergency_active": record.get("emergency_active", False),
                    "emergency_vehicle_id": record.get("emergency_vehicle_id"),
                    "simulation_time": sim_time,
                })
        except Exception:
            pass

    return observer


def run_controller(emergency: bool, steps: int, gui: bool):
    import traci as _traci

    traci_ref = [None]
    predictor_ref = [None]

    # Monkey-patch traci.start to capture the connection
    _original_start = _traci.start

    def _patched_start(cmd, **kwargs):
        result = _original_start(cmd, **kwargs)
        traci_ref[0] = _traci
        return result

    _traci.start = _patched_start

    step_obs = _build_step_observer(traci_ref, predictor_ref)
    decision_obs = _build_decision_observer(traci_ref)

    import shutil, os
    sumo_cmd = list(SUMO_CMD)
    if gui:
        gui_exe = shutil.which("sumo-gui")
        if gui_exe:
            sumo_cmd = [gui_exe, sumo_cmd[1], sumo_cmd[2]]

    if emergency:
        sumo_cmd = [sumo_cmd[0], "-c", str(EMERGENCY_SUMO_CONFIG)]

    state.add_log("system_start", mode="emergency_demo" if emergency else "normal")
    state.update_system("running", 0.0)

    controller = Phase2SUMOController(
        sumo_cmd=sumo_cmd,
        total_steps=steps,
        emergency_enabled=emergency,
        step_observer=step_obs,
        decision_observer=decision_obs,
    )

    # Patch predictor availability after build
    _orig_run = controller.run

    def _patched_run():
        try:
            _orig_run()
        finally:
            state.update_system("stopped", state.simulation_time)
            state.add_log("system_complete")

    controller.run = _patched_run
    controller.run()


def main():
    parser = argparse.ArgumentParser(description="Run dashboard + traffic controller")
    parser.add_argument("--emergency", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--api-only", action="store_true",
                        help="Start only the API server without running SUMO")
    args = parser.parse_args()

    if not args.api_only:
        # Run controller in background thread
        t = threading.Thread(
            target=run_controller,
            args=(args.emergency, args.steps, args.gui),
            daemon=True,
        )
        t.start()

    # Run FastAPI in main thread
    uvicorn.run(app, host="0.0.0.0", port=args.api_port, log_level="warning")


if __name__ == "__main__":
    main()
