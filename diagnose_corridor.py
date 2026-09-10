"""Diagnostic v5: trace why B0 stays at phase 2 (top1B0 RED) during t=36..50.

Log active_plans for B0 each step + downstream occupancy + ambulance ETAs.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import traci

from src.integration.sumo_decision_controller import Phase2SUMOController


def _find_sumo():
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe"))
    found = shutil.which("sumo")
    if found:
        candidates.append(found)
    for variable in ("ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(variable)
        if root:
            candidates.append(os.path.join(root, "Eclipse", "Sumo", "bin", "sumo.exe"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_sumo_gui():
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe"))
    found = shutil.which("sumo-gui")
    if found:
        candidates.append(found)
    for variable in ("ProgramFiles(x86)", "ProgramW6432"):
        root = os.environ.get(variable)
        if root:
            candidates.append(os.path.join(root, "Eclipse", "Sumo", "bin", "sumo-gui.exe"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


SUMO_BINARY = _find_sumo()
SUMO_GUI_BINARY = _find_sumo_gui()
CFG = os.path.join(PROJECT_ROOT, "simulation", "configs", "corridor_emergency.sumocfg")


class DiagnosticController(Phase2SUMOController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def _update_corridor_plans(self, emergency_states, topologies, runtimes, emergency_manager):
        active_plans = super()._update_corridor_plans(
            emergency_states, topologies, runtimes, emergency_manager
        )
        now = self.traci.simulation.getTime()
        if 20 <= now <= 95:
            row = {"t": now}
            # B0 signal
            try:
                row["B0_phase"] = self.traci.trafficlight.getPhase("B0")
            except Exception:
                pass
            # active plan for B0
            plan = active_plans.get("B0")
            if plan:
                row["active_plan"] = f"{plan.vehicle_id} phase={plan.selected_phase} eta={plan.eta_seconds} dist={plan.distance_to_junction} urg={plan.urgency:.3f}"
            else:
                row["active_plan"] = "NONE"
            # ambulance states
            for st in emergency_states:
                if st.vehicle_id in ("ambulance_0", "ambulance_1"):
                    row[f"{st.vehicle_id}"] = (
                        f"edge={st.current_edge} pos={st.current_position} "
                        f"speed={st.speed_mps} dist={st.distance_to_junction} "
                        f"eta={st.eta_seconds} junc={st.current_junction}"
                    )
            # downstream occupancy for B0 (B0bottom1 for amb1, B0C0 for amb0)
            for edge in ("B0bottom1", "B0C0"):
                try:
                    row[f"occ_{edge}"] = round(self.traci.edge.getLastStepOccupancy(edge), 3)
                except Exception:
                    pass
            # C0 phase + exit edges
            try:
                row["C0_phase"] = self.traci.trafficlight.getPhase("C0")
            except Exception:
                pass
            for edge in ("C0top2", "C0bottom2", "C0right0", "C0B0"):
                try:
                    row[f"occ_{edge}"] = round(self.traci.edge.getLastStepOccupancy(edge), 3)
                except Exception:
                    pass
            # internal connectors of C0 (cars stuck inside the junction)
            try:
                internal = []
                for edge_id in self.traci.edge.getIDList():
                    if edge_id.startswith(":C0_"):
                        vehs = self.traci.edge.getLastStepVehicleIDs(edge_id)
                        for vid in vehs:
                            pos = self.traci.vehicle.getLanePosition(vid)
                            internal.append(f"{vid}@{edge_id}@{pos:.0f}")
                row["C0_internal"] = ",".join(internal)
            except Exception:
                pass
            # cars queued on B0C0 (the shared exit edge for top1B0+bottom1B0)
            try:
                vehs = self.traci.edge.getLastStepVehicleIDs("B0C0")
                queued = []
                for vid in vehs:
                    pos = self.traci.vehicle.getLanePosition(vid)
                    queued.append(f"{vid}@{pos:.0f}")
                row["B0C0_cars"] = ",".join(sorted(queued, key=lambda s: -float(s.split("@")[1])))
            except Exception:
                pass
            # cars ahead of ambulance_1 on top1B0
            try:
                vehs = self.traci.edge.getLastStepVehicleIDs("top1B0")
                ahead = []
                for vid in vehs:
                    if vid.startswith("b0_car") or vid.startswith("flow"):
                        pos = self.traci.vehicle.getLanePosition(vid)
                        ahead.append(f"{vid}@{pos:.0f}")
                row["top1B0_cars"] = ",".join(sorted(ahead, key=lambda s: -float(s.split("@")[1])))
            except Exception:
                pass
            # internal connectors of B0 (cars stuck inside the junction)
            try:
                internal = []
                for edge_id in self.traci.edge.getIDList():
                    if edge_id.startswith(":B0_"):
                        vehs = self.traci.edge.getLastStepVehicleIDs(edge_id)
                        for vid in vehs:
                            pos = self.traci.vehicle.getLanePosition(vid)
                            internal.append(f"{vid}@{edge_id}@{pos:.0f}")
                row["B0_internal"] = ",".join(internal)
            except Exception:
                pass
            # B0A0 occupancy (left-turn exit for flow_top1_to_left)
            try:
                row["occ_B0A0"] = round(self.traci.edge.getLastStepOccupancy("B0A0"), 3)
            except Exception:
                pass
            self.rows.append(row)
        return active_plans


def main():
    parser = argparse.ArgumentParser(description="Diagnose emergency green-corridor behavior in SUMO.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the simulation in SUMO-GUI instead of the headless sumo.exe.",
    )
    args = parser.parse_args()

    binary = SUMO_GUI_BINARY if args.gui else SUMO_BINARY
    if binary is None:
        raise SystemExit(
            f"Could not find {'sumo-gui.exe' if args.gui else 'sumo.exe'} on PATH or in the standard SUMO install dirs."
        )

    controller = DiagnosticController(
        sumo_cmd=[binary, "-c", CFG],
        total_steps=400,
        control_interval=5.0,
        prediction_interval=10,
        warmup_steps=5,
        emergency_enabled=True,
    )
    controller.run()

    print("\n" + "=" * 130)
    print("ACTIVE PLANS FOR B0 (t=20..95)")
    print("=" * 130)
    for r in controller.rows:
        parts = [f"t={r['t']:5.1f} B0=ph{r.get('B0_phase')} C0=ph{r.get('C0_phase')} plan={r.get('active_plan')}"]
        for a in ("ambulance_0", "ambulance_1"):
            if a in r:
                parts.append(f"{a}[{r[a]}]")
        for e in ("occ_B0bottom1", "occ_B0C0", "occ_C0top2", "occ_C0bottom2", "occ_C0right0", "occ_C0B0"):
            if e in r:
                parts.append(f"{e}={r[e]}")
        if "top1B0_cars" in r:
            parts.append(f"top1B0=[{r['top1B0_cars']}]")
        if "B0_internal" in r:
            parts.append(f"B0int=[{r['B0_internal']}]")
        if "C0_internal" in r:
            parts.append(f"C0int=[{r['C0_internal']}]")
        if "B0C0_cars" in r:
            parts.append(f"B0C0=[{r['B0C0_cars']}]")
        if "occ_B0A0" in r:
            parts.append(f"occ_B0A0={r['occ_B0A0']}")
        print(" | ".join(parts))

    # All emergency decisions t=20..60
    print("\n" + "=" * 130)
    print("ALL EMERGENCY DECISIONS t=20..60")
    print("=" * 130)
    for d in controller.decisions:
        if 20 <= d["time"] <= 60 and d.get("emergency_active"):
            print(
                f"t={d['time']:.1f} j={d['junction']} phase={d['selected']} "
                f"veh={d.get('emergency_vehicle_id')} eta={d.get('emergency_eta')} "
                f"dist={d.get('emergency_distance')}"
            )


if __name__ == "__main__":
    main()
