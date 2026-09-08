"""Canonical Phase 2.4 end-to-end runtime launcher.

This is a thin entry point over the verified Phase2SUMOController.

Supports:
    - normal headless execution
    - normal SUMO-GUI execution
    - emergency demonstration
    - emergency demonstration with SUMO-GUI

The controller itself remains responsible for:
    - topology discovery
    - demand estimation
    - prediction
    - decision making
    - safety validation
    - fallback
    - TraCI signal execution
    - emergency priority
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from evaluation.phase2_evaluation import MetricAccumulator

from integration.sumo_decision_controller import (
    EMERGENCY_SUMO_CONFIG,
    SUMO_CMD,
    SUMO_CONFIG,
    Phase2SUMOController,
)


def _utc_now():
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(handle, event, **fields):
    """Write one structured JSONL event."""
    handle.write(
        json.dumps(
            {
                "event": event,
                "timestamp": _utc_now(),
                **fields,
            }
        )
        + "\n"
    )
    handle.flush()


def _find_sumo_gui():
    """
    Find the SUMO-GUI executable.

    Search order:
        1. SUMO_HOME/bin/sumo-gui.exe
        2. SUMO_HOME/bin/sumo-gui
        3. PATH
        4. Common Windows SUMO installation paths
    """

    candidates = []

    # ---------------------------------------------------------
    # 1. SUMO_HOME
    # ---------------------------------------------------------
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        sumo_home = Path(sumo_home)

        candidates.extend(
            [
                sumo_home / "bin" / "sumo-gui.exe",
                sumo_home / "bin" / "sumo-gui",
            ]
        )

    # ---------------------------------------------------------
    # 2. PATH
    # ---------------------------------------------------------
    path_gui = shutil.which("sumo-gui")

    if path_gui:
        candidates.append(Path(path_gui))

    # ---------------------------------------------------------
    # 3. Common Windows installation locations
    # ---------------------------------------------------------
    program_files_x86 = os.environ.get("ProgramFiles(x86)")

    if program_files_x86:
        candidates.append(
            Path(program_files_x86)
            / "Eclipse"
            / "Sumo"
            / "bin"
            / "sumo-gui.exe"
        )

    program_files = os.environ.get("ProgramW6432")

    if program_files:
        candidates.append(
            Path(program_files)
            / "Eclipse"
            / "Sumo"
            / "bin"
            / "sumo-gui.exe"
        )

    # ---------------------------------------------------------
    # Return first valid executable
    # ---------------------------------------------------------
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Run the canonical integrated traffic controller"
    )

    # ---------------------------------------------------------
    # Execution mode
    # ---------------------------------------------------------

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the deterministic emergency demonstration",
    )

    parser.add_argument(
        "--emergency",
        action="store_true",
        help="Use the dedicated emergency SUMO scenario",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO using SUMO-GUI instead of headless SUMO",
    )

    parser.add_argument(
        "--gui-delay",
        type=int,
        default=300,
        help="SUMO-GUI delay in milliseconds between simulation steps",
    )

    # ---------------------------------------------------------
    # Simulation configuration
    # ---------------------------------------------------------

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="SUMO configuration file",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup steps before normal control",
    )

    parser.add_argument(
        "--control-interval",
        type=float,
        default=10.0,
        help="Signal control interval in simulation seconds",
    )

    parser.add_argument(
        "--prediction-interval",
        type=int,
        default=10,
        help="Prediction update interval",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="SUMO random seed",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for runtime logs and metrics",
    )

    args = parser.parse_args(argv)

    # =========================================================
    # Determine scenario
    # =========================================================

    emergency_enabled = bool(
        args.demo or args.emergency
    )

    if args.config is not None:

        config = Path(args.config)

    elif emergency_enabled:

        config = Path(EMERGENCY_SUMO_CONFIG)

    else:

        config = Path(SUMO_CONFIG)

    if not config.is_absolute():
        config = ROOT / config

    if not config.exists():
        raise FileNotFoundError(
            f"SUMO configuration file not found: {config}"
        )

    # =========================================================
    # Output directory
    # =========================================================

    output_dir = args.output_dir or (
        ROOT
        / "results"
        / "final_runs"
        / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = output_dir / "runtime.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"

    # =========================================================
    # Find SUMO executable
    # =========================================================

    if args.gui:

        sumo_gui = _find_sumo_gui()

        if sumo_gui is None:

            raise RuntimeError(
                "SUMO-GUI was requested, but sumo-gui.exe "
                "could not be found.\n\n"
                "Please set SUMO_HOME, for example:\n"
                '$env:SUMO_HOME="C:\\Program Files (x86)\\Eclipse\\Sumo"\n\n'
                "Then verify:\n"
                "sumo-gui --version"
            )

        sumo_command = [
            sumo_gui,
            "-c",
            str(config),
            "--seed",
            str(args.seed),
            "--delay",
            str(args.gui_delay),
        ]

    else:

        sumo_command = [
            SUMO_CMD[0],
            "-c",
            str(config),
            "--seed",
            str(args.seed),
        ]

    # =========================================================
    # Print startup information
    # =========================================================

    print()
    print("=" * 70)
    print("AI TRAFFIC ADAPTIVE CONTROL SYSTEM")
    print("=" * 70)

    print(
        f"MODE               : "
        f"{'EMERGENCY DEMO' if emergency_enabled else 'NORMAL'}"
    )

    print(
        f"VISUALIZATION      : "
        f"{'SUMO-GUI' if args.gui else 'HEADLESS SUMO'}"
    )

    print(
        f"SUMO CONFIG        : "
        f"{config.relative_to(ROOT)}"
    )

    print(
        f"STEPS              : {args.steps}"
    )

    print(
        f"WARMUP             : {args.warmup}"
    )

    print(
        f"CONTROL INTERVAL   : "
        f"{args.control_interval}s"
    )

    print(
        f"PREDICTION INTERVAL: "
        f"{args.prediction_interval}s"
    )

    print(
        f"SEED               : {args.seed}"
    )

    print(
        f"SUMO COMMAND       : "
        f"{sumo_command[0]}"
    )

    print("=" * 70)
    print()

    # =========================================================
    # Metrics
    # =========================================================

    metrics = MetricAccumulator()

    result = None

    # =========================================================
    # Runtime log
    # =========================================================

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:

        _write_jsonl(
            log,
            "system_start",
            mode=(
                "emergency_demo"
                if emergency_enabled
                else "normal"
            ),
            visualization=(
                "sumo_gui"
                if args.gui
                else "headless"
            ),
            config=str(
                config.relative_to(ROOT)
            ),
            steps=args.steps,
            warmup_steps=args.warmup,
            control_interval=args.control_interval,
            prediction_interval=args.prediction_interval,
            seed=args.seed,
        )

        # -----------------------------------------------------
        # Decision observer
        # -----------------------------------------------------

        def decision_observer(record):

            _write_jsonl(
                log,
                "decision",
                **record,
            )

            # -------------------------------------------------
            # Human-readable terminal output
            # -------------------------------------------------

            junction = record.get(
                "junction",
                "UNKNOWN",
            )

            method = record.get(
                "method",
                "UNKNOWN",
            )

            target_phase = record.get(
                "target_phase",
                record.get(
                    "selected_phase",
                    "UNKNOWN",
                ),
            )

            emergency_active = record.get(
                "emergency_active",
                False,
            )

            safety = record.get(
                "safety",
                {},
            )

            approved = safety.get(
                "approved",
                False,
            )

            if emergency_active:

                print(
                    f"[EMERGENCY] "
                    f"{junction} | "
                    f"target={target_phase} | "
                    f"method={method} | "
                    f"safety={approved}"
                )

            else:

                print(
                    f"[AI] "
                    f"{junction} | "
                    f"target={target_phase} | "
                    f"method={method} | "
                    f"safety={approved}"
                )

        # =====================================================
        # Create controller
        # =====================================================

        controller = Phase2SUMOController(

            sumo_cmd=sumo_command,

            total_steps=args.steps,

            control_interval=args.control_interval,

            prediction_interval=args.prediction_interval,

            warmup_steps=args.warmup,

            step_observer=metrics.sample,

            decision_observer=decision_observer,

            emergency_enabled=emergency_enabled,
        )

        # =====================================================
        # Run
        # =====================================================

        try:

            print("SYSTEM START")
            print(
                "CANONICAL CONTROLLER: "
                "Phase2SUMOController"
            )

            if args.gui:

                print(
                    "SUMO-GUI will be launched "
                    "automatically by the controller."
                )

            print()

            controller.run()

            # -------------------------------------------------
            # Final metrics
            # -------------------------------------------------

            result = metrics.finalize()

            result.update(
                {
                    "decision_count": len(
                        controller.decisions
                    ),

                    "safety_gate_rejections": sum(
                        not decision["safety"]["approved"]
                        for decision in controller.decisions
                    ),

                    "fallback_decisions": sum(
                        decision["method"]
                        != "ai_adaptive"
                        for decision in controller.decisions
                    ),

                    "emergency_priority_decisions": sum(
                        decision["emergency_active"]
                        for decision in controller.decisions
                    ),

                    "emergency_priority_junctions": sorted(
                        {
                            decision["junction"]
                            for decision in controller.decisions
                            if decision["emergency_active"]
                        }
                    ),
                }
            )

            _write_jsonl(
                log,
                "system_complete",
                **result,
            )

        except Exception as exc:

            _write_jsonl(
                log,
                "system_failure",
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            print()
            print("=" * 70)
            print("SYSTEM FAILURE")
            print("=" * 70)
            print(
                f"{type(exc).__name__}: {exc}"
            )
            print("=" * 70)
            print()

            raise

    # =========================================================
    # Save metrics
    # =========================================================

    metrics_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # Save manifest
    # =========================================================

    manifest_path.write_text(
        json.dumps(
            {
                "config": str(
                    config.relative_to(ROOT)
                ),

                "mode": (
                    "emergency_demo"
                    if emergency_enabled
                    else "normal"
                ),

                "visualization": (
                    "sumo_gui"
                    if args.gui
                    else "headless"
                ),

                "seed": args.seed,

                "steps": args.steps,

                "warmup_steps": args.warmup,

                "control_interval":
                    args.control_interval,

                "prediction_interval":
                    args.prediction_interval,

                "gui_delay":
                    args.gui_delay
                    if args.gui
                    else None,

                "runtime_log":
                    str(
                        log_path.relative_to(ROOT)
                    ),

                "metrics":
                    str(
                        metrics_path.relative_to(ROOT)
                    ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================
    # Final terminal output
    # =========================================================

    print()
    print("=" * 70)
    print("SYSTEM COMPLETE")
    print("=" * 70)

    print(
        f"DECISIONS          : "
        f"{result['decision_count']}"
    )

    print(
        f"SAFETY REJECTIONS  : "
        f"{result['safety_gate_rejections']}"
    )

    print(
        f"FALLBACK DECISIONS : "
        f"{result['fallback_decisions']}"
    )

    print(
        f"EMERGENCY DECISIONS: "
        f"{result['emergency_priority_decisions']}"
    )

    print(
        f"RESULTS            : "
        f"{output_dir.relative_to(ROOT)}"
    )

    print(
        f"RUNTIME LOG        : "
        f"{log_path.relative_to(ROOT)}"
    )

    print(
        f"METRICS            : "
        f"{metrics_path.relative_to(ROOT)}"
    )

    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )