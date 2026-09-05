"""Canonical Phase 2.4 end-to-end runtime launcher.

This is a thin entry point over the verified Phase2SUMOController. It does
not implement a second controller. It adds reproducible configuration,
structured JSONL decision logging, and final TraCI metrics.
"""

from __future__ import annotations

import argparse
import json
import os
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
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(handle, event, **fields):
    handle.write(json.dumps({"event": event, "timestamp": _utc_now(), **fields}) + "\n")
    handle.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the canonical integrated traffic controller")
    parser.add_argument("--demo", action="store_true", help="Run the deterministic emergency demonstration")
    parser.add_argument("--emergency", action="store_true", help="Use the dedicated emergency SUMO scenario")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--control-interval", type=float, default=10.0)
    parser.add_argument("--prediction-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    emergency_enabled = bool(args.demo or args.emergency)
    config = Path(args.config or (EMERGENCY_SUMO_CONFIG if emergency_enabled else SUMO_CONFIG))
    if not config.is_absolute():
        config = ROOT / config
    output_dir = args.output_dir or (
        ROOT / "results" / "final_runs" /
        datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runtime.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"

    metrics = MetricAccumulator()
    with log_path.open("w", encoding="utf-8") as log:
        _write_jsonl(
            log,
            "system_start",
            mode="emergency_demo" if emergency_enabled else "normal",
            config=str(config.relative_to(ROOT)),
            steps=args.steps,
            warmup_steps=args.warmup,
            control_interval=args.control_interval,
            prediction_interval=args.prediction_interval,
            seed=args.seed,
        )

        def decision_observer(record):
            _write_jsonl(log, "decision", **record)

        command = [SUMO_CMD[0], "-c", str(config), "--seed", str(args.seed)]
        controller = Phase2SUMOController(
            sumo_cmd=command,
            total_steps=args.steps,
            control_interval=args.control_interval,
            prediction_interval=args.prediction_interval,
            warmup_steps=args.warmup,
            step_observer=metrics.sample,
            decision_observer=decision_observer,
            emergency_enabled=emergency_enabled,
        )
        try:
            print("SYSTEM START")
            print("CANONICAL CONTROLLER: Phase2SUMOController")
            controller.run()
            result = metrics.finalize()
            result.update({
                "decision_count": len(controller.decisions),
                "safety_gate_rejections": sum(
                    not decision["safety"]["approved"]
                    for decision in controller.decisions
                ),
                "fallback_decisions": sum(
                    decision["method"] != "ai_adaptive"
                    for decision in controller.decisions
                ),
                "emergency_priority_decisions": sum(
                    decision["emergency_active"]
                    for decision in controller.decisions
                ),
                "emergency_priority_junctions": sorted({
                    decision["junction"]
                    for decision in controller.decisions
                    if decision["emergency_active"]
                }),
            })
            _write_jsonl(log, "system_complete", **result)
        except Exception as exc:
            _write_jsonl(
                log,
                "system_failure",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "config": str(config.relative_to(ROOT)),
                "mode": "emergency_demo" if emergency_enabled else "normal",
                "seed": args.seed,
                "steps": args.steps,
                "warmup_steps": args.warmup,
                "control_interval": args.control_interval,
                "prediction_interval": args.prediction_interval,
                "runtime_log": str(log_path.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RESULTS: {output_dir.relative_to(ROOT)}")
    print(f"RUNTIME LOG: {log_path.relative_to(ROOT)}")
    print(f"METRICS: {metrics_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
