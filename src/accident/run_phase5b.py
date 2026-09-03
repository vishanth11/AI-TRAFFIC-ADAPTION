"""Phase 5B experiment driver.

Usage:
    python -m accident.run_phase5b EXP_A
    python -m accident.run_phase5b EXP_B
    python -m accident.run_phase5b ALL

Experiment definitions:
    EXP_A  correctness rerun  — fixed label mapping, otherwise the Phase 5
           configuration (3 epochs) to prove the mapping fix alone recovers
           accident AP.
    EXP_B  longer training    — same configuration trained 25 epochs with
           early stopping (patience 6) and per-epoch checkpointing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from accident.phase5b import PHASE5B_CONFIG, run_experiment
from accident.utils import REPORT_DIR

EXPERIMENTS = {
    "EXP_A": {
        "config": {**PHASE5B_CONFIG, "epochs": 3, "early_stopping_patience": 0},
        "notes": "Correctness rerun: fixed background/label mapping (+1 offset), "
        "otherwise identical to Phase 5 config; 3 epochs.",
        "augmentation": "none",
        "best_model_name": "phase5b_expA_best.pth",
    },
    "EXP_B": {
        "config": {**PHASE5B_CONFIG},
        "notes": "Longer training: fixed mapping, 25 epochs, early stopping "
        "patience 6, StepLR decay at epoch 8.",
        "augmentation": "none",
        "best_model_name": "phase5b_expB_best.pth",
    },
}


def main() -> int:
    targets = sys.argv[1:] or ["ALL"]
    resume = "RESUME" in targets
    targets = [t for t in targets if t != "RESUME"]
    if "ALL" in targets:
        targets = ["EXP_A", "EXP_B"]

    interrupted = Path("models/accident/phase5b_interrupted_checkpoint.pth")

    for name in targets:
        spec = EXPERIMENTS[name]
        config = dict(spec["config"])
        notes = spec["notes"]
        if resume and interrupted.exists():
            config["resume_from"] = str(interrupted.resolve())
            notes = notes + " | resumed from phase5b_interrupted_checkpoint.pth"
        summary = run_experiment(
            config=config,
            experiment_id=name,
            notes=notes,
            augmentation=spec["augmentation"],
            best_model_name=spec["best_model_name"],
        )
        print(
            f"\n[{name}] DONE — best epoch {summary['best_epoch']}: "
            f"accident AP@50={summary['val_accident_ap50']:.4f} "
            f"mAP@50={summary['val_map50']:.4f} "
            f"recall={summary['val_accident_recall']:.4f} "
            f"({summary['total_training_seconds']:.0f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())