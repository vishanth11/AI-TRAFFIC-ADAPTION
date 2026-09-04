# Steps 13 onward — Connected Intelligent Traffic Control

The supplied project already contains Steps 1–12. This continuation **does not replace them**.

## Step 13 — Connected 3-junction simulation
`src/13_connected_junction_simulation.py`

Takes the existing Random Forest predictions from:

`results/baseline_predictions.npy`

The 36 sensors are grouped as:
- Sensors 0–11 → Junction 1
- Sensors 12–23 → Junction 2
- Sensors 24–35 → Junction 3

It produces a normal adaptive signal plan and an emergency-priority plan.

## Step 13A — Emergency green corridor
`src/13a_emergency_corridor.py`

Simulates an emergency vehicle travelling through Junction 1 → Junction 2 → Junction 3.

The route receives signal priority at each connected junction while preserving a minimum cross-direction interval.

This is a simulation policy; it is not a claim that a real traffic signal can be safely controlled without certified hardware and authority approval.

## Step 13B — Confidence-aware control
`src/13b_confidence_aware_signal.py`

Uses variation among the 12 sensors assigned to each junction as a simple uncertainty proxy.

High uncertainty causes the controller to move toward a more conservative timing instead of blindly trusting a prediction.

## Step 13C — Connected-junction visualization
`src/13c_three_junction_demo.py`

Creates:

`results/step13_junction_traffic.png`

## Step 14 — Control evaluation
`src/14_control_evaluation.py`

Reports predicted load, green time, red time, and green-time utilization for the three junctions.

## Step 15 — Run the continuation
`src/15_run_all.py`

Runs all Step 13+ modules in sequence.

## Important
The existing model metrics and trained models from Steps 1–12 are retained. The new steps operate on the existing `baseline_predictions.npy` artifact so the continuation can be demonstrated without retraining.
