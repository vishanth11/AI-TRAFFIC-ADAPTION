# Final Integrated Traffic System

## Scope

This repository contains a topology-adaptive SUMO/TraCI traffic controller validated in the provided corridor. The canonical runtime is `Phase2SUMOController`; `run_system.py` is the reproducible launcher and does not implement a second controller.

## Architecture

```text
SUMO / TraCI
  -> dynamic signalized-junction discovery
  -> topology, movements, phase-derived conflict graph
  -> valid green phases
  -> SUMO traffic state and 10 x 36 history
  -> Member-3 GNN prediction
  -> movement and phase demand
  -> near-miss safety risk
  -> DecisionEngine
  -> emergency priority when present
  -> SafetyGate
  -> AI/rule/density/fixed-time/safe-stop fallback
  -> yellow transition and TraCI phase execution
  -> phase readback and metrics
```

The current conflict graph derives movement compatibility from the existing SUMO signal program. It is not a universal geometry-independent physical conflict detector.

The 36 deployment sensors are topology-derived observation zones. They are not claimed to reproduce Member 3's original physical sensor locations.

## Canonical Commands

Normal runtime:

```powershell
python run_system.py --steps 200 --seed 1
```

Emergency demonstration:

```powershell
python run_system.py --demo --steps 200 --seed 1
```

The emergency demonstration uses the separate `simulation/configs/corridor_emergency.sumocfg` scenario and leaves the normal corridor configuration unchanged.

Phase 2.1 baseline evaluation:

```powershell
python src\evaluation\phase2_evaluation.py --seeds 1 2 3 --steps 200 --warmup 10
```

Targeted tests:

```powershell
python -m pytest test_phase2_evaluation.py test_phase2_2_safety.py test_phase2_3_emergency.py test_controller.py test_fallback_controller.py test_safety_decision_integration.py test_safety_gate.py -q
```

Full suite:

```powershell
python -m pytest -q
```

The runtime resolves SUMO through `SUMO_HOME`, PATH, or the standard Windows SUMO installation path. No developer-specific absolute project path is required.

## Normal Control

Normal operation discovers signalized junctions dynamically, builds topology and phase representations, collects SUMO traffic into the 36-sensor deployment mapping, updates `TrafficHistory`, runs the GNN, calculates demand, computes safety risk, scores phases with the existing `DecisionEngine`, validates through `SafetyGate`, and executes only approved phases through TraCI.

## Safety

The SUMO safety adapter uses the existing calibrated near-miss Model D for selected closest telemetry pairs. TTC is reported only when relative motion is sufficient. PET is explicitly unavailable without a validated conflict point. The accident detector is image-based; SUMO telemetry does not provide a camera frame, so the runtime reports that accident inference is unavailable rather than fabricating an image or probability.

Model or telemetry failures produce degraded safety state and do not silently become a confident no-risk result.

## Emergency Control

Emergency vehicles are identified from SUMO vehicle type/classification. The emergency state includes route, edge, lane, position, speed, distance, current/next junction, destination, confidence, and ETA.

The `EmergencyCorridorManager` maps route edges to discovered movements and selects only compatible green phases containing the required movement. It does not directly call TraCI signal APIs. The normal controller performs SafetyGate validation and safe yellow/all-red transition handling before execution.

After the vehicle leaves the active set, an explicit release event is emitted and normal AI control resumes.

## Fallback

Every normal and emergency candidate uses the existing chain:

```text
AI adaptive -> rule based -> density based -> fixed time -> safe stop
```

An unsafe phase is never executed directly.

## Logging and Results

`run_system.py` writes:

```text
results/final_runs/<run>/runtime.jsonl
results/final_runs/<run>/metrics.json
results/final_runs/<run>/manifest.json
```

The JSONL contains system start/completion/failure events and executed decision records, including junction, selected method, safety result, executed phase, and emergency fields when present.

The Phase 2.1 evaluator writes raw controller results, summaries, and comparison plots under `results/evaluation/`.

## Known Limitations

- Validation is against the supplied SUMO corridor and a dedicated emergency corridor scenario.
- Conflict compatibility is signal-program-derived.
- PET and image-based accident inference are unavailable in SUMO-only mode.
- Pedestrian and downstream checks remain explicit Phase-2 assumptions.
- The GNN uses the fixed 10 x 36 input contract and topology-derived deployment mapping.
- Full repository collection still includes legacy dependency/import failures unrelated to the canonical runtime, including missing `pycocotools`, old import assumptions, and SUMO tests that invoke `sumo` without the runtime resolver.
- This is not a claim of real-world traffic-control readiness or real-world emergency-response improvement.
