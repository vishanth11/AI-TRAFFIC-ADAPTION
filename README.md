# AI Traffic Adaptation

## Project Overview

This project runs an integrated 16-junction multimodal SUMO environment with
AI adaptive traffic signal control. The runtime combines:

- Traffic-state history and prediction
- Emergency and ambulance priority
- Safety-gated signal decisions with fallback behavior
- TraCI closed-loop integration with SUMO

The canonical launcher is `run_system.py`. Its default scenario is
`simulation/configs/large_grid_multimodal.sumocfg`.

## Prerequisites

- Windows with Python installed and available as `python`
- Eclipse SUMO, including `sumo`, and optionally `sumo-gui`
- Python packages `traci` and `sumolib`
- Runtime packages used by the controller, including `numpy`, `pandas`, and
  `torch`
- `pytest` for the regression suite

The repository also contains the pinned accident-model environment in
`requirements-accident.txt`. Install it when working with the accident-model
pipeline in addition to the SUMO runtime packages.

## Installation and Setup on Windows

From PowerShell at the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install traci sumolib numpy pandas torch pytest
```

For the complete pinned accident-model dependency set:

```powershell
python -m pip install -r requirements-accident.txt
```

If PowerShell does not allow script activation, use the Python executable in
the environment directly, for example:

```powershell
.\.venv\Scripts\python.exe --version
```

SUMO can be found through `PATH`, `SUMO_HOME`, or the standard Windows Eclipse
SUMO installation locations. When using `SUMO_HOME`, it should point to the
SUMO installation directory, such as:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

## Verify Installation

Run these checks from the repository root:

```powershell
python --version
sumo --version
python -c "import traci, sumolib; print('TraCI/SUMO OK')"
```

For GUI runs, also verify:

```powershell
sumo-gui --version
```

## How to Run

### Normal execution

```powershell
python run_system.py --steps 200 --seed 1
```

### Full simulation

```powershell
python run_system.py --steps 3600 --seed 1
```

### Emergency demonstration

```powershell
python run_system.py --demo --steps 3600 --seed 1
```

`--demo` enables the emergency demonstration mode and uses the same current
multimodal SUMO configuration selected by the launcher. The equivalent
explicit emergency flag is also supported:

```powershell
python run_system.py --emergency --steps 3600 --seed 1
```

### SUMO-GUI execution

The launcher supports `--gui` and passes the optional delay to SUMO-GUI:

```powershell
python run_system.py --gui --steps 200 --seed 1
python run_system.py --demo --gui --steps 3600 --seed 1 --gui-delay 300
```

Other supported runtime options include `--config`, `--warmup`,
`--control-interval`, `--prediction-interval`, and `--output-dir`.

## Execution Flow

Each simulation step follows this closed loop:

```text
SUMO -> traffic state -> history/prediction -> emergency detection
     -> phase demand/scoring -> safety gate -> TraCI signal control -> SUMO
```

The controller discovers the signal topology through TraCI, estimates demand,
updates prediction and history, evaluates emergency state, selects a phase,
checks the decision through the safety gate, and applies the approved signal
state back to SUMO. Rejected decisions use the controller's fallback behavior.

## Multimodal Environment

The new environment is a 4x4 grid with 16 signalized junctions:
`J01` through `J16`. It includes:

- Cars and six ambulances
- Scheduled buses and bus stops
- Bicycles and dedicated bike lanes
- Pedestrians, sidewalks, crossings, and walking areas
- Trains and rail crossings
- Train platforms/stops

The generated scenario assets are under `simulation/networks/`,
`simulation/routes/`, and `simulation/additional/`. The multimodal scenario
definition and validation notes are in
`simulation/large_grid_multimodal/large_grid_multimodal/README.md`.

## Emergency Mode

When an ambulance is detected, the system identifies affected junctions along
the emergency route and applies emergency-priority signal decisions through
TraCI. Emergency decisions still pass through the controller's safety and
transition logic. This documentation describes the new `J01`-`J16`
multimodal environment; it does not use the old `A0`/`B0`/`C0` corridor as its
environment.

## Output and Artifacts

By default, each successful `run_system.py` execution creates a timestamped
directory under:

```text
results/final_runs/run_YYYYMMDD_HHMMSS/
```

Use `--output-dir` to choose a different location. Successful runs produce:

- `runtime.jsonl`: structured startup, decision, completion, and runtime-log
  events
- `metrics.json`: accumulated run metrics and controller decision counts
- `manifest.json`: configuration, mode, visualization, seed, step settings,
  and artifact paths

## Testing

Run the full regression suite:

```powershell
python -m pytest -q
```

The current full-suite result is **119 passed, 27 skipped, and 7 failed**.
The seven failures are pre-existing dataset availability failures caused by
missing accident-dataset Parquet files under `dataset/`, including the
expected `train-00000-of-00002.parquet` file. The project does not currently
have a zero-failure full test run.

## Troubleshooting

### `sumo` or `sumo-gui` is not found

Install Eclipse SUMO and either add its `bin` directory to `PATH` or set
`SUMO_HOME` to the SUMO installation directory. Re-run `sumo --version` and,
for GUI runs, `sumo-gui --version`.

### TraCI import failure

Activate the intended virtual environment and install the runtime packages:

```powershell
python -m pip install traci sumolib
python -c "import traci, sumolib; print('TraCI/SUMO OK')"
```

### TraCI port or connection problems

Close stale SUMO/SUMO-GUI processes and retry the command. Do not start a
second controller against the same active TraCI connection. For GUI runs,
confirm that `sumo-gui` is discoverable and reduce or increase `--gui-delay`
as needed.

### Missing dataset files

The SUMO runtime does not create the accident-model Parquet files. The
accident-related tests require the expected files in `dataset/`, such as the
train split files. Restore or prepare that dataset before running those tests;
the missing files explain the seven current full-suite failures.