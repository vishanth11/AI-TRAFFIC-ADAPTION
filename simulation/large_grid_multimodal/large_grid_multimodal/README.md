# `large_grid_multimodal` SUMO scenario

This scenario extends the validated 4×4 `large_grid` road network into a
realistic microscopic multimodal traffic environment while keeping the
original `large_grid`, `corridor` and `four_way` scenarios untouched.

## What is included

| Mode | Implementation |
|------|----------------|
| Cars + ambulances | Existing vTypes; reduced volumes to coexist with PT/bikes/pedestrians |
| Buses | 4 scheduled lines with 6 departures each; 12 bus stops with 20 s dwell |
| Bicycles | Dedicated bike lanes; 6 bicycle flows across the grid |
| Pedestrians | Sidewalks, crossings, walkingareas; person flows walking and using `modes="public"` |
| Trains | One north-south rail line with 3 stations; 6 trains per direction with 40 s dwell |
| Rail crossings | 8 automatic `rail_crossing` junctions where roads cross the rail line |

## File layout

```text
simulation/
  configs/
    large_grid_multimodal.sumocfg      # headless simulation config
    large_grid_multimodal_gui.sumocfg  # GUI config with delay/start
  networks/
    large_grid_multimodal.net.xml      # generated multimodal network
  routes/
    large_grid_multimodal.rou.xml      # multimodal demand
  additional/
    large_grid_multimodal_stops.add.xml # 12 bus + 6 train stops
  large_grid_multimodal/
    build_network.py                   # netconvert + stops generator
    validate_multimodal.py             # end-to-end validation script
    README.md                          # this file
src/
  integration/
    sumo_multimodal_state_adapter.py   # additive, backward-compatible adapter
```

## Build / regenerate

```bash
python simulation/large_grid_multimodal/build_network.py
```

This rebuilds `networks/large_grid_multimodal.net.xml` and
`additional/large_grid_multimodal_stops.add.xml` from the source
`nod.xml`/`edg.xml` files.

## Run a smoke test

```bash
sumo -c simulation/configs/large_grid_multimodal.sumocfg --no-step-log -e 3600
```

## Run with GUI

```bash
sumo-gui -c simulation/configs/large_grid_multimodal_gui.sumocfg
```

## Validate end-to-end integration

```bash
python simulation/large_grid_multimodal/validate_multimodal.py \
    --duration 3600 --output multimodal_validation_report.json
```

The validator exercises:

* SUMO → TraCI connection
* `SUMOTopologyAdapter` discovering the 16 signalized junctions
* `SUMOTrafficStateBuilder.build_state()` returning the unchanged `(36,)` vector
* `SUMOMultimodalStateAdapter` collecting mode counts
* A lightweight AI/controller making traffic-light decisions for every junction
* Bus/train stop service and pedestrian/public-transport interactions

It emits a JSON report with observed mode counts, stops served, TraCI
warnings/teleports and AI/controller decision counts.

## Run regression tests

```bash
python -m pytest tests/test_large_grid_multimodal_sumo.py -v
```

## Design notes

* **Backward compatibility**: the existing `SUMOTrafficStateBuilder.build_state()`
  and its 36-element state are unchanged.  Multimodal information is provided
  by the new additive `SUMOMultimodalStateAdapter`.
* **No scenario deletion**: `large_grid`, `corridor` and `four_way` files are
  left untouched.
* **Demand tuning**: car/bike/pedestrian volumes were reduced relative to
  `large_grid` so the network remains free of excessive teleports while
  still showing simultaneous multimodal interaction.
