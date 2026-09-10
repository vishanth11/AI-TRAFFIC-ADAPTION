# Ambulance SUMO Traffic Signal Verification

## 1. Executive Verdict

PARTIALLY IMPLEMENTED

The executable SUMO/TraCI path does identify emergency vehicles, derives their route movement, selects a compatible native SUMO phase, calls `trafficlight.setPhase`, reads the resulting signal state, and records emergency preemption. The canonical emergency runtime contains evidence of sequential preemption decisions at A0, B0, and C0 for `ambulance_0`.

The full requirement is not proven end to end for arbitrary routes. Priority is selected for only the first upcoming signalized movement at each control cycle, then reconsidered after the route advances. Hold/release is based on distance/ETA thresholds and vehicle presence, not on an explicit confirmation that the ambulance has crossed the junction. Tests do not start SUMO or assert the actual signal state for a moving ambulance.

## 2. Requirement

“Ambulance approaching a SUMO junction must receive GREEN for its movement.”

## 3. Ambulance Identification in SUMO

**Status: IMPLEMENTED.**

SUMO represents an ambulance with a vehicle type and emergency vehicle class:

- `simulation/routes/corridor_emergency.rou.xml:3` defines `<vType id="ambulance" vClass="emergency">`.
- The same route file defines `ambulance_0` through `ambulance_3` using that type at lines 26, 29, 65, and 68.
- `simulation/routes/large_grid_multimodal.rou.xml:9` defines the same ambulance type, with six ambulance vehicles at lines 111, 119, 136, 147, 152, and 155.

The executable identification logic is `EmergencyCorridorManager._is_emergency()` in `src/integration/emergency.py:77-85`. It checks `vehicle.getTypeID()` and `vehicle.getVehicleClass()` against emergency names and substrings. `discover()` obtains both values from TraCI and filters vehicles at `src/integration/emergency.py:123-131`.

Vehicle IDs are retained for logging and state tracking, but identification is not hard-coded to a particular ID. The visual `vehicle.setSignals()` call at `src/integration/emergency.py:87-107` sets vehicle light markers only; it is not traffic-light priority.

## 4. Ambulance Approach Detection

**Status: IMPLEMENTED, with limitations.**

`discover()` reads the live SUMO vehicle edge, lane, lane position, speed, route, and route index through TraCI at `src/integration/emergency.py:133-163`. It calculates distance to the end of the current lane and ETA at `src/integration/emergency.py:201-205`.

Junction and movement detection is route/topology based:

- The current route edge pair is matched to a discovered topology movement at `src/integration/emergency.py:165-185`.
- A live-edge/current-edge-to-next-edge fallback is used at `src/integration/emergency.py:187-199`.
- The resulting state stores `current_junction`, `next_junction`, `current_lane`, `current_edge`, `next_edge`, route, and destination at `src/integration/emergency.py:218-234`.
- The more important route-ahead planner scans the ordered route and finds the first signalized edge pair ahead in `src/integration/sumo_decision_controller.py:802-850`.

The `next_junction` field in `discover()` is inferred by finding any topology movement whose incoming road equals the current movement's outgoing road (`src/integration/emergency.py:207-216`). The executable preemption decision instead uses the ordered route scan in `_build_route_ahead_plan()`.

## 5. SUMO Traffic Signal Control

**Status: IMPLEMENTED.**

Signalized junctions and controlled movements are discovered dynamically:

- `SUMOTopologyAdapter.discover_junction()` calls `trafficlight.getControlledLinks()` and converts incoming/outgoing SUMO lanes into movement IDs (`src/integration/sumo_topology_adapter.py:25-31`, `src/integration/sumo_topology_adapter.py:44-118`).
- `discover_signalized_junctions()` calls `trafficlight.getIDList()`, `getControlledLinks()`, and `getAllProgramLogics()` to select usable multi-phase signal controllers (`src/integration/sumo_topology_adapter.py:192-220`).
- `SUMOPhaseAdapter` obtains current phase/program data with `getPhase()`, `getProgram()`, and `getAllProgramLogics()` (`src/integration/sumo_phase_adapter.py:15-67`).
- Native SUMO green phase state strings are converted into movement IDs by `phase_to_movements()` and `build_phase_representation()` (`src/integration/sumo_phase_adapter.py:276-367`).

The only traffic-light write found in the executable controller is `self.traci.trafficlight.setPhase(junction_id, transition)` in `_execute()` (`src/integration/sumo_decision_controller.py:610-690`, specifically line 659). No `setRedYellowGreenState()` call was found. The implementation changes the active native phase rather than constructing a custom signal-state string.

## 6. Ambulance → GREEN Signal Execution

**Status: IMPLEMENTED for the canonical path; end-to-end passage is not fully proven.**

Exact executable path:

1. `Phase2SUMOController.run()` calls `emergency_manager.discover()` each simulation iteration (`src/integration/sumo_decision_controller.py:1298-1315`).
2. `_build_upcoming_emergency_plans()` calls `_build_route_ahead_plan()` for each detected emergency state (`src/integration/sumo_decision_controller.py:930-987`).
3. `_build_route_ahead_plan()` matches each ordered route edge pair to a topology movement, chooses the first native phase containing that movement, and records conflicting movements (`src/integration/sumo_decision_controller.py:852-929`).
4. `plan_for_junction()` uses the same movement-to-compatible-phase logic for current-junction planning (`src/integration/emergency.py:293-338`). It selects a phase only when the ambulance movement is present and the phase passes `generator.is_compatible()`.
5. `_emergency_requires_preemption()` activates priority when ETA is at most 45 seconds or route distance is at most 500 meters (`src/integration/sumo_decision_controller.py:1000-1042`).
6. The per-junction loop selects `emergency_plan.selected_phase`, resolves its SUMO phase index, and calls `_execute()` (`src/integration/sumo_decision_controller.py:1425-1490`).
7. `_execute()` calls `trafficlight.setPhase()`, steps SUMO through the transition, and reads back the executed phase/state (`src/integration/sumo_decision_controller.py:617-690`).
8. The decision record stores `method: emergency_preemption`, the emergency vehicle ID, selected phase, executed phase, and executed signal state (`src/integration/sumo_decision_controller.py:1517-1538`).

This is a real signal-control path, not merely emergency detection. Existing runtime evidence includes:

- `results/final_runs/run_20260906_182533/runtime.jsonl:2` records `ambulance_0`, A0, emergency preemption, phase 2, and state `rrrrGGggrrrrGGgg`.
- The same runtime records emergency preemption for B0 and C0 at lines 5 and 10, with the same executed green-phase state.
- `results/final_runs/run_20260906_182533/metrics.json:14-17` reports three emergency priority decisions and junctions A0, B0, and C0.

The source proves that the selected native phase contains the ambulance movement. The runtime log proves a phase/state was executed, but does not independently log the movement ID or the ambulance crossing the stop line.

## 7. Multi-Junction / All-Junction Behavior

**Does the ambulance receive GREEN at every upcoming junction on its route?**

PARTIALLY

The implementation can handle sequential junctions:

- `_build_route_ahead_plan()` scans the ordered route but deliberately stops at the **first** signalized movement (`src/integration/sumo_decision_controller.py:802-850`).
- After the vehicle advances its SUMO route index, the next control cycle scans again and can select the next junction.
- `corridor_emergency.rou.xml:26` gives `ambulance_0` the route `A0B0 -> B0C0 -> C0top2`, which contains three signalized junction movements.
- The recorded emergency run shows emergency-preemption decisions at A0, B0, and C0 for `ambulance_0` (`results/final_runs/run_20260906_182533/runtime.jsonl:2`, `:5`, and `:10`).
- The large-grid route file includes routes with multiple successive junction movements, including `amb_5_long_snake` (`simulation/routes/large_grid_multimodal.rou.xml:5`) and other multi-junction routes at lines 1-6.

This is sequential re-planning, not a route-wide reservation or a proof that every future junction is preempted before the ambulance reaches it. There is no test that parametrizes routes with two, three, four, or more junctions and asserts a green at each one. Therefore the all-route requirement is not fully verified.

## 8. Direction/Maneuver Handling

**Status: IMPLEMENTED, topology/route derived.**

The implementation is not hard-coded to North/South/East/West labels. It derives the movement from the ambulance route's current edge and next edge, then matches that pair to `movement.from_road` and `movement.to_road` (`src/integration/emergency.py:249-291`). The route-ahead planner performs the same edge-pair matching over the ordered route (`src/integration/sumo_decision_controller.py:852-872`).

The selected phase must contain the movement ID (`src/integration/sumo_decision_controller.py:885-897` and `src/integration/emergency.py:305-315`). Since the movement is built from SUMO controlled links, the phase corresponds to the ambulance's actual incoming-to-outgoing maneuver, subject to the correctness of the discovered SUMO link mapping.

## 9. Conflict Clearance

**Status: IMPLEMENTED through native phase compatibility and transition; not independently verified with an end-to-end conflict test.**

The emergency plan records movements conflicting with the ambulance movement using the discovered conflict graph (`src/integration/emergency.py:317-322`, `src/integration/sumo_decision_controller.py:913-924`). Candidate phases are accepted only when `generator.is_compatible()` returns true (`src/integration/emergency.py:305-315`).

When changing phases, `_safe_transition_index()` chooses the preceding yellow/red phase where available (`src/integration/sumo_decision_controller.py:204-229`). `_execute()` calls `setPhase()` on that transition, advances SUMO, and waits for the target phase (`src/integration/sumo_decision_controller.py:636-680`). The resulting native phase state is read back.

There is no explicit `setRedYellowGreenState()` or explicit all-red command. Conflict clearance therefore relies on SUMO's native program phases and the preceding transition phase. The runtime state `rrrrGGggrrrrGGgg` demonstrates red characters for non-selected links, but the repository does not contain an assertion checking every conflicting link during an ambulance event.

## 10. Signal Restoration

**Status: PARTIALLY IMPLEMENTED.**

While an emergency plan continues to satisfy the ETA/distance trigger, the controller stores `emergency_hold_phase` and `emergency_vehicle_id`, and keeps the selected phase (`src/integration/sumo_decision_controller.py:1542-1578`). When the emergency preemption condition is no longer true, the per-junction hold is cleared and normal AI control resumes (`src/integration/sumo_decision_controller.py:1581-1610`).

At the global vehicle level, `release_completed()` compares the current emergency vehicle IDs with the previous active set and returns vehicles that disappeared from SUMO (`src/integration/emergency.py:366-370`). The run loop logs an explicit release and normal control resumption at `src/integration/sumo_decision_controller.py:1315-1330`.

The limitation is causal: the junction hold is released when the preemption trigger is false, not when a stop-line/crossing event proves the ambulance passed. A stopped or delayed ambulance with distance above 500 m and no valid ETA can lose priority. Conversely, a route transition can move planning to the next junction without a direct passage confirmation for the previous one.

## 11. Testing Evidence

**Status: IMPLEMENTED BUT NOT FULLY TESTED.**

Focused tests pass: `python -m pytest test_phase2_3_emergency.py test_phase2_4_final.py -q` produced `7 passed`.

What those tests prove:

- Synthetic emergency urgency and zero-speed handling (`test_phase2_3_emergency.py:21-38`).
- Topology-derived movement and candidate phase selection (`test_phase2_3_emergency.py:40-60`).
- Conflicting phase rejection (`test_phase2_3_emergency.py:63-76`).
- Multiple emergency states choose the most urgent plan for one junction (`test_phase2_3_emergency.py:79-105`).
- Release detection when a vehicle is absent from the active state (`test_phase2_3_emergency.py:108-113`).
- Runtime/config path existence only (`test_phase2_4_final.py:9-24`).

The repository also contains SUMO inspection scripts (`test_sumo_adapter.py` and `test_sumo_topology.py`), but they print topology/phase information and do not assert ambulance-to-green behavior. No test was found that starts the canonical emergency controller and asserts, for a moving ambulance, the selected traffic-light state, conflict clearance, passage, restoration, or all upcoming route junctions.

The existing runtime artifacts are useful operational evidence, not a complete proof: `runtime.jsonl` records executed phase/state and emergency metadata, while `metrics.json` counts emergency priority decisions. No artifact found records ambulance position crossing each stop line or links the logged state to a specific movement ID.

## 12. Implemented vs Missing

| Requirement | Status | Evidence |
|---|---|---|
| Ambulance exists in SUMO | IMPLEMENTED | Ambulance vType and vehicles in `simulation/routes/corridor_emergency.rou.xml:3,26,29,65,68` and `simulation/routes/large_grid_multimodal.rou.xml:9,111-155`. |
| Ambulance detection | IMPLEMENTED | Type/class TraCI checks in `src/integration/emergency.py:77-85,123-131`. |
| Ambulance approaching junction detection | IMPLEMENTED | Live edge/lane/position/speed plus route-ahead planning in `src/integration/emergency.py:133-163` and `src/integration/sumo_decision_controller.py:802-850`. |
| Junction identification | IMPLEMENTED | Controlled-link topology discovery and route-pair matching in `src/integration/sumo_topology_adapter.py:25-118` and `src/integration/emergency.py:165-199`. |
| Ambulance direction identification | IMPLEMENTED | Route edge-pair to movement mapping in `src/integration/emergency.py:249-291`. |
| SUMO signal control | IMPLEMENTED | Native phase discovery/readback and `trafficlight.setPhase()` in `src/integration/sumo_phase_adapter.py:15-67` and `src/integration/sumo_decision_controller.py:610-690`. |
| Ambulance direction gets GREEN | IMPLEMENTED in executable path | Candidate phase must contain the derived movement in `src/integration/emergency.py:305-315`; runtime A0/B0/C0 records show executed green states. |
| Conflicting traffic cleared | PARTIALLY | Compatible native phases and transition phase are used, but no end-to-end assertion checks all conflicting links. |
| GREEN maintained until ambulance passes | PARTIALLY | Hold exists in `src/integration/sumo_decision_controller.py:1542-1578`, but release uses ETA/distance trigger state, not passage detection. |
| Signal returns to normal | PARTIALLY | Hold release and normal AI resume are implemented at `src/integration/sumo_decision_controller.py:1581-1610`; passage-causal restoration is not tested. |
| Next junction handled | IMPLEMENTED in sequential path | Route is rescanned after route-index progression; runtime shows A0 then B0 then C0 for `ambulance_0`. |
| Multiple upcoming junctions handled | PARTIALLY | First upcoming signalized movement only per plan (`src/integration/sumo_decision_controller.py:802-850`); later junctions depend on subsequent cycles and route progress. |
| All route junctions handled | PARTIALLY | No route-wide invariant or test proves every junction on arbitrary long routes receives priority. |
| Multiple ambulances | PARTIALLY | All emergency IDs are discovered; one most-urgent plan is retained per junction in `src/integration/sumo_decision_controller.py:944-987`. No simultaneous conflicting-ambulance arbitration proof exists. |
| End-to-end SUMO test | NOT IMPLEMENTED | Focused tests are synthetic; SUMO scripts print topology; no test asserts moving ambulance -> TraCI green -> passage -> restoration. |

## 13. Exact File / Function / Line References

- `src/integration/emergency.py:77-85` - emergency type/class identification.
- `src/integration/emergency.py:115-234` - SUMO vehicle discovery and state construction.
- `src/integration/emergency.py:249-338` - route movement resolution and compatible phase selection.
- `src/integration/emergency.py:366-370` - release detection when emergency vehicles leave SUMO.
- `src/integration/sumo_topology_adapter.py:25-118` - controlled-link to movement mapping.
- `src/integration/sumo_topology_adapter.py:192-220` - signalized-junction discovery.
- `src/integration/sumo_phase_adapter.py:15-67` - SUMO program and phase reads.
- `src/integration/sumo_phase_adapter.py:276-367` - green link to movement representation.
- `src/integration/sumo_decision_controller.py:204-229` - safe native transition selection.
- `src/integration/sumo_decision_controller.py:610-690` - TraCI phase execution, stepping, and readback.
- `src/integration/sumo_decision_controller.py:802-929` - first upcoming route movement and phase plan.
- `src/integration/sumo_decision_controller.py:930-987` - per-junction emergency plan arbitration.
- `src/integration/sumo_decision_controller.py:1000-1042` - ETA/distance preemption trigger.
- `src/integration/sumo_decision_controller.py:1298-1330` - emergency discovery and vehicle-release logging.
- `src/integration/sumo_decision_controller.py:1425-1538` - emergency phase selection, execution, and decision logging.
- `src/integration/sumo_decision_controller.py:1542-1610` - emergency hold and normal-control release.
- `simulation/routes/corridor_emergency.rou.xml:3,26-29,65-68` - ambulance type and multi-junction emergency routes.
- `simulation/routes/large_grid_multimodal.rou.xml:1-9,111-155` - multi-junction ambulance routes and vehicles.
- `test_phase2_3_emergency.py:21-113` - synthetic emergency planning/release tests.
- `test_phase2_4_final.py:9-24` - runtime/config existence tests.
- `results/final_runs/run_20260906_182533/runtime.jsonl:2,5,10` - recorded A0/B0/C0 emergency-preemption executions.
- `results/final_runs/run_20260906_182533/metrics.json:14-17` - three emergency priority decisions at A0, B0, and C0.
