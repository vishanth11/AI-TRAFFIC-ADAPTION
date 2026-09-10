# Ambulance Green Corridor — Implementation Report

## Architecture

The ambulance green corridor feature enables proactive, route-wide emergency signal preemption. When an ambulance is detected, the system analyzes its **complete SUMO route**, identifies **all upcoming signalized junctions**, and creates a corridor plan that prepares green phases at each junction before the ambulance arrives.

### Key Components

| Component | File | Role |
|-----------|------|------|
| `EmergencyCorridorManager` | `src/integration/emergency.py` | Discovers emergency vehicles, builds corridor plans, manages lifecycle |
| `Phase2SUMOController` | `src/integration/sumo_decision_controller.py` | Main loop — corridor-aware preemption, activation, release |
| `AmbulanceCorridorState` | `src/integration/emergency.py` | Per-ambulance corridor tracking |
| `CorridorJunctionState` | `src/integration/emergency.py` | Per-junction activation/passage/release state |

---

## Implementation Flow

```
Ambulance Detection (TraCI type/class check)
        ↓
Current Position + Route (TraCI getRoute, getRouteIndex)
        ↓
Find ALL Upcoming Signalized Junctions
  (walk route edges, match to topology movements)
        ↓
Determine Movement + ETA for Each Junction
  (route distance calculation + speed)
        ↓
Build Green Corridor (AmbulanceCorridorState)
  - Per-junction: movement, phase, ETA, distance
  - All junctions start in "prepared" state
        ↓
ETA-Based Activation Gating
  - Only activate when ETA ≤ 45s or distance ≤ 500m
  - Far-ahead junctions stay "prepared"
        ↓
Downstream Congestion Check
  - Query traci.edge.getLastStepOccupancy()
  - Defer activation if downstream > 80% occupancy
        ↓
Schedule/Activate Native SUMO Green Phases
  - Use existing _execute() + _safe_transition_index()
  - Yellow/all-red transitions preserved
        ↓
Monitor Ambulance Progress
  - Route-index-based passage detection
  - update_corridor_progression() each step
        ↓
Release Passed Junctions
  - Junction released when route_index > movement_route_index
  - emergency_hold_phase cleared
  - Normal adaptive control resumes
        ↓
Corridor Completion
  - All junctions passed → corridor marked inactive
  - Cleaned from active corridors dict
```

---

## Detection Distance

- **ETA threshold:** 45 seconds (`EMERGENCY_PREEMPT_ETA_SECONDS`)
- **Distance threshold:** 500 meters (`EMERGENCY_PREEMPT_DISTANCE_METERS`)
- **Route distance:** Calculated by walking SUMO route edges from current position to each junction's approach edge

These thresholds control **activation**, not **planning**. The full corridor is planned immediately upon ambulance detection, but junctions are only activated (signal changed) when the ambulance is within the activation window.

---

## Route-Wide Planning

The corridor plan walks the ambulance's complete SUMO route:

```
for movement_route_index in range(route_index, len(route) - 1):
    from_edge = route[movement_route_index]
    to_edge = route[movement_route_index + 1]
    → match to topology movements
    → find compatible green phase
    → add to corridor plan
```

This supports **arbitrary numbers of junctions** — 2, 3, 4, 5+ — without code changes.

---

## Junction Selection

Junctions are selected dynamically from SUMO topology:

1. For each consecutive edge pair `(from_edge, to_edge)` on the route
2. Search all signalized junction topologies for a movement matching this pair
3. If found, find a native SUMO green phase containing that movement
4. Verify phase compatibility using the conflict graph

No junction IDs are hardcoded. Works for any SUMO network.

---

## Phase Selection

1. Identify the ambulance's **required movement** (from_edge → to_edge)
2. Find all SUMO signal phases containing that movement
3. Filter by conflict-graph compatibility
4. Select the first compatible phase (SUMO's native ordering)
5. Execute via `_safe_transition_index()` → yellow → green

---

## ETA Calculation

```
distance = remaining_current_lane + sum(edge_lengths[route_index+1 : movement_route_index+1])
ETA = distance / ambulance_speed  (if speed ≥ 0.5 m/s)
```

Distance is calculated per-junction by walking forward through the route.

---

## Congestion Handling

### Downstream Congestion Check
Before activating a junction, the controller queries:
```python
occupancy = traci.edge.getLastStepOccupancy(downstream_edge)
```
If occupancy ≥ 0.8 (80%), activation is deferred to prevent trapping the ambulance.

### Traffic Disruption Minimization
- Only the ambulance's required movement gets priority
- Conflicting phases are terminated safely (via yellow transition)
- Non-conflicting traffic continues normally
- Each junction is held only while the ambulance approaches/passes
- Released immediately after passage → normal adaptive AI resumes

---

## Corridor State

```python
AmbulanceCorridorState:
    ambulance_id: str
    route: tuple[str, ...]
    current_route_index: int
    junctions: list[CorridorJunctionState]
    active: bool
    route_generation: int

CorridorJunctionState:
    plan: EmergencyPlan
    status: str       # "prepared" | "activated" | "released"
    activated: bool
    passed: bool
    released: bool
```

### Junction Lifecycle

```
prepared → activated → passed → released
                       ↑
                 (ambulance crosses junction)
```

---

## Release Logic

Junction passage is detected by comparing:
- `ambulance.route_index > junction.movement_route_index`
- Fallback: ambulance's current_edge is at or past the junction's outgoing edge

This is more reliable than distance-based release because:
- Distance can fluctuate during junction traversal
- Route index monotonically increases as the vehicle progresses

---

## Tests

### Unit Tests (13 tests, no SUMO required)

| Test | Description | Status |
|------|-------------|--------|
| 1 | Single junction — correct movement + phase | ✅ |
| 2 | 3-junction route — all in corridor | ✅ |
| 3 | 5+ junction route — all planned | ✅ |
| 4 | Distant ambulance — downstream included | ✅ |
| 5 | J1 passed → J1 released, J2 remains | ✅ |
| 6 | Route completed → all released | ✅ |
| 7 | Conflicting phase excluded | ✅ |
| 8 | Emergency hold not permanent | ✅ |
| 9 | Ambulance disappears → safe release | ✅ |
| 10 | Route changes → corridor rebuilt | ✅ |
| 11 | Multiple ambulances → deterministic | ✅ |
| 12 | ETA-based activation gating | ✅ |
| 13 | Route-index passage detection | ✅ |

### Integration Test (SUMO required)

- Uses `corridor_emergency.sumocfg` with 3-junction corridor (A0, B0, C0)
- Verifies ambulance_3 traversing C0 → B0 → A0
- Checks corridor creation, activation, passage, and release

---

## Limitations

1. **Congestion check is edge-level only** — uses `getLastStepOccupancy()` which is a coarse measure. Does not account for individual lane congestion.

2. **Route changes** — if SUMO reroutes the ambulance, the corridor is rebuilt from scratch. In-progress activations at junctions no longer on the route are released.

3. **Signal timing optimization** — the system selects the first compatible phase. It does not optimize for minimal disruption among multiple compatible phases.

4. **Coordinated green wave timing** — junctions are activated independently based on ETA. There is no coordinated offset timing to create a perfect green wave. The ETA-based activation approximates this.

5. **Pedestrian detection** — pedestrian crossings are assumed clear during emergency preemption. SUMO's pedestrian model is not queried.
