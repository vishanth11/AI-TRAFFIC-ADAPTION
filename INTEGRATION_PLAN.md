# Integration Plan: large_grid_multimodal as Default SUMO Environment

**Status:** PLAN ONLY — No implementation, deletion, or renaming yet.

**Scope:** Make `large_grid_multimodal` the main default SUMO environment, replacing the current `corridor` default.

---

## 1. Files That Must Change

### 1.1 Core Configuration Files

#### A. `src/integration/sumo_decision_controller.py` (MUST CHANGE)
- **Line 41:** Change `SUMO_CONFIG` constant
  ```python
  CURRENT:  SUMO_CONFIG = os.path.join(PROJECT_ROOT, "simulation", "configs", "corridor.sumocfg")
  NEW:      SUMO_CONFIG = os.path.join(PROJECT_ROOT, "simulation", "configs", "large_grid_multimodal.sumocfg")
  ```
- **Line 42-44:** Change `EMERGENCY_SUMO_CONFIG` constant (NEW emergency variant)
  ```python
  CURRENT:  EMERGENCY_SUMO_CONFIG = os.path.join(PROJECT_ROOT, "simulation", "configs", "corridor_emergency.sumocfg")
  NEW:      EMERGENCY_SUMO_CONFIG = os.path.join(PROJECT_ROOT, "simulation", "configs", "large_grid_multimodal_emergency.sumocfg")
  ```
- **Rationale:** This is the runtime initialization point; all SUMO commands chain through these constants.

#### B. `run_system.py` (MUST CHANGE)
- **Line 46:** Import updated constants (no code change needed, already imports from sumo_decision_controller)
- **Lines 240-250:** Default config resolution already delegates to `SUMO_CONFIG` and `EMERGENCY_SUMO_CONFIG` from sumo_decision_controller
  - No changes needed here IF sumo_decision_controller constants are updated
  - Validation: `--config` CLI arg still overrides defaults ✓

### 1.2 Integration Module Files (NO CHANGES — Dynamic Discovery)

#### A. `src/integration/sumo_topology_adapter.py`
- **Status:** NO CHANGE
- **Reason:** `discover_signalized_junctions()` method dynamically discovers all traffic-controlled junctions at runtime
  - Will automatically adapt to large_grid_multimodal's 16 junctions
  - Does NOT use hardcoded `TARGET_JUNCTIONS` list (that's only in `sumo_prediction_controller.py`, which is NOT the canonical runtime)
  
#### B. `src/integration/sumo_phase_adapter.py`
- **Status:** NO CHANGE
- **Reason:** Works with discovered junctions; phase representation is built dynamically per junction

#### C. `src/integration/sumo_conflict_adapter.py`
- **Status:** NO CHANGE
- **Reason:** Derives conflict graph from SUMO phase definitions; works with any junction topology

#### D. `src/integration/sumo_demand.py`
- **Status:** NO CHANGE
- **Reason:** Demand is calculated per junction; scales with discovered junctions

#### E. `src/integration/emergency.py`
- **Status:** NO CHANGE
- **Reason:** Topology-adaptive; works with any set of discovered junctions and edges

#### F. `src/integration/sumo_prediction_controller.py`
- **Status:** INFORMATIONAL ONLY (not used by canonical runtime)
- **Has hardcoded `TARGET_JUNCTIONS = ["A0", "B0", "C0"]`** — this is a LEGACY/DEMO file, not the production code path
  - This file exists for research/testing but is NOT invoked by the canonical `Phase2SUMOController`
  - For documentation: this file would need TARGET_JUNCTIONS updated if ever resurrected, but is not critical for main integration

### 1.3 State & Prediction Pipeline (NO CHANGES — Adaptive)

#### A. `src/integration/sumo_prediction_controller.py` — SUMOTrafficStateBuilder class
- **Status:** NO CHANGE (dynamic)
- **Current:** `NUM_SENSORS = 36` and `ZONES_PER_ROAD = 3`
  - These define the FIXED observation space (12 incoming roads × 3 zones = 36 sensors)
  - The sensor mapping is computed dynamically from discovered junctions
  - The builder is instantiated with `num_sensors=NUM_SENSORS, target_junctions=list(runtimes)`
  - `list(runtimes)` = the discovered junctions, so it scales automatically

#### B. `src/integration/traffic_history.py`
- **Status:** NO CHANGE
- **Reason:** History buffer is fixed at 36 sensors; this is the CANONICAL observation space dimension
- **Important:** Even with 16 junctions in large_grid_multimodal, the 36-D representation is preserved by the sensor builder's dynamic zone allocation

#### C. Prediction Pipeline Integration (in `sumo_decision_controller.py`)
- **Line 299-300:**
  ```python
  sensor_builder = SUMOTrafficStateBuilder(
      num_sensors=NUM_SENSORS,
      target_junctions=list(runtimes),  # Dynamically populated from discovered junctions
  )
  ```
- **Status:** NO CHANGE
- **Reason:** Uses discovered junctions; scales automatically

### 1.4 Decision Engine & Control Flow (NO CHANGES)

#### A. `src/integration/decision_engine.py`
- **Status:** NO CHANGE
- **Reason:** Makes decisions per junction; works with any number of junctions

#### B. `src/integration/safety_gate.py`
- **Status:** NO CHANGE
- **Reason:** Risk assessment per junction; topology-agnostic

#### C. `src/integration/fallback_controller.py`
- **Status:** NO CHANGE
- **Reason:** Fallback logic per junction; scales with discovered junctions

---

## 2. Constants/Config References That Must Change

### Summary Table

| File | Constant | Current Value | New Value | Reason |
|------|----------|---------------|-----------|--------|
| `src/integration/sumo_decision_controller.py` | `SUMO_CONFIG` | `simulation/configs/corridor.sumocfg` | `simulation/configs/large_grid_multimodal.sumocfg` | Main scenario default |
| `src/integration/sumo_decision_controller.py` | `EMERGENCY_SUMO_CONFIG` | `simulation/configs/corridor_emergency.sumocfg` | `simulation/configs/large_grid_multimodal_emergency.sumocfg` | Emergency scenario (NEW file required) |
| `src/integration/sumo_prediction_controller.py` | `TARGET_JUNCTIONS` (legacy) | `["A0", "B0", "C0"]` | LEAVE AS-IS or update to large_grid junctions if file is ever used | NOT USED by canonical runtime |
| `src/integration/sumo_prediction_controller.py` | `SUMO_CONFIG` (in file) | `corridor.sumocfg` | LEAVE AS-IS or REMOVE (duplicate) | Only used if this file is directly executed (not production) |

### Non-Code Constants (Configuration)

**These are NOT Python constants but must be set up correctly:**

1. **Simulation Duration:** Currently `TOTAL_STEPS = 200` in `sumo_decision_controller.py`
   - Likely acceptable for large_grid_multimodal
   - May need tuning based on network size/congestion

2. **Control Interval:** Currently `CONTROL_INTERVAL = 10.0` seconds
   - Likely acceptable for all network sizes
   - Adaptive per junction, so scales automatically

3. **Prediction Interval:** Currently `PREDICTION_INTERVAL = 10` steps
   - Likely acceptable; works with sensor builder regardless of topology

4. **Warmup Steps:** Currently `WARMUP_STEPS = 10`
   - May want to increase for larger network (more vehicles to load)
   - Optional tuning

---

## 3. How run_system.py Will Use the New Environment

### Execution Flow (No Changes Needed)

```
run_system.py
  ↓
  Imports Phase2SUMOController from sumo_decision_controller.py
  ↓
  Reads SUMO_CONFIG and EMERGENCY_SUMO_CONFIG (now large_grid_multimodal variants)
  ↓
  CLI args override:
    --config PATH         : explicit config file
    --emergency           : use EMERGENCY_SUMO_CONFIG (now large_grid_multimodal_emergency.sumocfg)
    --demo                : enable emergency mode + use EMERGENCY_SUMO_CONFIG
    --gui                 : launch SUMO-GUI with the selected config
  ↓
  Creates sumo_command with updated config
  ↓
  Phase2SUMOController.run() starts TraCI connection
  ↓
  Discovers all signalized junctions dynamically (will find 16 junctions)
  ↓
  Builds runtimes for each junction
  ↓
  Runs adaptive control loop
  ↓
  Outputs metrics to results/final_runs/run_YYYYMMDD_HHMMSS/
```

### CLI Examples (After Integration)

```powershell
# Normal operation (uses large_grid_multimodal.sumocfg)
python run_system.py --steps 200 --seed 1

# Emergency demo (uses large_grid_multimodal_emergency.sumocfg)
python run_system.py --demo --steps 200 --seed 1

# With GUI visualization
python run_system.py --gui --steps 200 --seed 1

# Explicit config override (backward compatibility)
python run_system.py --config simulation/configs/corridor.sumocfg --steps 200
```

---

## 4. Emergency Mode with the New Multimodal Environment

### Current Architecture
- **Normal mode:** Uses `large_grid_multimodal.sumocfg` + `large_grid_multimodal.rou.xml`
- **Emergency mode:** Uses `large_grid_multimodal_emergency.sumocfg` + `large_grid_multimodal_emergency.rou.xml`

### What Must Happen
1. **Create `large_grid_multimodal_emergency.sumocfg`**
   - References: `../networks/large_grid_multimodal.net.xml` (SAME network)
   - References: `../routes/large_grid_multimodal_emergency.rou.xml` (NEW emergency routes)
   - Time: Adjust `<end>` value if needed for demonstration

2. **Create `large_grid_multimodal_emergency.rou.xml`**
   - Must include at least one ambulance/emergency vehicle on a specific route
   - Route must pass through at least 2 junctions (to demonstrate preemption)
   - Recommend: deterministic route + seed for reproducibility

### Emergency Flow (No Code Changes)
```
Phase2SUMOController.__init__(emergency_enabled=True)
  ↓
  phase_adapter runs()
  ↓
  command = [sumo_exe, "-c", EMERGENCY_SUMO_CONFIG]  # Now large_grid_multimodal_emergency.sumocfg
  ↓
  traci.start(command) → loads network + emergency routes
  ↓
  EmergencyCorridorManager discovers ambulance on network
  ↓
  For each discovered junction:
    - compute_eta_to_junction()
    - if eta < EMERGENCY_PREEMPT_ETA_SECONDS (30s) or distance < EMERGENCY_PREEMPT_DISTANCE_METERS (350m):
      → trigger emergency green
  ↓
  Emergency logic remains identical; only the network/routes change
```

---

## 5. 36-D Traffic State/Prediction Pipeline Compatibility

### The 36-D Representation is Independent of Network Size

**Current Design:**
- 36 sensors = fixed observation space dimension
- Derived from: 12 incoming roads × 3 observation zones (FAR/MIDDLE/NEAR)
- These 12 roads are discovered dynamically per junction

**With large_grid_multimodal (16 junctions):**
- Each junction independently has incoming roads
- SUMOTrafficStateBuilder discovers roads for EACH junction
- Total observations = sum of (roads × 3) across all discovered junctions, packed into 36-D space

**Example:**
```
Corridor (3 junctions):
  A0: 4 incoming roads × 3 zones = 12 sensors
  B0: 4 incoming roads × 3 zones = 12 sensors
  C0: 4 incoming roads × 3 zones = 12 sensors
  TOTAL = 36 sensors ✓

large_grid_multimodal (16 junctions, hypothetical):
  If each junction has ~2.25 incoming roads on average:
    16 × 2.25 × 3 = 108 potential sensors
  
  PACKING STRATEGY: Map these into 36-D space via:
    - Zone indices from SUMOTrafficStateBuilder
    - Weighted aggregation or sparse coding
    - OR: Select subset of most-active junctions
    - OR: Re-allocate zones per junction to fit 36 total
```

**Critical Question:** How many junctions will large_grid_multimodal actually have, and how many incoming roads per junction?

- **If ≤ 12 total roads across all junctions:** Fits naturally into 36-D (12 roads × 3 zones)
- **If > 12 roads:** SUMOTrafficStateBuilder must handle overflow. Current implementation likely:
  - Fills available slots (sensor_id < NUM_SENSORS)
  - Pads with zeros if fewer than 36 total
  - **Risk:** If more than 12 roads, some may not map to sensors

### What Remains Unchanged

1. **TrafficHistory buffer:** Still 10 × 36
2. **Prediction model input:** Still 10 × 36 (history length × sensors)
3. **GNN model weights:** Still expecting 36-D input
4. **Sensor builder API:** Still `num_sensors=36`

### Action: Validate Sensor Mapping

**Before switching to large_grid_multimodal, must verify:**
```
sensor_builder = SUMOTrafficStateBuilder(
    num_sensors=36,
    target_junctions=discovered_junctions  # Will be 16 junctions
)

# At runtime, confirm:
history.get_padded_history().shape == (10, 36)  # Must be exact
```

If large_grid_multimodal has too many roads, will need to:
- Redefine NUM_SENSORS (update everywhere)
- OR adjust sensor zones (FAR/MIDDLE/NEAR boundaries)
- OR select subset of junctions for observation

---

## 6. The 16 Dynamically Discovered Junctions

### Discovery Mechanism (No Code Changes)

**In `Phase2SUMOController._build_runtime()` (line ~160):**
```python
signalized_ids = topology_adapter.discover_signalized_junctions()  # Returns ALL traffic-controlled junctions
# For large_grid_multimodal: returns 16 junction IDs

topologies = {
    junction_id: topology_adapter.discover_junction(junction_id)
    for junction_id in signalized_ids
}

runtimes = {}
for junction_id, topology in topologies.items():
    # Build conflict graph, phases, sensors for each junction
    runtimes[junction_id] = { ... }

if not runtimes:
    raise RuntimeError("No usable green signal phases were discovered")
```

### Flow Through the System

**Stage 1: Discovery (automatic)**
```
TraCI connection to SUMO
  ↓
  sumo_topology_adapter.discover_signalized_junctions()
  ↓
  Returns: {"J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13", "J14", "J15", "J16"}
  (exact IDs depend on large_grid_multimodal network definition)
```

**Stage 2: Topology & Phase Analysis (per junction)**
```
For each junction_id in discovered_ids:
  ↓
  topology_adapter.discover_junction(junction_id)
    ↓ Returns: JunctionTopology with roads, lanes, movements
  ↓
  conflict_adapter.build_conflict_graph(junction_id, topology, sumo_metadata)
    ↓ Returns: ConflictGraph (movement compatibility)
  ↓
  phase_adapter.build_phase_representation(junction_id, sumo_metadata)
    ↓ Returns: list of decision phases (green-only, not yellow/red)
  ↓
  Store in runtimes[junction_id]
```

**Stage 3: Per-Junction Control Loop (main loop)**
```
For step in range(TOTAL_STEPS):
  ↓
  For each junction_id in runtimes:
    ↓
    if now - runtime["last_control"] >= CONTROL_INTERVAL:
      ↓
      # Gather sensor data for THIS junction
      sensor_state = sensor_builder.build_state()
      ↓
      # Add to history
      history.add_state(sensor_state)
      ↓
      # Predict demand (GNN on 36-D state)
      predictions = predictor.predict(history.get_padded_history())  # Shape: (36,)
      ↓
      # Assess risk
      safety_assessment = safety_adapter.assess()
      ↓
      # Make decision for THIS junction
      decision = decision_engine.decide(
          junction=junction_id,
          topology=runtimes[junction_id]["topology"],
          history=history,
          predictions=predictions,
          safety_assessment=safety_assessment
      )
      ↓
      # Execute phase change
      self._execute(junction_id, target_phase, phase_adapter)
      ↓
      runtime["last_control"] = now
      runtime["last_phase"] = executed_phase
```

**Stage 4: Per-Junction Metrics**
```
Each decision is logged with:
  {
    "time": simulation_time,
    "junction": junction_id,
    "method": "ai" | "fallback" | "safe_stop",
    "target_phase": phase_id,
    "safety": { "approved": bool, "risk_score": float },
    "emergency_active": bool,
    "metrics": { ... }
  }

All logged to results/final_runs/.../runtime.jsonl
```

### Key Points

1. **Junctions are processed independently:** Each junction's control decision is made separately based on its own topology and incoming traffic
2. **Sensor builder dynamically allocates 36 zones:** Distributes zones across discovered junctions
3. **No hardcoded junction list in canonical runtime:** Unlike `sumo_prediction_controller.py` (legacy), the main controller adapts to whatever junctions SUMO discovers
4. **Emergency preemption scales:** EmergencyCorridorManager works with any number of discovered junctions

---

## 7. Old SUMO Files Still Needed (Temporary Keeps)

### Critical for Backward Compatibility & Testing

| File | Purpose | Keep Until |
|------|---------|-----------|
| `simulation/configs/corridor.sumocfg` | CI/CD fallback; legacy tests; corridor-specific validation | All tests pass + documented in test transition plan |
| `simulation/configs/corridor_emergency.sumocfg` | Emergency mode fallback; emergency-specific tests | Emergency tests migrated to large_grid_multimodal_emergency |
| `simulation/networks/corridor.net.xml` | Baseline for regression tests; reference topology | Integration verified + test suite updated |
| `simulation/routes/corridor.rou.xml` | Baseline traffic patterns | All main tests pass with large_grid_multimodal |
| `simulation/routes/corridor_emergency.rou.xml` | Baseline emergency demonstration | Emergency demo works with large_grid_multimodal_emergency |

### Use Cases for Keeping Old Files

1. **Regression Testing:** Run same test suite against both environments
2. **Debugging:** Compare old (corridor) vs. new (large_grid_multimodal) behavior
3. **Performance Baseline:** Validate no degradation
4. **Documentation:** Show before/after
5. **CI/CD:** Fallback if new environment has issues

**Recommendation:** Keep all old corridor files indefinitely or at least 1 release cycle.

---

## 8. Files Safe to Remove ONLY After Successful Integration

### Success Criteria FIRST
Before removing anything, verify:

✓ `python run_system.py --steps 200 --seed 1` completes without error  
✓ `python run_system.py --demo --steps 200 --seed 1` completes (emergency mode)  
✓ `python run_system.py --gui --steps 200` runs with SUMO-GUI  
✓ All junctions discovered and controlled (should show 16 junctions)  
✓ 36-D predictions work (no shape mismatch errors)  
✓ Metrics output valid (results/final_runs/run_*/metrics.json)  
✓ Emergency preemption works (if emergency demo ran)  
✓ All test_*.py pass  
✓ Backward compat: `--config corridor.sumocfg` still works  

### Files That Can Be Removed

**Only after all success criteria met AND 1 week of production validation:**

| File | Reason | Retention Policy |
|------|--------|------------------|
| `simulation/routes/corridor.rou.backup.xml` | Backup of old routes; now obsolete | SAFE TO DELETE immediately |
| `simulation/networks/J1_backup.net.xml` | Old network backup | SAFE TO DELETE immediately |
| Test-specific corridor scenarios (if any) | Superseded by large_grid tests | CASE-BY-CASE |

**Do NOT remove:**
- `corridor.sumocfg`, `corridor_emergency.sumocfg`, corridor networks/routes (keep for regression)
- Any integration module files (non-functional changes only)

---

## 9. Exact Validation Commands

### Pre-Integration Validation

```powershell
# Verify new files exist
Test-Path "simulation/configs/large_grid_multimodal.sumocfg"
Test-Path "simulation/configs/large_grid_multimodal_gui.sumocfg"
Test-Path "simulation/networks/large_grid_multimodal.net.xml"
Test-Path "simulation/routes/large_grid_multimodal.rou.xml"
Test-Path "simulation/additional/large_grid_multimodal_stops.add.xml"

# Count junctions in network
grep -c "<junction" simulation/networks/large_grid_multimodal.net.xml
# Expected: 16 (or verify against requirement)
```

### Post-Integration Validation

#### 1. Normal Operation
```powershell
python run_system.py --steps 200 --seed 1 --output-dir results/validation/normal_run_1

# Verify output
$result = Get-Content results/validation/normal_run_1/manifest.json | ConvertFrom-Json
$result.config  # Should show: "simulation\\configs\\large_grid_multimodal.sumocfg"
```

**Check:** 
- No traci/SUMO errors
- Junctions discovered count (should be 16 or match network)
- All 36 sensors initialized
- Predictions run successfully
- Metrics generated

#### 2. Emergency Mode
```powershell
python run_system.py --demo --steps 200 --seed 1 --output-dir results/validation/emergency_run_1

# Verify
$log = Get-Content results/validation/emergency_run_1/runtime.jsonl | ConvertFrom-Json
$log[0].mode  # Should be: "emergency_demo"
$log[0].config  # Should show: "large_grid_multimodal_emergency.sumocfg"

# Count emergency events
$decisions = Get-Content results/validation/emergency_run_1/runtime.jsonl | ConvertFrom-Json | Where-Object event -eq decision
$emergency_decisions = $decisions | Where-Object { $_.emergency_active -eq $true }
$emergency_decisions.Count  # Should be > 0
```

**Check:**
- Emergency config loaded correctly
- Ambulance discovered
- At least one emergency preemption event logged

#### 3. GUI Mode
```powershell
# Launch and visually verify network renders correctly
python run_system.py --gui --steps 200 --seed 1

# Should see:
# - 16 junctions
# - Signal changes at each junction
# - Traffic flowing through multimodal network
# - (Close SUMO-GUI after visual inspection)
```

#### 4. Backward Compatibility
```powershell
# Old corridor config still works
python run_system.py --config simulation/configs/corridor.sumocfg --steps 100 --seed 1

# Verify
$result = Get-Content results/final_runs/run_*/manifest.json | Select-Object -Last 1 | ConvertFrom-Json
$result.config  # Should still show corridor.sumocfg
```

#### 5. Unit Tests
```powershell
# Run all test files
pytest test_phase2_4_final.py -v
pytest test_phase2_evaluation.py -v
pytest tests/ -v

# Verify all pass with new environment
```

**Check:**
- All test_*.py files pass
- test_phase2_4_final.py validates file existence
- No hardcoded corridor paths break tests

#### 6. State/Prediction Pipeline Validation
```powershell
# Inline validation (Python script)
python -c "
import sys; sys.path.insert(0, 'src')
from integration.traffic_history import TrafficHistory
from integration.sumo_prediction_controller import SUMOTrafficStateBuilder

history = TrafficHistory(history_length=10, num_sensors=36)
state = [0.5] * 36
history.add_state(state)
padded = history.get_padded_history()

print(f'History shape: {padded.shape}')
assert padded.shape == (10, 36), f'FAIL: Expected (10, 36), got {padded.shape}'
print('PASS: 36-D state pipeline compatible')
"
```

#### 7. Full Integration Test (Comprehensive)
```powershell
# Run 3 seeds, validate metrics consistency
foreach ($seed in 1, 2, 3) {
    python run_system.py --steps 200 --seed $seed --output-dir results/validation/seed_$seed
    
    $manifest = Get-Content results/validation/seed_$seed/manifest.json | ConvertFrom-Json
    Write-Host "Seed $seed - Config: $($manifest.config)"
    Write-Host "Seed $seed - Decisions: $($manifest.decision_count)"
    Write-Host "Seed $seed - Safety rejections: $($manifest.safety_gate_rejections)"
}

# Compare outputs
$seed1_metrics = Get-Content results/validation/seed_1/metrics.json | ConvertFrom-Json
$seed2_metrics = Get-Content results/validation/seed_2/metrics.json | ConvertFrom-Json
$seed3_metrics = Get-Content results/validation/seed_3/metrics.json | ConvertFrom-Json

# Verify determinism (same seed should give same results)
$seed1_again = Get-Content results/validation/seed_1_repeat/metrics.json | ConvertFrom-Json
Write-Host "Seed 1 metrics match on re-run: $(
    $seed1_metrics.decision_count -eq $seed1_again.decision_count
)"
```

### Validation Output Checklist

**For each validation run, confirm:**

- ✓ `runtime.jsonl` has system_start event with correct config
- ✓ Multiple decision events logged (at least 1 per junction per control interval)
- ✓ No traci errors in stderr
- ✓ `metrics.json` contains:
  - `decision_count > 0`
  - `safety_gate_rejections >= 0`
  - `fallback_decisions >= 0`
  - All timing statistics present
- ✓ `manifest.json` shows correct config path
- ✓ For emergency: `emergency_demo=true`, EMERGENCY_PREEMPT events logged
- ✓ Prediction adapter loaded successfully (no "PREDICTION unavailable" message)

---

## 10. Risks and Incompatibilities

### Risk 1: Too Many Roads / Sensor Overflow
**Severity:** MEDIUM  
**Description:** If large_grid_multimodal has more than 12 incoming roads across junctions, sensor mapping to 36-D space may fail or drop data.

**Detection:**
```python
# In Phase2SUMOController._build_runtime():
total_roads = sum(len(topology.roads) for topology in topologies.values())
print(f"Total roads discovered: {total_roads}")

# If total_roads * 3 > 36 × junction_count:
# → Sensor overflow likely
```

**Mitigation:**
- Validate SUMOTrafficStateBuilder packing logic
- If overflow: redefine NUM_SENSORS or adjust zones
- Document actual zone allocation per junction

---

### Risk 2: Network Complexity → Control Overhead
**Severity:** MEDIUM  
**Description:** 16 junctions × per-junction control decisions = 16× more phase changes and TraCI calls per control interval

**Detection:**
```powershell
# Monitor CPU/memory during run
# Check runtime.jsonl for decision latency
$decisions = Get-Content runtime.jsonl | ConvertFrom-Json | Where-Object event -eq decision
$decisions | Measure-Object -Property execution_time_ms -Average, Maximum
```

**Mitigation:**
- May need to increase `CONTROL_INTERVAL` from 10s to 15-20s
- Implement batch TraCI calls (low priority, existing design is sequential)
- Profile with --steps 500 to see cumulative effect

---

### Risk 3: Emergency Preemption Interaction
**Severity:** MEDIUM  
**Description:** With 16 junctions, multiple emergency preemptions could fire simultaneously, creating conflicting requests or phase locks.

**Detection:**
```python
# In emergency demo, look for:
# - EMERGENCY_PREEMPT events at multiple junctions simultaneously
# - EMERGENCY_REJECT events (phase not available)
# - Ambulance getting stuck or rerouted

$log = Get-Content runtime.jsonl | ConvertFrom-Json
$log | Where-Object { $_.event -eq "EMERGENCY_PREEMPT" } | Group-Object -Property time | Where-Object Count -gt 3
# If multiple preemptions at same time: risk of conflicts
```

**Mitigation:**
- Test emergency demo with multiple ambulances (future enhancement)
- Ensure EmergencyCorridorManager priority queuing works
- Document any preemption conflicts in results

---

### Risk 4: Prediction Model Overfitting to Corridor
**Severity:** LOW to MEDIUM  
**Description:** GNN model trained on corridor traffic patterns; performance may degrade on large_grid_multimodal topology.

**Detection:**
```powershell
# Compare prediction error:
# - Corridor: baseline
# - large_grid_multimodal: see if validation loss increases

# Check prediction outputs during run:
# If all predictions are near 0 or clipped: model not adapting
```

**Mitigation:**
- Use "safe baseline" mode if prediction confidence drops
- Log prediction stats for monitoring
- Consider retraining GNN on multimodal network (future work)
- For now, fallback controller handles prediction failure gracefully

---

### Risk 5: Backward Compatibility Breaking
**Severity:** LOW  
**Description:** Old scripts/tests expecting corridor config may fail silently.

**Detection:**
- All test_*.py files pass
- No hardcoded "corridor" paths in code (grep check)
- CLI `--config` option still works

**Mitigation:**
- Keep corridor files indefinitely
- Backward compat test: `--config corridor.sumocfg` must work
- Document breaking changes (none expected)

---

### Risk 6: GUI Config Mismatch
**Severity:** LOW  
**Description:** `large_grid_multimodal_gui.sumocfg` referenced in structure but may not exist or may have wrong viewport.

**Detection:**
```powershell
Test-Path "simulation/configs/large_grid_multimodal_gui.sumocfg"
# If not found, GUI mode will fail
```

**Mitigation:**
- Verify GUI config file exists with correct name
- Set appropriate `<viewer-settings>` in GUI config
- Test with `--gui --steps 50` (short run for quick visual check)

---

### Risk 7: Determinism Loss
**Severity:** LOW  
**Description:** Different junction topologies, number of junctions, or route patterns could affect determinism with fixed seed.

**Detection:**
```powershell
# Run same seed 3× and compare metrics
# If they diverge: non-deterministic behavior
python run_system.py --steps 200 --seed 42 --output-dir run_42_a
python run_system.py --steps 200 --seed 42 --output-dir run_42_b
python run_system.py --steps 200 --seed 42 --output-dir run_42_c

# Compare metrics files
```

**Mitigation:**
- SUMO seed is set in run_system.py
- Python random seed not currently set (could add if needed)
- Large network may expose any existing non-determinism in decision engine
- Document actual behavior if non-deterministic (acceptable for safety studies)

---

### Risk 8: Old Emergency Scenario Still Referenced
**Severity:** LOW  
**Description:** Tests or CI/CD might hardcode corridor_emergency path.

**Detection:**
```powershell
grep -r "corridor_emergency" src/ tests/
```

**Mitigation:**
- Update all references to use EMERGENCY_SUMO_CONFIG constant (already in code)
- Verify test_phase2_4_final.py imports correctly
- Create large_grid_multimodal_emergency.sumocfg + routes

---

## Summary: What Must Happen, In Order

### Phase 0: Preparation (Before Code Changes)
- [ ] Verify large_grid_multimodal.sumocfg, .net.xml, .rou.xml exist in simulation/
- [ ] Create large_grid_multimodal_emergency.sumocfg (references same network, emergency routes)
- [ ] Create large_grid_multimodal_emergency.rou.xml (include ambulance)
- [ ] Create large_grid_multimodal_gui.sumocfg (if GUI validation needed)
- [ ] Count junctions in large_grid_multimodal.net.xml (verify 16)
- [ ] Verify 36-D sensor mapping will work (< 12 incoming roads across junctions, or adjust strategy)

### Phase 1: Configuration Change (Minimal Code Edit)
- [ ] Edit `src/integration/sumo_decision_controller.py`:
  - Line 41: Update SUMO_CONFIG to point to large_grid_multimodal.sumocfg
  - Lines 42-44: Update EMERGENCY_SUMO_CONFIG to point to large_grid_multimodal_emergency.sumocfg
- [ ] Verify imports still work: `python -c "from src.integration.sumo_decision_controller import SUMO_CONFIG, EMERGENCY_SUMO_CONFIG; print(SUMO_CONFIG)"`

### Phase 2: Validation
- [ ] Run normal mode: `python run_system.py --steps 200 --seed 1`
  - Check: Config printed correctly, 16 junctions discovered, 36-D state initialized, metrics generated
- [ ] Run emergency mode: `python run_system.py --demo --steps 200 --seed 1`
  - Check: Emergency config used, ambulance detected, preemption triggered
- [ ] Run with GUI: `python run_system.py --gui --steps 100 --seed 1`
  - Check: Network renders, 16 junctions visible, signal changes visible
- [ ] Run backward compat: `python run_system.py --config simulation/configs/corridor.sumocfg --steps 100 --seed 1`
  - Check: Old config still works
- [ ] Run test suite: `pytest test_phase2_4_final.py test_phase2_evaluation.py -v`
  - Check: All tests pass

### Phase 3: Stability (1-2 weeks)
- [ ] Monitor production runs
- [ ] Collect metrics from 10+ runs with different seeds
- [ ] Validate prediction accuracy (model doesn't overfit to corridor)
- [ ] Document any performance changes (decision latency, etc.)

### Phase 4: Cleanup (After Successful Validation)
- [ ] Remove corridor.rou.backup.xml (not needed)
- [ ] Remove J1_backup.net.xml (not needed)
- [ ] Keep all other corridor files (for regression testing)
- [ ] Archive old validation runs to backups/

---

## Additional Notes

### Decision Engine Scope
The decision engine and all safety, fallback, and emergency logic are **network-topology-agnostic**. They work by:
1. Receiving discovered junctions
2. Building per-junction topologies
3. Making decisions per junction independently
4. Logging all decisions uniformly

This design is robust and requires **NO changes** to adapt to large_grid_multimodal.

### Prediction Model Assumption
The GNN model was trained on corridor traffic patterns. Performance on large_grid_multimodal is an **empirical question**. The system is designed to fall back gracefully if prediction fails, but actual traffic outcomes may differ. This is expected and acceptable for an adaptive controller evaluation.

### Timeline Estimate
- Configuration changes: 5 minutes
- First test run: 5 minutes
- Full validation suite: 30 minutes
- Stability monitoring: 1-2 weeks before cleanup

**Total integration time: < 2 hours (+ monitoring period)**

---

END OF INTEGRATION PLAN

