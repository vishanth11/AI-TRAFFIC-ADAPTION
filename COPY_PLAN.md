# Copy Plan: Integrate large_grid_multimodal from C:\Users\Varun\Desktop\sumo

**Source:** `C:\Users\Varun\Desktop\sumo\simulation\`  
**Destination:** `C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\`

**Status:** PLAN ONLY — No copying yet.

---

## 1. Source Files/Directories Found in C:\Users\Varun\Desktop\sumo\simulation

### Existing Corridor & Four-Way (Already in Destination)
- `configs/corridor.sumocfg` ✓
- `configs/corridor_emergency.sumocfg` ✓
- `configs/four_way.sumocfg` ✓
- `networks/corridor.net.xml` ✓
- `networks/four_way.net.xml` ✓
- `networks/J1_backup.net.xml` ✓
- `routes/corridor.rou.xml` ✓
- `routes/corridor.rou.backup.xml` ✓
- `routes/corridor_emergency.rou.xml` ✓
- `routes/four_way.rou.xml` ✓

### NEW: Large Grid Single-Modal (Supporting Infrastructure)
- `configs/large_grid.sumocfg` (NEW)
- `configs/large_grid_gui.sumocfg` (NEW)
- `networks/large_grid.net.xml` (NEW)
- `routes/large_grid.rou.xml` (NEW)
- `large_grid/` directory (source build files: .nod.xml, .edg.xml, build_network.py, README.md)

### NEW: Large Grid Multimodal (MAIN TARGET)
- `configs/large_grid_multimodal.sumocfg` (NEW - **CRITICAL**)
- `configs/large_grid_multimodal_gui.sumocfg` (NEW - **CRITICAL**)
- `networks/large_grid_multimodal.net.xml` (NEW - **CRITICAL**)
- `routes/large_grid_multimodal.rou.xml` (NEW - **CRITICAL** - contains ambulances)
- `additional/large_grid_multimodal_stops.add.xml` (NEW - **CRITICAL** - referenced by config)
- `large_grid_multimodal/` directory (source build files: .nod.xml, .edg.xml, build_network.py, validate_multimodal.py, README.md)

---

## 2. Exact Copy Destinations

### Minimum Required (Just the Multimodal Environment)

| Source File | Destination | Type | Purpose |
|-------------|-------------|------|---------|
| `configs/large_grid_multimodal.sumocfg` | `simulation/configs/large_grid_multimodal.sumocfg` | Config | Main scenario config |
| `configs/large_grid_multimodal_gui.sumocfg` | `simulation/configs/large_grid_multimodal_gui.sumocfg` | Config | GUI-enabled variant |
| `networks/large_grid_multimodal.net.xml` | `simulation/networks/large_grid_multimodal.net.xml` | Network | Network topology (generated) |
| `routes/large_grid_multimodal.rou.xml` | `simulation/routes/large_grid_multimodal.rou.xml` | Routes | Traffic + ambulances |
| `additional/large_grid_multimodal_stops.add.xml` | `simulation/additional/large_grid_multimodal_stops.add.xml` | Additional | Bus/train stops |
| `large_grid_multimodal/` | `simulation/large_grid_multimodal/` | Directory | Source build files |

### Optional (Large Grid Single-Modal, for Comparison/Testing)

| Source File | Destination | Type | Purpose |
|-------------|-------------|------|---------|
| `configs/large_grid.sumocfg` | `simulation/configs/large_grid.sumocfg` | Config | Baseline grid scenario |
| `configs/large_grid_gui.sumocfg` | `simulation/configs/large_grid_gui.sumocfg` | Config | GUI-enabled grid variant |
| `networks/large_grid.net.xml` | `simulation/networks/large_grid.net.xml` | Network | Network topology |
| `routes/large_grid.rou.xml` | `simulation/routes/large_grid.rou.xml` | Routes | Traffic routes |
| `large_grid/` | `simulation/large_grid/` | Directory | Source build files |

---

## 3. Filename/Path Conflicts

**GOOD NEWS: Zero Conflicts**

The destination directory (`AI-TRAFFIC-ADAPTION\simulation\`) contains only:
- `configs/corridor.sumocfg`, `corridor_emergency.sumocfg`, `four_way.sumocfg`
- `networks/corridor.net.xml`, `four_way.net.xml`, `J1_backup.net.xml`
- `routes/corridor.rou.xml`, `corridor.rou.backup.xml`, `corridor_emergency.rou.xml`, `four_way.rou.xml`

**No `large_grid*` or `additional/` directories exist in destination.**

All filenames are unique. Safe to copy without overwrites.

---

## 4. Internal Relative-Path References (VERIFIED COMPATIBLE)

### Config Files: large_grid_multimodal.sumocfg

```xml
<input>
    <net-file value="../networks/large_grid_multimodal.net.xml"/>
    <route-files value="../routes/large_grid_multimodal.rou.xml"/>
    <additional-files value="../additional/large_grid_multimodal_stops.add.xml"/>
</input>
```

**Status:** ✓ All references use relative paths (`../` from configs/ to networks/, routes/, additional/)

**Verification:** After copying, paths will resolve correctly:
- `simulation/configs/../networks/large_grid_multimodal.net.xml` → `simulation/networks/large_grid_multimodal.net.xml` ✓
- `simulation/configs/../routes/large_grid_multimodal.rou.xml` → `simulation/routes/large_grid_multimodal.rou.xml` ✓
- `simulation/configs/../additional/large_grid_multimodal_stops.add.xml` → `simulation/additional/large_grid_multimodal_stops.add.xml` ✓

### Config File: large_grid_multimodal_gui.sumocfg

```xml
<input>
    <net-file value="../networks/large_grid_multimodal.net.xml"/>
    <route-files value="../routes/large_grid_multimodal.rou.xml"/>
    <additional-files value="../additional/large_grid_multimodal_stops.add.xml"/>
</input>
```

**Status:** ✓ Same relative paths as non-GUI config

### Routes File: large_grid_multimodal.rou.xml

```xml
<route id="bus_l1_w_e" edges="in_w_J05 J05_J06_p1 ...">
    <stop busStop="bs_l1_J05J06" duration="20" />
    ...
</route>
```

**Status:** ✓ No file references; only internal edge and stop IDs (defined in .rou.xml itself)

### Additional File: large_grid_multimodal_stops.add.xml

```xml
<busStop id="bs_l1_J05J06" lane="J05_J06_p1_2" startPos="90.0" endPos="110.0">
    <access lane="J05_J06_p1_0" length="5"/>
</busStop>
```

**Status:** ✓ No file references; only lane IDs (reference network structure by name)

### Network File: large_grid_multimodal.net.xml

**File Header (First 20 lines):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- generated on 2026-09-07T15:43:08.805212+05:30 by Eclipse SUMO netconvert 1.27.1
<netconvertConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">
    <input>
        <node-files value="C:\Users\Varun\Desktop\sumo\simulation\large_grid_multimodal\large_grid_multimodal.nod.xml"/>
        <edge-files value="C:\Users\Varun\Desktop\sumo\simulation\large_grid_multimodal\large_grid_multimodal.edg.xml"/>
    </input>
    <output>
        <output-file value="C:\Users\Varun\Desktop\sumo\simulation\networks\large_grid_multimodal.net.xml"/>
    </output>
```

**Status:** ⚠️ CONTAINS ABSOLUTE PATHS (metadata only)

**Impact:** NONE - These are netconvert build metadata embedded in the file. They don't affect runtime operation.

**Verification:** The .net.xml file is a GENERATED artifact. Once copied, SUMO/TraCI will load it without needing to follow the netconvert paths. The metadata is read-only documentation.

### Build Source Directories

**large_grid_multimodal/ contents:**
- `build_network.py` — build script (path references to source, not critical for runtime)
- `validate_multimodal.py` — validation script (reads .rou.xml, .net.xml by relative path)
- `large_grid_multimodal.nod.xml` — node definitions for build (used to generate .net.xml)
- `large_grid_multimodal.edg.xml` — edge definitions for build (used to generate .net.xml)
- `README.md` — documentation
- `__pycache__/` — Python cache (can be excluded)

**Status:** ✓ Build scripts use relative paths if called from the correct directory. Safe to copy.

---

## 5. Exact Copy Plan

### CRITICAL PATH (Must execute in order)

#### Step 1: Create Directories (if they don't exist)
```powershell
New-Item -ItemType Directory -Path "simulation/additional" -Force
New-Item -ItemType Directory -Path "simulation/large_grid_multimodal" -Force
```

#### Step 2: Copy Configuration Files (4 files)
```
Source → Destination
─────────────────────────────────────────────────────────────────
C:\Users\Varun\Desktop\sumo\simulation\configs\large_grid_multimodal.sumocfg
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\configs\large_grid_multimodal.sumocfg

C:\Users\Varun\Desktop\sumo\simulation\configs\large_grid_multimodal_gui.sumocfg
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\configs\large_grid_multimodal_gui.sumocfg
```

#### Step 3: Copy Network File (1 file)
```
C:\Users\Varun\Desktop\sumo\simulation\networks\large_grid_multimodal.net.xml
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\networks\large_grid_multimodal.net.xml
```

#### Step 4: Copy Routes File (1 file)
```
C:\Users\Varun\Desktop\sumo\simulation\routes\large_grid_multimodal.rou.xml
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\routes\large_grid_multimodal.rou.xml
```

#### Step 5: Copy Additional Files (1 file)
```
C:\Users\Varun\Desktop\sumo\simulation\additional\large_grid_multimodal_stops.add.xml
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\additional\large_grid_multimodal_stops.add.xml
```

#### Step 6: Copy Build Directory (entire directory)
```
C:\Users\Varun\Desktop\sumo\simulation\large_grid_multimodal\
  → C:\Users\Varun\Desktop\AI-TRAFFIC-ADAPTION\simulation\large_grid_multimodal\
```

**Exclude from copy:**
- `__pycache__/` subdirectory (can be regenerated)

**Include:**
- `build_network.py`
- `validate_multimodal.py`
- `large_grid_multimodal.nod.xml`
- `large_grid_multimodal.edg.xml`
- `README.md`

---

## 6. Post-Copy Validation Checklist

### File Existence Check
```powershell
Test-Path "simulation/configs/large_grid_multimodal.sumocfg"
Test-Path "simulation/configs/large_grid_multimodal_gui.sumocfg"
Test-Path "simulation/networks/large_grid_multimodal.net.xml"
Test-Path "simulation/routes/large_grid_multimodal.rou.xml"
Test-Path "simulation/additional/large_grid_multimodal_stops.add.xml"
Test-Path "simulation/large_grid_multimodal/README.md"
```

### Relative Path Verification
```powershell
# Verify config file can find network by relative path
$config = Get-Content "simulation/configs/large_grid_multimodal.sumocfg"
$config -match "../networks/large_grid_multimodal.net.xml"  # Should match

# Verify files referenced by config exist
Test-Path "simulation/networks/large_grid_multimodal.net.xml"  # Must be True
Test-Path "simulation/routes/large_grid_multimodal.rou.xml"  # Must be True
Test-Path "simulation/additional/large_grid_multimodal_stops.add.xml"  # Must be True
```

### Network File Integrity
```powershell
# Check that network file is valid XML and mentions junctions
Select-String -Path "simulation/networks/large_grid_multimodal.net.xml" -Pattern "<junction" | Measure-Object
# Should return count > 0 (indicates junctions were found)
```

### Routes File Integrity
```powershell
# Check that routes include ambulances
Select-String -Path "simulation/routes/large_grid_multimodal.rou.xml" -Pattern "ambulance" | Measure-Object
# Should return count > 0 (indicates ambulance definitions exist)
```

---

## 7. Optional: Large_grid Single-Modal (For Comparison)

If you want to keep the intermediate large_grid variant for testing/comparison:

| Source File | Destination |
|-------------|-------------|
| `configs/large_grid.sumocfg` | `simulation/configs/large_grid.sumocfg` |
| `configs/large_grid_gui.sumocfg` | `simulation/configs/large_grid_gui.sumocfg` |
| `networks/large_grid.net.xml` | `simulation/networks/large_grid.net.xml` |
| `routes/large_grid.rou.xml` | `simulation/routes/large_grid.rou.xml` |
| `large_grid/` | `simulation/large_grid/` |

**Recommendation:** Copy these too (minimal additional space, useful for regression testing).

---

## 8. Summary

### Minimum Files to Copy (for large_grid_multimodal only)
- **6 files** (2 configs + 1 network + 1 routes + 1 additional + 1 directory)
- **~15-20 MB** total (mostly the .net.xml file)

### Total Size Estimate
```
large_grid_multimodal.sumocfg                      ~1 KB
large_grid_multimodal_gui.sumocfg                  ~1 KB
large_grid_multimodal.net.xml                      ~8-10 MB
large_grid_multimodal.rou.xml                      ~100-200 KB
large_grid_multimodal_stops.add.xml                ~20 KB
large_grid_multimodal/ directory                   ~100-200 KB
────────────────────────────────────────────────────────────
TOTAL (multimodal only)                            ~8-10 MB

TOTAL (with large_grid single-modal)              ~15-20 MB
```

---

## 9. No Conflicts or Issues Found

✓ No filename conflicts
✓ All relative paths are compatible
✓ No absolute path dependencies (only metadata)
✓ No circular references
✓ Ambulances already present in routes (CASE 1 scenario)
✓ No modifications needed to controller code for path resolution

---

## NEXT STEPS (after approval)

1. Create `simulation/additional/` directory
2. Create `simulation/large_grid_multimodal/` directory
3. Copy 6 files as specified above
4. Run validation checks
5. Then modify `src/integration/sumo_decision_controller.py` (only 1 line change: SUMO_CONFIG path)

---

END OF COPY PLAN

