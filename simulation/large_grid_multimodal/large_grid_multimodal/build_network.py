#!/usr/bin/env python3
"""Build the multimodal large-grid SUMO network.

Runs netconvert over the plain-XML node/edge sources to generate
    simulation/networks/large_grid_multimodal.net.xml
and then generates
    simulation/additional/large_grid_multimodal_stops.add.xml
(bus stops and train-station platforms, resolved against the built
network so lane IDs and permissions are always correct).

netconvert generates the multimodal infrastructure natively:
  --sidewalks.guess     sidewalk lane on every road edge (40 km/h edges
                        are below the 13.89 m/s guess threshold)
  --crossings.guess     signalized pedestrian crossings at the 16 TLS
                        junctions (never across rail edges: 20 m/s is
                        above the speed threshold)
  --bikelanes.guess     protected bike lane on every road edge
  --walkingareas        walking areas so persons can move through all
                        junctions
Rail level crossings come from the explicit rail_crossing nodes in the
node file; netconvert turns them into automatic barrier junctions.

Reproducibility: rerun this script any time the .nod.xml/.edg.xml
sources change. Never edit the generated .net.xml by hand.
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCENARIO_DIR.parents[1]
NET_FILE = PROJECT_ROOT / "simulation" / "networks" / "large_grid_multimodal.net.xml"
STOPS_FILE = (
    PROJECT_ROOT / "simulation" / "additional" / "large_grid_multimodal_stops.add.xml"
)

# --- Public-transport stop definitions -------------------------------------
# (stop_id, edge_id, position_m). Edge IDs are stable: they come straight
# from large_grid_multimodal.edg.xml, so this table is the single source
# of truth for stop placement. The lane each stop sits on (a car lane
# for buses, the rail lane for trains) and the sidewalk lane used for
# pedestrian <access> are resolved against the built network.
BUS_STOPS = [
    ("bs_l1_J05J06", "J05_J06_p1", 100),  # line 1, west of the rail line
    ("bs_l1_J06J07", "J06_J07", 240),
    ("bs_l1_J07J08", "J07_J08", 240),
    ("bs_l2_J02J06", "J02_J06", 240),  # line 2, north-south via centre
    ("bs_l2_J06J10", "J06_J10", 240),
    ("bs_l2_J10J14", "J10_J14", 240),
    ("bs_l3_J15J11", "J15_J11", 240),  # line 3, south-north via centre
    ("bs_l3_J11J07", "J11_J07", 240),
    ("bs_l3_J07J03", "J07_J03", 240),
    ("bs_l4_J09J10", "J09_J10_p1", 100),  # line 4, west of the rail line
    ("bs_l4_J10J11", "J10_J11", 240),
    ("bs_l4_J11J12", "J11_J12", 240),
]

# (stop_id, rail_edge_id, position_m): one platform per direction at each
# of the three stations (A south, B centre, C north).
TRAIN_STOPS = [
    ("ts_stA_south", "rw_stA_s0", 5),
    ("ts_stB_south", "rw_stB_rc300", 5),
    ("ts_stC_south", "rw_stC_rc900", 5),
    ("ts_stA_north", "re_stA_rc0", 5),
    ("ts_stB_north", "re_stB_rc600", 5),
    ("ts_stC_north", "re_stC_n0", 5),
]


def find_netconvert() -> str:
    """Locate netconvert.exe the same way as large_grid/build_network.py."""
    candidates = []
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.append(Path(sumo_home) / "bin" / "netconvert.exe")
    from shutil import which

    found = which("netconvert")
    if found:
        candidates.append(Path(found))
    for program_files in (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramW6432"),
    ):
        if program_files:
            candidates.append(
                Path(program_files) / "Eclipse" / "Sumo" / "bin" / "netconvert.exe"
            )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("netconvert not found (set SUMO_HOME or add it to PATH)")


def run_netconvert() -> None:
    netconvert = find_netconvert()
    command = [
        netconvert,
        "--node-files", str(SCENARIO_DIR / "large_grid_multimodal.nod.xml"),
        "--edge-files", str(SCENARIO_DIR / "large_grid_multimodal.edg.xml"),
        "--output-file", str(NET_FILE),
        # identical road-signal settings as the validated large_grid build
        "--tls.default-type", "static",
        "--no-turnarounds.tls", "true",
        "--tls.cycle.time", "90",
        "--junctions.corner-detail", "5",
        # multimodal infrastructure
        "--sidewalks.guess", "true",
        "--sidewalks.guess.max-speed", "13.89",
        "--crossings.guess", "true",
        "--bikelanes.guess", "true",
        "--bikelanes.guess.max-speed", "13.89",
        "--walkingareas", "true",
    ]
    print(f"netconvert: {' '.join(command[1:])}")
    subprocess.run(command, check=True)
    if not NET_FILE.exists():
        raise RuntimeError(f"netconvert produced no output at {NET_FILE}")


# --- Built-network inspection helpers ---------------------------------------

def _distance_to_shape(point: tuple[float, float], shape) -> float:
    """Minimum distance from a point to a polyline."""
    px, py = point
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        if length2 == 0:
            best = min(best, ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5)
            continue
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length2))
        best = min(best, ((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2) ** 0.5)
    return best


def _lane_permissions(lane) -> set[str]:
    """Return the set of vClasses allowed on a SUMO lane."""
    permissions = getattr(lane, "permissions", None)
    if permissions is None:
        permissions = lane.getPermissions()
    if permissions is None:
        return set()
    return set(permissions)


def _lane_allows(lane, vclass: str) -> bool:
    return vclass in _lane_permissions(lane)


def _lane_is_exclusive(lane, vclass: str) -> bool:
    """True if the lane permits exactly one vClass (e.g. rail-only track)."""
    return _lane_permissions(lane) == {vclass}


def _pick_lane(edge, vclass: str):
    """Rightmost lane of the edge that allows vclass (lowest lane index)."""
    for lane in sorted(edge.getLanes(), key=lambda l: l.getID()):
        if _lane_allows(lane, vclass):
            return lane
    return None


def verify_and_collect(net) -> dict:
    """Check the built network and gather lanes needed for the stops file."""
    nodes = net.getNodes()
    tls_nodes = [n for n in nodes if n.getType() == "traffic_light"]
    rail_crossings = [n for n in nodes if n.getType() == "rail_crossing"]
    if len(tls_nodes) != 16:
        raise RuntimeError(f"expected 16 traffic_light junctions, got {len(tls_nodes)}")
    if len(rail_crossings) != 8:
        raise RuntimeError(
            f"expected 8 rail_crossing junctions, got {len(rail_crossings)}"
        )

    sidewalk_lanes = []
    bikelanes = []
    rail_lanes = []
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        for lane in edge.getLanes():
            if _lane_allows(lane, "pedestrian"):
                sidewalk_lanes.append(lane)
            if _lane_allows(lane, "bicycle"):
                bikelanes.append(lane)
            if _lane_is_exclusive(lane, "rail"):
                rail_lanes.append(lane)
    if not sidewalk_lanes:
        raise RuntimeError("no sidewalks were generated (--sidewalks.guess failed)")
    if not bikelanes:
        raise RuntimeError("no bike lanes were generated (--bikelanes.guess failed)")
    if len(rail_lanes) != 16:
        raise RuntimeError(f"expected 16 rail lanes, got {len(rail_lanes)}")

    # walkingareas / crossings via raw XML (sumolib hides the function attr)
    tree = ET.parse(NET_FILE)
    functions = [e.get("function") for e in tree.getroot() if e.tag == "edge"]
    n_walkingareas = functions.count("walkingarea")
    n_crossings = functions.count("crossing")
    if n_walkingareas == 0:
        raise RuntimeError("no walkingareas were generated")
    if n_crossings == 0:
        raise RuntimeError("no pedestrian crossings were generated")

    # every stop edge must exist in the built net
    for _, edge_id, _ in BUS_STOPS + TRAIN_STOPS:
        if net.getEdge(edge_id) is None:
            raise RuntimeError(f"stop edge '{edge_id}' missing from built network")

    print(f"Junctions: 16 traffic_light, {len(rail_crossings)} rail_crossing")
    print(
        f"Pedestrian infra: {len(sidewalk_lanes)} sidewalk lanes, "
        f"{n_crossings} crossings, {n_walkingareas} walkingareas"
    )
    print(f"Bike lanes: {len(bikelanes)}, rail lanes: {len(rail_lanes)}")
    return {"sidewalk_lanes": sidewalk_lanes}


def _nearest_sidewalk_lane(net, point):
    """Sidewalk lane whose shape passes closest to the given point."""
    best_lane, best_distance = None, float("inf")
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        for lane in edge.getLanes():
            if "pedestrian" not in _lane_permissions(lane):
                continue  # only true sidewalk lanes
            distance = _distance_to_shape(point, lane.getShape())
            if distance < best_distance:
                best_lane, best_distance = lane, distance
    if best_lane is None:
        raise RuntimeError(f"no sidewalk lane found near {point}")
    return best_lane, best_distance


TRAIN_LENGTH = 40.0  # must match the vType length used in the route file


def generate_stops_file(net) -> None:
    """Write bus stops (on car lanes) and train stops (on rail lanes),
    each with a pedestrian <access> connection to the walking network."""
    lines = [
        "<!-- Public-transport stops for the multimodal large grid.",
        "     GENERATED by simulation/large_grid_multimodal/build_network.py",
        "     from the BUS_STOPS / TRAIN_STOPS tables in that script.",
        "     Bus stops sit on a car lane with a short access link to the",
        "     sidewalk of the same edge; train platforms sit on the rail",
        "     lanes with an access link to the nearest sidewalk lane. -->",
        "<additional>",
    ]
    for stop_id, edge_id, pos in BUS_STOPS:
        edge = net.getEdge(edge_id)
        lane = _pick_lane(edge, "bus")
        if lane is None:
            raise RuntimeError(f"no bus-compatible lane on edge {edge_id}")
        sidewalk = _pick_lane(edge, "pedestrian")
        length = max(edge.getLength(), 0.0)
        start, end = max(0.0, pos - 10), min(length, pos + 10)
        if sidewalk is not None:
            lines.append(
                f'    <busStop id="{stop_id}" lane="{lane.getID()}" '
                f'startPos="{start:.1f}" endPos="{end:.1f}">'
            )
            lines.append(
                f'        <access lane="{sidewalk.getID()}" length="5"/>'
            )
            lines.append("    </busStop>")
        else:
            lines.append(
                f'    <busStop id="{stop_id}" lane="{lane.getID()}" '
                f'startPos="{start:.1f}" endPos="{end:.1f}"/>'
            )
    for stop_id, edge_id, pos in TRAIN_STOPS:
        edge = net.getEdge(edge_id)
        lane = edge.getLanes()[0]
        if not _lane_allows(lane, "rail"):
            raise RuntimeError(f"edge {edge_id} is not a rail edge")
        # platform position in world coordinates for the access search
        shape = lane.getShape()
        platform = _point_along_shape(shape, pos)
        sidewalk, distance = _nearest_sidewalk_lane(net, platform)
        length = max(edge.getLength(), 0.0)
        start = max(0.0, min(pos, length - TRAIN_LENGTH))
        end = min(length, start + TRAIN_LENGTH)
        lines.append(
            f'    <trainStop id="{stop_id}" lane="{lane.getID()}" '
            f'startPos="{start:.1f}" endPos="{end:.1f}">'
        )
        lines.append(
            f'        <access lane="{sidewalk.getID()}" '
            f'length="{max(10.0, distance):.0f}"/>'
        )
        lines.append("    </trainStop>")
    lines.append("</additional>")
    STOPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOPS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Stops file written: {STOPS_FILE} ({len(BUS_STOPS)} bus, {len(TRAIN_STOPS)} train)")


def _point_along_shape(shape, pos: float):
    """Point at 'pos' metres along a polyline (clamped to its ends)."""
    remaining = pos
    for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if remaining <= length:
            t = remaining / length if length else 0.0
            return (x1 + t * dx, y1 + t * dy)
        remaining -= length
    return shape[-1]


def main() -> int:
    run_netconvert()
    import sumolib  # deferred: only needed after a successful build

    net = sumolib.net.readNet(str(NET_FILE))
    verify_and_collect(net)
    generate_stops_file(net)
    print("Multimodal network build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())