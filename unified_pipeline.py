"""
Phase 6 - Unified traffic perception pipeline.
Combines yolo11n.pt (tracking) + emergency.pt (emergency detection).
Outputs structured JSON per frame.
"""

from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json
import sys
import os


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

GENERAL_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
EMERGENCY_MODEL_PATH = MODELS_DIR / "emergency.pt"

JSON_OUTPUT_PATH = OUTPUT_DIR / "perception_output.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAFFIC CLASSES
# ============================================================

TRAFFIC_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# CONFIGURATION
# ============================================================

TRAJECTORY_LEN = 30
PIXELS_PER_METER = 15
FPS = 30


# ============================================================
# DIRECTION
# ============================================================

def get_direction(positions: list) -> str:
    if len(positions) < 2:
        return "unknown"

    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]

    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"

    return "down" if dy > 0 else "up"


# ============================================================
# SPEED
# ============================================================

def get_speed_kmh(positions: list) -> float:
    if len(positions) < 2:
        return 0.0

    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]

    pixel_dist = (dx ** 2 + dy ** 2) ** 0.5

    frames = len(positions) - 1

    if frames <= 0:
        return 0.0

    mps = (pixel_dist / PIXELS_PER_METER) / (frames / FPS)

    return round(mps * 3.6, 1)


# ============================================================
# IOU
# ============================================================

def iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter == 0:
        return 0.0

    area1 = (
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    denominator = area1 + area2 - inter

    if denominator <= 0:
        return 0.0

    return inter / denominator


# ============================================================
# EMERGENCY MATCHING
# ============================================================

def is_emergency(box, emergency_boxes, threshold=0.1) -> bool:
    for ebox in emergency_boxes:
        if iou(box, ebox) >= threshold:
            return True

    return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(source: str, mode: str):

    print(f"\n--- Phase 6 Unified Pipeline: {source} ---")

    # --------------------------------------------------------
    # Verify model files
    # --------------------------------------------------------

    if not GENERAL_MODEL_PATH.exists():
        print(f"ERROR: YOLO model not found:")
        print(f"       {GENERAL_MODEL_PATH}")
        print("Run: python setup_models.py")
        sys.exit(1)

    if not EMERGENCY_MODEL_PATH.exists():
        print(f"ERROR: Emergency model not found:")
        print(f"       {EMERGENCY_MODEL_PATH}")
        print("Run: python setup_models.py")
        sys.exit(1)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(f"[MODEL] Loading YOLO11n:")
    print(f"        {GENERAL_MODEL_PATH}")

    general_model = YOLO(str(GENERAL_MODEL_PATH))

    print(f"[MODEL] Loading emergency detector:")
    print(f"        {EMERGENCY_MODEL_PATH}")

    emergency_model = YOLO(str(EMERGENCY_MODEL_PATH))

    print("[MODEL] Both models loaded successfully.")

    # --------------------------------------------------------
    # Video/image mode
    # --------------------------------------------------------

    is_video = mode == "video"

    trajectories = defaultdict(list)

    all_frames = []

    # --------------------------------------------------------
    # General vehicle tracking
    # --------------------------------------------------------

    general_results = general_model.track(
        source=source,
        conf=0.25,
        device=0,
        classes=list(TRAFFIC_CLASSES.keys()),
        tracker="bytetrack.yaml",
        persist=True,
        stream=is_video,
    )

    # --------------------------------------------------------
    # Emergency detection
    # --------------------------------------------------------

    emergency_results = emergency_model.predict(
        source=source,
        conf=0.30,
        device=0,
        stream=is_video,
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_count = 0

    for g_result, e_result in zip(
        general_results,
        emergency_results
    ):

        frame_count += 1

        # ----------------------------------------------------
        # Emergency boxes
        # ----------------------------------------------------

        emergency_boxes = []

        for box in e_result.boxes:

            if int(box.cls) == 0:

                emergency_boxes.append(
                    list(
                        map(
                            int,
                            box.xyxy[0]
                        )
                    )
                )

        # ----------------------------------------------------
        # Frame detections
        # ----------------------------------------------------

        frame_detections = []

        if g_result.boxes.id is None:

            all_frames.append(
                {
                    "frame": frame_count,
                    "detections": []
                }
            )

            continue

        # ----------------------------------------------------
        # Vehicle detections
        # ----------------------------------------------------

        for box in g_result.boxes:

            track_id = int(box.id)

            cls_id = int(box.cls)

            label = TRAFFIC_CLASSES.get(
                cls_id,
                "unknown"
            )

            conf = round(
                float(box.conf),
                2
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ------------------------------------------------
            # Trajectory
            # ------------------------------------------------

            trajectories[track_id].append(
                (cx, cy)
            )

            if len(
                trajectories[track_id]
            ) > TRAJECTORY_LEN:

                trajectories[track_id].pop(0)

            traj = trajectories[track_id]

            # ------------------------------------------------
            # Motion information
            # ------------------------------------------------

            direction = get_direction(traj)

            speed = get_speed_kmh(traj)

            # ------------------------------------------------
            # Emergency status
            # ------------------------------------------------

            emergency = is_emergency(
                [x1, y1, x2, y2],
                emergency_boxes
            )

            # ------------------------------------------------
            # Structured detection
            # ------------------------------------------------

            detection = {

                "vehicle_id": track_id,

                "type":
                    "emergency_vehicle"
                    if emergency
                    else label,

                "confidence": conf,

                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "position": {
                    "x": cx,
                    "y": cy,
                },

                "speed": speed,

                "direction": direction,

                "trajectory":
                    list(traj[-5:]),

                "tracking_status":
                    "active",

                "emergency":
                    emergency,
            }

            frame_detections.append(
                detection
            )

        # ----------------------------------------------------
        # Store frame
        # ----------------------------------------------------

        all_frames.append(
            {
                "frame": frame_count,
                "detections": frame_detections
            }
        )

        # ----------------------------------------------------
        # Console output every 30 frames
        # ----------------------------------------------------

        if frame_count % 30 == 0:

            print(
                f"\nFrame {frame_count} — "
                f"{len(frame_detections)} objects"
                +
                (
                    f" | {len(emergency_boxes)} EMERGENCY"
                    if emergency_boxes
                    else ""
                )
            )

            for d in frame_detections:

                tag = (
                    " *** EMERGENCY ***"
                    if d["emergency"]
                    else ""
                )

                print(
                    f"  ID={d['vehicle_id']:4d} | "
                    f"{d['type']:20s} | "
                    f"conf={d['confidence']:.2f} | "
                    f"pos=("
                    f"{d['position']['x']},"
                    f"{d['position']['y']}"
                    f") | "
                    f"dir={d['direction']:7s} | "
                    f"speed={d['speed']} km/h"
                    f"{tag}"
                )

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        JSON_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_frames,
            f,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\nTotal frames: "
        f"{frame_count}"
    )

    print(
        f"Unique vehicles tracked: "
        f"{len(trajectories)}"
    )

    print(
        f"JSON saved to: "
        f"{JSON_OUTPUT_PATH}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 3:

        print("Usage:")

        print(
            "  Image: "
            "python unified_pipeline.py image <path>"
        )

        print(
            "  Video: "
            "python unified_pipeline.py video <path>"
        )

        sys.exit(1)

    mode = sys.argv[1].lower()

    path = sys.argv[2]

    if not os.path.exists(path):

        print(
            f"ERROR: File not found: {path}"
        )

        sys.exit(1)

    if mode not in (
        "image",
        "video"
    ):

        print(
            "ERROR: mode must be "
            "'image' or 'video'"
        )

        sys.exit(1)

    run_pipeline(
        path,
        mode
    )