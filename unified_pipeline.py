"""
Phase 6 - Unified traffic perception pipeline.

Combines:
    - yolo11n.pt     -> vehicle detection + ByteTrack tracking
    - emergency.pt   -> emergency vehicle detection

Outputs:
    - Structured JSON per frame
    - Console progress information

Device:
    - CUDA GPU if available
    - CPU otherwise
"""

from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json
import sys
import os
import torch


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
# DEVICE SELECTION
# ============================================================

def get_device():
    """
    Automatically select the best available device.

    Returns:
        0     -> NVIDIA CUDA GPU
        "cpu" -> CPU
    """

    if torch.cuda.is_available():

        print("[DEVICE] CUDA GPU detected.")

        print(
            f"[DEVICE] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return 0

    print("[DEVICE] CUDA not available.")

    print("[DEVICE] Using CPU.")

    return "cpu"


# ============================================================
# DIRECTION
# ============================================================

def get_direction(positions: list) -> str:

    if len(positions) < 2:

        return "unknown"

    dx = (
        positions[-1][0]
        - positions[0][0]
    )

    dy = (
        positions[-1][1]
        - positions[0][1]
    )

    if abs(dx) >= abs(dy):

        return (
            "right"
            if dx > 0
            else "left"
        )

    return (
        "down"
        if dy > 0
        else "up"
    )


# ============================================================
# SPEED
# ============================================================

def get_speed_kmh(positions: list) -> float:

    if len(positions) < 2:

        return 0.0

    dx = (
        positions[-1][0]
        - positions[0][0]
    )

    dy = (
        positions[-1][1]
        - positions[0][1]
    )

    pixel_dist = (
        (dx ** 2 + dy ** 2)
        ** 0.5
    )

    frames = len(positions) - 1

    if frames <= 0:

        return 0.0

    mps = (
        pixel_dist
        / PIXELS_PER_METER
    ) / (
        frames / FPS
    )

    return round(
        mps * 3.6,
        1
    )


# ============================================================
# IOU
# ============================================================

def iou(box1, box2) -> float:

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    inter = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    if inter == 0:

        return 0.0

    area1 = (
        (box1[2] - box1[0])
        *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        *
        (box2[3] - box2[1])
    )

    denominator = (
        area1
        + area2
        - inter
    )

    if denominator <= 0:

        return 0.0

    return inter / denominator


# ============================================================
# EMERGENCY MATCHING
# ============================================================

def is_emergency(
    box,
    emergency_boxes,
    threshold=0.1
) -> bool:

    for ebox in emergency_boxes:

        if iou(
            box,
            ebox
        ) >= threshold:

            return True

    return False


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    source: str,
    mode: str
):

    print(
        f"\n--- Phase 6 Unified Pipeline: "
        f"{source} ---"
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Verify model files
    # --------------------------------------------------------

    if not GENERAL_MODEL_PATH.exists():

        print(
            "ERROR: YOLO model not found:"
        )

        print(
            f"       {GENERAL_MODEL_PATH}"
        )

        print(
            "Run: python setup_models.py"
        )

        sys.exit(1)

    if not EMERGENCY_MODEL_PATH.exists():

        print(
            "ERROR: Emergency model not found:"
        )

        print(
            f"       {EMERGENCY_MODEL_PATH}"
        )

        print(
            "Run: python setup_models.py"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "[MODEL] Loading YOLO11n:"
    )

    print(
        f"        {GENERAL_MODEL_PATH}"
    )

    general_model = YOLO(
        str(GENERAL_MODEL_PATH)
    )

    print(
        "[MODEL] Loading emergency detector:"
    )

    print(
        f"        {EMERGENCY_MODEL_PATH}"
    )

    emergency_model = YOLO(
        str(EMERGENCY_MODEL_PATH)
    )

    print(
        "[MODEL] Both models loaded successfully."
    )

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    print(
        f"[DEVICE] Selected device: "
        f"{device}"
    )

    # --------------------------------------------------------
    # Video/image mode
    # --------------------------------------------------------

    is_video = mode == "video"

    # --------------------------------------------------------
    # Tracking history
    # --------------------------------------------------------

    trajectories = defaultdict(list)

    # --------------------------------------------------------
    # Store all frame outputs
    # --------------------------------------------------------

    all_frames = []

    # ========================================================
    # GENERAL VEHICLE TRACKING
    # ========================================================

    print(
        "\n[PIPELINE] Starting vehicle tracking..."
    )

    general_results = general_model.track(

        source=source,

        conf=0.25,

        device=device,

        classes=list(
            TRAFFIC_CLASSES.keys()
        ),

        tracker="bytetrack.yaml",

        persist=True,

        stream=is_video,
    )

    # ========================================================
    # EMERGENCY DETECTION
    # ========================================================

    print(
        "[PIPELINE] Starting emergency detection..."
    )

    emergency_results = emergency_model.predict(

        source=source,

        conf=0.30,

        device=device,

        stream=is_video,
    )

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    frame_count = 0

    for (
        g_result,
        e_result
    ) in zip(
        general_results,
        emergency_results
    ):

        frame_count += 1

        # ----------------------------------------------------
        # Emergency boxes
        # ----------------------------------------------------

        emergency_boxes = []

        if e_result.boxes is not None:

            for box in e_result.boxes:

                cls_id = int(
                    box.cls
                )

                if cls_id == 0:

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

        # ----------------------------------------------------
        # No tracked objects
        # ----------------------------------------------------

        if (
            g_result.boxes is None
            or
            g_result.boxes.id is None
        ):

            all_frames.append(
                {
                    "frame": frame_count,
                    "detections": []
                }
            )

            continue

        # ====================================================
        # VEHICLE DETECTIONS
        # ====================================================

        for box in g_result.boxes:

            # ------------------------------------------------
            # Track ID
            # ------------------------------------------------

            track_id = int(
                box.id
            )

            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            cls_id = int(
                box.cls
            )

            label = TRAFFIC_CLASSES.get(
                cls_id,
                "unknown"
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            conf = round(
                float(box.conf),
                2
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ------------------------------------------------
            # Center position
            # ------------------------------------------------

            cx = (
                x1 + x2
            ) // 2

            cy = (
                y1 + y2
            ) // 2

            # ------------------------------------------------
            # Update trajectory
            # ------------------------------------------------

            trajectories[
                track_id
            ].append(
                (cx, cy)
            )

            # Keep only last N positions

            if len(
                trajectories[track_id]
            ) > TRAJECTORY_LEN:

                trajectories[
                    track_id
                ].pop(0)

            traj = trajectories[
                track_id
            ]

            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            direction = get_direction(
                traj
            )

            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            speed = get_speed_kmh(
                traj
            )

            # ------------------------------------------------
            # Emergency status
            # ------------------------------------------------

            emergency = is_emergency(

                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                emergency_boxes
            )

            # ------------------------------------------------
            # Structured detection
            # ------------------------------------------------

            detection = {

                "vehicle_id":
                    track_id,

                "type":
                    (
                        "emergency_vehicle"
                        if emergency
                        else label
                    ),

                "confidence":
                    conf,

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

                "speed":
                    speed,

                "direction":
                    direction,

                "trajectory":
                    list(
                        traj[-5:]
                    ),

                "tracking_status":
                    "active",

                "emergency":
                    emergency,
            }

            frame_detections.append(
                detection
            )

        # ====================================================
        # STORE FRAME
        # ====================================================

        all_frames.append(
            {
                "frame":
                    frame_count,

                "detections":
                    frame_detections
            }
        )

        # ====================================================
        # CONSOLE OUTPUT
        # ====================================================

        if frame_count % 30 == 0:

            print(

                f"\nFrame {frame_count} — "

                f"{len(frame_detections)} objects"

                +

                (

                    f" | "
                    f"{len(emergency_boxes)} "
                    f"EMERGENCY"

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

                    f"  ID="
                    f"{d['vehicle_id']:4d} | "

                    f"{d['type']:20s} | "

                    f"conf="
                    f"{d['confidence']:.2f} | "

                    f"pos=("

                    f"{d['position']['x']},"

                    f"{d['position']['y']}"

                    f") | "

                    f"dir="
                    f"{d['direction']:7s} | "

                    f"speed="
                    f"{d['speed']} km/h"

                    f"{tag}"
                )

    # ========================================================
    # SAVE JSON
    # ========================================================

    print(
        "\n[OUTPUT] Saving perception JSON..."
    )

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

    unique_vehicles = len(
        trajectories
    )

    total_detections = sum(
        len(frame["detections"])
        for frame in all_frames
    )

    emergency_detections = sum(

        sum(
            1
            for detection
            in frame["detections"]
            if detection.get(
                "emergency",
                False
            )
        )

        for frame in all_frames
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PHASE 6 PIPELINE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Total frames: "
        f"{frame_count}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    print(
        f"Unique vehicles tracked: "
        f"{unique_vehicles}"
    )

    print(
        f"Emergency detections: "
        f"{emergency_detections}"
    )

    print(
        f"JSON saved to: "
        f"{JSON_OUTPUT_PATH}"
    )

    print(
        "=" * 60
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    mode = sys.argv[1].lower()

    # --------------------------------------------------------
    # Input path
    # --------------------------------------------------------

    path = sys.argv[2]

    # --------------------------------------------------------
    # Validate input file
    # --------------------------------------------------------

    if not os.path.exists(path):

        print(
            f"ERROR: File not found: "
            f"{path}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate mode
    # --------------------------------------------------------

    if mode not in (
        "image",
        "video"
    ):

        print(
            "ERROR: mode must be "
            "'image' or 'video'"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    run_pipeline(
        path,
        mode
    )