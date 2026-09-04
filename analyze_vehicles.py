"""
Phase 4 - Extract vehicle ID, type, confidence, position,
          direction, trajectory, and approximate speed.
"""

from ultralytics import YOLO
from collections import defaultdict
import sys
import os

MODEL_PATH = r"E:\Traffic_AI\models\yolo11n.pt"
OUTPUT_DIR = r"E:\Traffic_AI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAFFIC_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# How many past positions to keep per vehicle
TRAJECTORY_LEN = 30

# Pixels-per-meter estimate (tune based on your camera/scene)
# For a typical road scene, ~10-20 pixels = 1 meter
PIXELS_PER_METER = 15
FPS = 30  # assumed video FPS


def get_direction(positions: list) -> str:
    """Determine movement direction from trajectory."""
    if len(positions) < 2:
        return "unknown"
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def get_speed_kmh(positions: list, fps: int, ppm: float) -> float:
    """Estimate speed in km/h from pixel displacement."""
    if len(positions) < 2:
        return 0.0
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    frames = len(positions) - 1
    meters_per_sec = (pixel_dist / ppm) / (frames / fps)
    return round(meters_per_sec * 3.6, 1)


def analyze_video(video_path: str):
    print(f"\n--- Phase 4 Vehicle Analysis: {video_path} ---")
    model = YOLO(MODEL_PATH)

    # Store trajectory per vehicle ID
    trajectories = defaultdict(list)

    results = model.track(
        source=video_path,
        conf=0.25,
        device=0,
        classes=list(TRAFFIC_CLASSES.keys()),
        tracker="bytetrack.yaml",
        persist=True,
        save=True,
        project=OUTPUT_DIR,
        name="analysis",
        stream=True,
    )

    frame_count = 0
    for r in results:
        frame_count += 1

        if r.boxes.id is None:
            continue

        for box in r.boxes:
            track_id = int(box.id)
            cls_id = int(box.cls)
            label = TRAFFIC_CLASSES.get(cls_id, "unknown")
            conf = round(float(box.conf), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update trajectory
            trajectories[track_id].append((cx, cy))
            if len(trajectories[track_id]) > TRAJECTORY_LEN:
                trajectories[track_id].pop(0)

            # Print every 30 frames
            if frame_count % 30 == 0:
                traj = trajectories[track_id]
                direction = get_direction(traj)
                speed = get_speed_kmh(traj, FPS, PIXELS_PER_METER)

                print(
                    f"  ID={track_id:4d} | {label:12s} | conf={conf:.2f} | "
                    f"pos=({cx},{cy}) | dir={direction:7s} | speed={speed} km/h | "
                    f"box=[{x1},{y1},{x2},{y2}]"
                )

    print(f"\nTotal frames: {frame_count}")
    print(f"Unique vehicles tracked: {len(trajectories)}")
    print(f"Output saved to: {OUTPUT_DIR}\\analysis")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_vehicles.py <path_to_video>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    analyze_video(path)
