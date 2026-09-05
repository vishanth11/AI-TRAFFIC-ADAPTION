"""
Phase 3 - ByteTrack multi-object tracking with persistent IDs.
Tracks: car, bus, truck, motorcycle, person
"""

from ultralytics import YOLO
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


def track_video(video_path: str):
    print(f"\n--- Phase 3 ByteTrack Tracking: {video_path} ---")
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=video_path,
        conf=0.25,
        device=0,
        classes=list(TRAFFIC_CLASSES.keys()),
        tracker="bytetrack.yaml",       # ByteTrack algorithm
        persist=True,                   # persist IDs across frames
        save=True,
        project=OUTPUT_DIR,
        name="tracking",
        stream=True,
    )

    frame_count = 0
    for r in results:
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"\nFrame {frame_count}:")
            if r.boxes.id is not None:
                for i, box in enumerate(r.boxes):
                    track_id = int(box.id) if box.id is not None else -1
                    cls_id = int(box.cls)
                    label = TRAFFIC_CLASSES.get(cls_id, "unknown")
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    print(f"  ID={track_id:3d}  {label:12s}  conf={conf:.2f}  center=({cx},{cy})")
            else:
                print("  No tracked objects this frame")

    print(f"\nTotal frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_DIR}\\tracking")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_vehicles.py <path_to_video>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    track_video(path)
