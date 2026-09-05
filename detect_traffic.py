"""
Phase 2 - Detect traffic-relevant objects only.
Classes: car, bus, truck, motorcycle, person
"""

from ultralytics import YOLO
import sys
import os

MODEL_PATH = r"E:\Traffic_AI\models\yolo11n.pt"
OUTPUT_DIR = r"E:\Traffic_AI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO class IDs for traffic-relevant objects
TRAFFIC_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect_traffic(source: str, mode: str):
    print(f"\n--- Phase 2 Traffic Detection: {source} ---")
    model = YOLO(MODEL_PATH)

    is_video = mode == "video"
    results = model.predict(
        source=source,
        conf=0.25,
        device=0,
        classes=list(TRAFFIC_CLASSES.keys()),   # filter classes here
        save=True,
        project=OUTPUT_DIR,
        name=f"{mode}_traffic",
        stream=is_video,
    )

    frame_count = 0
    for r in results:
        frame_count += 1
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls)
            label = TRAFFIC_CLASSES[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((label, conf, x1, y1, x2, y2))

        if not is_video:
            print(f"Detected {len(detections)} traffic objects:")
            for label, conf, x1, y1, x2, y2 in detections:
                print(f"  {label:12s} conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")
        elif frame_count % 30 == 0:
            print(f"  Frame {frame_count}: {len(detections)} detections")

    if is_video:
        print(f"Total frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_DIR}\\{mode}_traffic")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Image: python detect_traffic.py image <path>")
        print("  Video: python detect_traffic.py video <path>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    path = sys.argv[2]

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    if mode not in ("image", "video"):
        print("ERROR: mode must be 'image' or 'video'")
        sys.exit(1)

    detect_traffic(path, mode)
