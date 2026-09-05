"""
Phase 5 - Emergency vehicle detection.
Uses emergency.pt to flag emergency vs non-emergency vehicles.
"""

from ultralytics import YOLO
import sys
import os

EMERGENCY_MODEL_PATH = r"E:\Traffic_AI\models\emergency.pt"
OUTPUT_DIR = r"E:\Traffic_AI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMERGENCY_CLASS_ID = 0  # 'emergency_vehicle'


def detect_emergency(source: str, mode: str):
    print(f"\n--- Phase 5 Emergency Vehicle Detection: {source} ---")
    model = YOLO(EMERGENCY_MODEL_PATH)

    is_video = mode == "video"
    results = model.predict(
        source=source,
        conf=0.30,
        device=0,
        save=True,
        project=OUTPUT_DIR,
        name=f"{mode}_emergency",
        stream=is_video,
    )

    frame_count = 0
    emergency_count = 0

    for r in results:
        frame_count += 1
        frame_emergency = []

        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = round(float(box.conf), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls_id == EMERGENCY_CLASS_ID:
                emergency_count += 1
                frame_emergency.append((label, conf, cx, cy, x1, y1, x2, y2))

        if not is_video and frame_emergency:
            print(f"EMERGENCY VEHICLES DETECTED: {len(frame_emergency)}")
            for label, conf, cx, cy, x1, y1, x2, y2 in frame_emergency:
                print(f"  {label:20s} conf={conf:.2f}  center=({cx},{cy})  box=[{x1},{y1},{x2},{y2}]")
        elif not is_video:
            print("No emergency vehicles detected.")

        if is_video and frame_count % 30 == 0:
            if frame_emergency:
                print(f"  Frame {frame_count}: {len(frame_emergency)} EMERGENCY vehicle(s) detected!")
                for label, conf, cx, cy, x1, y1, x2, y2 in frame_emergency:
                    print(f"    conf={conf:.2f}  center=({cx},{cy})")
            else:
                print(f"  Frame {frame_count}: no emergency vehicles")

    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total emergency detections: {emergency_count}")
    print(f"Output saved to: {OUTPUT_DIR}\\{mode}_emergency")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Image: python detect_emergency.py image <path>")
        print("  Video: python detect_emergency.py video <path>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    path = sys.argv[2]

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    if mode not in ("image", "video"):
        print("ERROR: mode must be 'image' or 'video'")
        sys.exit(1)

    detect_emergency(path, mode)
