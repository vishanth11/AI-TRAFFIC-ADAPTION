"""
Phase 1 - Test YOLO on a single image or video.
Model: yolo11n.pt (auto-downloaded to E:\Traffic_AI\models)
"""

from ultralytics import YOLO
import sys
import os

MODEL_PATH = r"E:\Traffic_AI\models\yolo11n.pt"
OUTPUT_DIR = r"E:\Traffic_AI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_image(image_path: str):
    print(f"\n--- Testing YOLO on image: {image_path} ---")
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=image_path,
        conf=0.25,
        device=0,
        save=True,
        project=OUTPUT_DIR,
        name="image_test",
    )
    for r in results:
        print(f"Detected {len(r.boxes)} objects")
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {label:15s} conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")
    print(f"\nOutput saved to: {OUTPUT_DIR}\\image_test")


def test_video(video_path: str):
    print(f"\n--- Testing YOLO on video: {video_path} ---")
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=video_path,
        conf=0.25,
        device=0,
        save=True,
        project=OUTPUT_DIR,
        name="video_test",
        stream=True,
    )
    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}: {len(r.boxes)} detections")
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_DIR}\\video_test")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Image test:  python test_yolo.py image <path_to_image>")
        print("  Video test:  python test_yolo.py video <path_to_video>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    path = sys.argv[2]

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    if mode == "image":
        test_image(path)
    elif mode == "video":
        test_video(path)
    else:
        print("ERROR: mode must be 'image' or 'video'")
        sys.exit(1)
