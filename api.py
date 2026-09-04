"""
Phase 7 - REST API for traffic perception pipeline.
Endpoints:
  POST /analyze/image  - upload image, get detections
  POST /analyze/video  - upload video, get full JSON output
  GET  /results        - get latest perception output
  GET  /health         - check API status
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import json
import os
from collections import defaultdict
from ultralytics import YOLO

# ── paths ──────────────────────────────────────────────────────────────────
GENERAL_MODEL_PATH   = r"E:\Traffic_AI\models\yolo11n.pt"
EMERGENCY_MODEL_PATH = r"E:\Traffic_AI\models\emergency.pt"
UPLOAD_DIR           = r"E:\Traffic_AI\uploads"
OUTPUT_DIR           = r"E:\Traffic_AI\outputs"
JSON_OUTPUT_PATH     = r"E:\Traffic_AI\outputs\perception_output.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
TRAFFIC_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAJECTORY_LEN   = 30
PIXELS_PER_METER = 15
FPS              = 30

# ── load models once at startup ────────────────────────────────────────────
general_model   = YOLO(GENERAL_MODEL_PATH)
emergency_model = YOLO(EMERGENCY_MODEL_PATH)

app = FastAPI(
    title="Traffic Perception API",
    description="Vehicle detection, tracking, and emergency vehicle identification.",
    version="1.0.0",
)


# ── helpers ────────────────────────────────────────────────────────────────
def get_direction(positions: list) -> str:
    if len(positions) < 2:
        return "unknown"
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def get_speed_kmh(positions: list) -> float:
    if len(positions) < 2:
        return 0.0
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    frames = len(positions) - 1
    mps = (pixel_dist / PIXELS_PER_METER) / (frames / FPS)
    return round(mps * 3.6, 1)


def iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def is_emergency(box, emergency_boxes, threshold=0.1) -> bool:
    return any(iou(box, eb) >= threshold for eb in emergency_boxes)


def process_source(source: str, is_video: bool) -> list:
    """Run both models on source and return list of frame detections."""
    trajectories = defaultdict(list)
    all_frames   = []

    general_results = general_model.track(
        source=source,
        conf=0.25,
        device=0,
        classes=list(TRAFFIC_CLASSES.keys()),
        tracker="bytetrack.yaml",
        persist=True,
        stream=is_video,
    )
    emergency_results = emergency_model.predict(
        source=source,
        conf=0.30,
        device=0,
        stream=is_video,
    )

    for frame_idx, (g_result, e_result) in enumerate(zip(general_results, emergency_results), 1):
        emergency_boxes = [
            list(map(int, box.xyxy[0]))
            for box in e_result.boxes
            if int(box.cls) == 0
        ]

        frame_detections = []
        if g_result.boxes.id is not None:
            for box in g_result.boxes:
                track_id = int(box.id)
                cls_id   = int(box.cls)
                label    = TRAFFIC_CLASSES.get(cls_id, "unknown")
                conf     = round(float(box.conf), 2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2

                trajectories[track_id].append((cx, cy))
                if len(trajectories[track_id]) > TRAJECTORY_LEN:
                    trajectories[track_id].pop(0)

                traj      = trajectories[track_id]
                emergency = is_emergency([x1, y1, x2, y2], emergency_boxes)

                frame_detections.append({
                    "vehicle_id":      track_id,
                    "type":            "emergency_vehicle" if emergency else label,
                    "confidence":      conf,
                    "bounding_box":    {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "position":        {"x": cx, "y": cy},
                    "speed":           get_speed_kmh(traj),
                    "direction":       get_direction(traj),
                    "trajectory":      list(traj[-5:]),
                    "tracking_status": "active",
                    "emergency":       emergency,
                })

        all_frames.append({"frame": frame_idx, "detections": frame_detections})

    # persist latest output
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(all_frames, f, indent=2)

    return all_frames


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/results")
def get_results():
    if not os.path.exists(JSON_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="No results yet. Run /analyze/image or /analyze/video first.")
    with open(JSON_OUTPUT_PATH) as f:
        return JSONResponse(content=json.load(f))


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only jpg/jpeg/png images supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frames = process_source(save_path, is_video=False)
    detections = frames[0]["detections"] if frames else []

    return {
        "filename":   file.filename,
        "total_detections": len(detections),
        "detections": detections,
    }


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Only mp4/avi/mov videos supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frames = process_source(save_path, is_video=True)
    total_detections = sum(len(f["detections"]) for f in frames)

    return {
        "filename":         file.filename,
        "total_frames":     len(frames),
        "total_detections": total_detections,
        "frames":           frames,
    }


# ── run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
