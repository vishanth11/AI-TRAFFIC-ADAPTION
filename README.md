# AI Traffic Adaptation

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/vishanth11/AI-TRAFFIC-ADAPTION.git
cd AI-TRAFFIC-ADAPTION
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup models
```bash
python setup_models.py
```
This will auto-download `yolo11n.pt` into the `models/` folder.

> **Important:** `emergency.pt` is a custom trained model and cannot be auto-downloaded.
> Ask **Member 2 (Boomika)** to share `emergency.pt` and place it manually in the `models/` folder.

### 4. Run the pipeline
```bash
# Image
python unified_pipeline.py image <path_to_image>

# Video
python unified_pipeline.py video <path_to_video>
```

## Project Structure
```
Traffic_AI/
├── models/               # Place model files here (not tracked by Git)
│   ├── yolo11n.pt        # Auto-downloaded via setup_models.py
│   └── emergency.pt      # Get from Member 2 (Boomika)
├── inputs/               # Input images/videos
├── outputs/              # Detection results
├── unified_pipeline.py   # Main pipeline
├── detect_traffic.py     # Traffic detection
├── detect_emergency.py   # Emergency vehicle detection
├── track_vehicles.py     # Vehicle tracking
├── analyze_vehicles.py   # Vehicle analysis
├── api.py                # API
└── setup_models.py       # Run once to setup models
```
