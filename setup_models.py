"""
Run this once after cloning the repo to download required models.
Usage: python setup_models.py
"""

import os
import shutil
from ultralytics import YOLO

os.makedirs("models", exist_ok=True)

# Download yolo11n.pt
print("Downloading yolo11n.pt...")
YOLO("yolo11n.pt")
if os.path.exists("yolo11n.pt"):
    shutil.move("yolo11n.pt", "models/yolo11n.pt")
print("yolo11n.pt ready at models/yolo11n.pt")

# Check emergency.pt
if os.path.exists("models/emergency.pt"):
    print("emergency.pt found.")
else:
    print("\nWARNING: models/emergency.pt is missing!")
    print("Ask Member 2 (Boomika) to share emergency.pt and place it in the models/ folder.")
