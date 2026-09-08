"""
Run this once after cloning the repo to download required models.
Usage: python setup_models.py
"""

import os
import shutil
import gdown
from ultralytics import YOLO

os.makedirs("models", exist_ok=True)

# Download yolo11n.pt
print("Downloading yolo11n.pt...")
YOLO("yolo11n.pt")
if os.path.exists("yolo11n.pt"):
    shutil.move("yolo11n.pt", "models/yolo11n.pt")
print("yolo11n.pt ready at models/yolo11n.pt")

# Download emergency.pt from Google Drive
EMERGENCY_GDRIVE_URL = "https://drive.google.com/uc?id=18K2Vfv3ADBPbxU-XOMjr3ld2lX9sP6ti"
EMERGENCY_PATH = "models/emergency.pt"

if not os.path.exists(EMERGENCY_PATH):
    print("\nDownloading emergency.pt from Google Drive...")
    gdown.download(EMERGENCY_GDRIVE_URL, EMERGENCY_PATH, quiet=False)
    print("emergency.pt ready at models/emergency.pt")
else:
    print("emergency.pt already exists.")
