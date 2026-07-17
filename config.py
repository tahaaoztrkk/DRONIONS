"""
=========================================================
DRONIONS AI
Configuration File
---------------------------------------------------------
Author : Taha Ozturk
Project: Dronions AI - Personal Drone Assistant
=========================================================
"""

from pathlib import Path
import torch

# =========================================================
# PROJECT
# =========================================================

PROJECT_NAME = "Dronions AI"

ROOT_DIR = Path(__file__).resolve().parent

MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
LOG_DIR = ROOT_DIR / "logs"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# =========================================================
# CAMERA
# =========================================================

# Laptop Webcam
# CAMERA_SOURCE = 0

# Android IP Webcam
CAMERA_SOURCE = "http://10.116.143.7:8080/video"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# YOLO-WORLD
# =========================================================

YOLO_MODEL = "yolov8x-worldv2.pt"

IMAGE_SIZE = 1280

CONFIDENCE_THRESHOLD = 0.02

IOU_THRESHOLD = 0.45

MAX_DETECTIONS = 20

# =========================================================
# TRACKING
# =========================================================

TRACK_BUFFER = 30

TRACK_FPS = 30

# =========================================================
# GEOMETRY FILTER
# =========================================================

MIN_BOX_AREA_RATIO = 0.0002

MAX_BOX_AREA_RATIO = 0.18

MAX_ASPECT_RATIO = 2.5

MIN_ASPECT_RATIO = 0.45

# =========================================================
# SCORING
# =========================================================

YOLO_WEIGHT = 0.40

GEOMETRY_WEIGHT = 0.20

CENTER_WEIGHT = 0.20

SEMANTIC_WEIGHT = 0.20

# =========================================================
# NAVIGATION
# =========================================================

CENTER_TOLERANCE = 60

LEFT_THRESHOLD = 250

RIGHT_THRESHOLD = 250

UP_THRESHOLD = 120

DOWN_THRESHOLD = 120

# =========================================================
# VERIFICATION
# =========================================================

USE_SIGLIP = False

USE_GEMINI = False

GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_API_KEY = ""

VERIFY_MARGIN = 0.10

# =========================================================
# SPEECH
# =========================================================

VOICE_ENABLED = True

VOICE_INTERVAL = 1.2

VOICE_RATE = 180

# =========================================================
# DEBUG
# =========================================================

SHOW_ALL_BOXES = True

SHOW_CENTER = True

SHOW_FPS = True

SHOW_TRACK_IDS = True

SAVE_LOGS = True