"""
Görevi: Bütün ayarlar burada tutulur.
"""
import os
from dotenv import load_dotenv

load_dotenv()

YOLO_MODEL = "yolov8x-worldv2.pt"
DEVICE = "cuda"
IMAGE_SIZE = 640
CAMERA_SOURCE = "http://10.116.143.7:8080/video"
CONFIDENCE_THRESHOLD = 0.5
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VOICE_ENABLED = False  # Disabled for now based on user feedback
