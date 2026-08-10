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
CONFIDENCE_THRESHOLD = 0.15
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The free-tier limit is 20 requests/day *per project per model*
# (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier), so issuing a new
# API key inside the same Google Cloud project does not reset it -- but each
# model has its own separate daily allowance. Switching this is the quickest
# way to keep testing once one model is exhausted.
#
# Moved off gemini-2.5-flash: it has entered retirement and answers a freshly
# issued API key with 404 "no longer available to new users". A whole test
# flight was lost to that, every vision call failing while the drone went on
# searching. Existing keys still reach it, so pin GEMINI_MODEL=gemini-2.5-flash
# in .env to reproduce the measurements in docs/, which were taken with it.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
VOICE_ENABLED = False

# --- User pose, for describing where things are ---
# The assistive answer is "your keys are two metres ahead of you, slightly
# right" -- which is only computable if the system knows where the user is
# standing and which way they face. In simulation this is a known constant; on
# real hardware it would come from the wearable dock.
# ENU metres, and the facing angle measured from +x, counter-clockwise.
USER_POSITION = (0.0, 0.0)
USER_YAW = 0.0
