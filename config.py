"""
Görevi: Bütün ayarlar burada tutulur.
"""
import os
from dotenv import load_dotenv

load_dotenv()

YOLO_MODEL = "yolov8x-worldv2.pt"
DEVICE = "cuda"
# Inference resolution. The camera publishes 1280x960, so at 640 every object is
# halved before YOLO ever sees it -- which looks like an obvious thing to fix
# and is not. Measured on the same sweep (docs/PLAN_EVERYDAY_SCENARIOS.md 5f),
# 1280 *halves* close-range recall, 8/9 to 4/9 at 2 m, and gains nothing far
# out: the model is trained at 640 and running it wider moves objects outside
# the scale distribution it expects. Kept overridable so this stays checkable
# rather than remembered, but 640 is the measured optimum, not a default nobody
# revisited.
IMAGE_SIZE = int(os.getenv("DRONIONS_IMAGE_SIZE", "640"))
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
VOICE_ENABLED = True

# --- User pose, for describing where things are ---
# The assistive answer is "your keys are two metres ahead of you, slightly
# right" -- which is only computable if the system knows where the user is
# standing and which way they face. In simulation this is a known constant; on
# real hardware it would come from the wearable dock.
# ENU metres, and the facing angle measured from +x, counter-clockwise.
USER_POSITION = (0.0, 0.0)
USER_YAW = 0.0
