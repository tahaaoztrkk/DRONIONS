"""
=========================================================
YOLO Loader
=========================================================
"""

from ultralytics import YOLO

from config import YOLO_MODEL, DEVICE


def load_model():

    model = YOLO(YOLO_MODEL)

    model.to(DEVICE)

    return model