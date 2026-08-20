"""
=========================================================
YOLO World Detector

Dronions AI

This module is responsible ONLY for object detection.

Responsibilities

✔ Load model

✔ Set target

✔ Run inference

✔ Convert detections to DetectionCandidate

NO filtering

NO tracking

NO Gemini

NO speech
=========================================================
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    IMAGE_SIZE,
)

from perception.loader import load_model

from perception.candidate import DetectionCandidate

from utils.prompts import get_prompts, get_negative_prompts


class YOLOWorldDetector:

    """
    Main perception module.

    This class wraps YOLO-World and converts every
    detection into DetectionCandidate objects.
    """

    def __init__(self):

        self.model = load_model()

        self.target = ""

        self.prompts = []

        self.negative_prompts = []

    # --------------------------------------------------

    def set_target(self, target: str):

        self.target = target.lower().strip()

        self.prompts = get_prompts(self.target)

        # Negative prompts are handed to the model as ordinary classes so that
        # background structures have a class of their own to land in, then
        # dropped in _parse_result. Without them an open-vocabulary detector
        # has to assign every box its best-matching *positive* class, which is
        # how a 3 m wall came back labelled "cardboard box". Measured on one
        # frame: the real box went from 0.359 to 0.566 confidence with these
        # in place.
        self.negative_prompts = get_negative_prompts(self.target)

        self.model.set_classes(self.prompts + self.negative_prompts)

    # --------------------------------------------------

    def set_target_classes(self, classes):
        """Detect an arbitrary class list, outside the target machinery.

        The surface scan asks a different question from the search -- "what
        furniture is in view" rather than "is my mug here" -- so it wants its
        own classes without a target's prompt expansion or its negatives. The
        caller restores the target afterwards with set_target.
        """
        self.target = ""
        self.prompts = list(classes)
        self.negative_prompts = []
        self.model.set_classes(self.prompts)

    # --------------------------------------------------

    def get_target(self):

        return self.target

    # --------------------------------------------------

    def detect(self, frame: np.ndarray):

        """
        Main detection function.

        Returns

        -------

        List[DetectionCandidate]
        """

        if frame is None:

            return []

        results = self.model.predict(

            source=frame,

            conf=CONFIDENCE_THRESHOLD,

            imgsz=IMAGE_SIZE,

            verbose=False,

            device=DEVICE,

        )

        candidates = []

        for result in results:
            candidates.extend(
                self._parse_result(result, frame)
            )

        return candidates

    # --------------------------------------------------

    def _parse_result(self, result, frame: np.ndarray) -> List[DetectionCandidate]:
        """
        Parses YOLO-World result into DetectionCandidate objects.
        """
        candidates = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return candidates

        h, w = frame.shape[:2]
        img_area = h * w

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            # Label might be mapped from prompt list
            all_classes = self.prompts + self.negative_prompts
            label = all_classes[cls_id] if cls_id < len(all_classes) else str(cls_id)

            # Detection landed in a negative class -- background, not target.
            if label in self.negative_prompts:
                continue

            cand = DetectionCandidate(
                label=label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                prompt=self.target
            )
            
            # Basic geometry population
            cand.image_width = w
            cand.image_height = h
            cand.width = x2 - x1
            cand.height = y2 - y1
            cand.pixel_area = cand.width * cand.height
            cand.relative_area = cand.pixel_area / img_area if img_area > 0 else 0
            cand.aspect_ratio = cand.width / max(1, cand.height)
            cand.center = (int(x1 + cand.width/2), int(y1 + cand.height/2))
            cand.normalized_center = (cand.center[0]/w, cand.center[1]/h)
            
            # Calculate distance from image center
            cx, cy = cand.normalized_center
            cand.distance_to_center = ((cx - 0.5)**2 + (cy - 0.5)**2)**0.5
            
            candidates.append(cand)
            
        return candidates