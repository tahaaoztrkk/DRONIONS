"""
Görevi: Açık sözlüklü ve göreve özel dedektörleri birlikte çalıştırır.

Measured at the viewpoints the sweep flies from, the open-vocabulary detector
finds the phone in 1 viewpoint of 8, the book 2 of 8, the mug 3 of 8 -- 47%
overall. A small model trained on the same five objects reaches 97%. That gap
is where every unexplained failure this week came from: the size gate, the
colour gate and the model choosing between crops all only ever see what the
detector found first, so none of the work on them could move it.

Replacing the open model would cost the thing the system is for. A user saying
"find my charger" has to get an attempt, not "unknown class" -- so the open
model stays and answers everything, and the trained one is consulted only for
the objects it has actually been shown. What comes out is the union.

The confidences are comparable enough to share one ranking, which was measured
rather than assumed: across correct detections the open model sits at a median
of 0.74 and the trained one at 0.87, a ratio of 1.18. Had they been far apart,
one would have won every ranking for reasons unconnected to being right.

Worth knowing about the trained half. It learned one room of Gazebo meshes under
one light, so outside that room it is confident and unreliable, and it is
therefore off unless its weights are present and the target is one of its own
classes. Its confidence was measured on detections that were correct; what it
does on the ones that are not is not known.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from perception.candidate import DetectionCandidate
from perception.detector import YOLOWorldDetector

# How much of the smaller box has to sit inside the larger before the two are
# taken to be about the same thing in the picture.
#
# Overlap on its own cannot say whether that thing is the target: measured, a
# world box and a trained box of the same object share between 0.80 and 1.00 of
# the smaller, and a world box sitting on the *book* while hunting the phone
# shares up to 0.87 with the trained model's book. The populations overlap, so
# geometry alone would either leave duplicates or merge a phone with a book.
#
# What separates them is the class. Where the trained model has an opinion it
# is the authority for its own classes -- 97% against 47% -- so a world box
# landing on something it has identified is resolved by that identification: the
# same class is a duplicate, a different class is the open model calling a book
# a phone, which is the confusion this whole thread has been about. World boxes
# overlapping nothing it recognises are left alone, because they may be objects
# it was never shown.
OVERLAP_SAME_THING = 0.55
DEFAULT_WEIGHTS = os.getenv("DRONIONS_TRAINED_WEIGHTS", "models/room_detector.pt")
TRAINED_CONF = float(os.getenv("DRONIONS_TRAINED_CONF", "0.25"))


def _overlap(a, b) -> float:
    """Fraction of the smaller box that lies inside the other.

    Not IoU: the two models disagree about extent on small objects -- on the
    phone one boxed 39x50 and the other 76x70, an IoU of 0.28 -- while the
    smaller box sat almost entirely inside the larger. Containment is the
    question being asked, which is whether they are looking at the same thing.
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    smaller = min(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
                  max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class HybridDetector:
    """Drop-in for YOLOWorldDetector that also consults a trained model."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        self.world = YOLOWorldDetector()
        self.target = ""
        self.suppressed = 0
        self.trained = None
        self.trained_classes = []
        if weights and os.path.exists(weights):
            try:
                from ultralytics import YOLO
                self.trained = YOLO(weights)
                names = self.trained.names
                self.trained_classes = [names[i] for i in sorted(names)]
                print(f"[HIBRIT] Egitilmis model yuklendi: {weights} "
                      f"({', '.join(self.trained_classes)})")
            except Exception as exc:              # noqa: BLE001
                # A missing or broken checkpoint must not take the search down
                # with it. The open model alone is the system as it was.
                print(f"[HIBRIT] Egitilmis model yuklenemedi ({exc}) -- "
                      f"sadece YOLO-World kullanilacak.")
                self.trained = None
        else:
            print(f"[HIBRIT] Egitilmis model yok ({weights}) -- "
                  f"sadece YOLO-World kullanilacak.")

    # ------------------------------------------------------------------
    # The rest of the pipeline talks to a detector through these three.

    def set_target(self, target: str):
        self.target = (target or "").lower().strip()
        self.world.set_target(target)

    def set_target_classes(self, classes):
        """Surface scan: furniture, which the trained model knows nothing of."""
        self.target = ""
        self.world.set_target_classes(classes)

    def get_target(self):
        return self.target

    @property
    def prompts(self):
        return self.world.prompts

    @property
    def negative_prompts(self):
        return self.world.negative_prompts

    @property
    def model(self):
        return self.world.model

    def uses_trained(self) -> bool:
        return self.trained is not None and self.target in self.trained_classes

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectionCandidate]:
        found = self.world.detect(frame)
        if frame is None or not self.uses_trained():
            return found
        everything = self._trained_candidates(frame, all_classes=True)
        if not everything:
            return found
        mine = [c for c in everything if c.label == self.target]
        merged = list(mine)
        for c in found:
            overlapping = [t for t in everything
                           if _overlap(c.bbox, t.bbox) > OVERLAP_SAME_THING]
            if not overlapping:
                # Nothing the trained model recognises is there. It may be an
                # object it was never shown, so the open model keeps its say.
                merged.append(c)
                continue
            # It is on something identified. Same class means the two models
            # found the same thing and one candidate is enough; a different
            # class means the open model has boxed, say, the book and called it
            # a phone -- which is the failure this exists to remove.
            if not any(t.label == self.target for t in overlapping):
                self.suppressed += 1
        merged.sort(key=lambda c: c.confidence, reverse=True)
        return merged

    def _trained_candidates(self, frame, all_classes: bool = False
                            ) -> List[DetectionCandidate]:
        res = self.trained.predict(frame, conf=TRAINED_CONF, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        h, w = frame.shape[:2]
        img_area = h * w
        out = []
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(),
                                   boxes.conf.tolist(),
                                   boxes.cls.tolist()):
            label = self.trained_classes[int(cls)]
            if not all_classes and label != self.target:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cand = DetectionCandidate(label=label, confidence=float(conf),
                                      bbox=(x1, y1, x2, y2), prompt=self.target)
            # The same geometry the open detector populates, because every gate
            # downstream reads these and none of them should be able to tell
            # which model produced a candidate.
            cand.image_width, cand.image_height = w, h
            cand.width, cand.height = x2 - x1, y2 - y1
            cand.pixel_area = cand.width * cand.height
            cand.relative_area = cand.pixel_area / img_area if img_area else 0
            cand.aspect_ratio = cand.width / max(1, cand.height)
            cand.center = (int(x1 + cand.width / 2), int(y1 + cand.height / 2))
            cand.normalized_center = (cand.center[0] / w, cand.center[1] / h)
            cx, cy = cand.normalized_center
            cand.distance_to_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            cand.source = "trained"
            out.append(cand)
        return out
