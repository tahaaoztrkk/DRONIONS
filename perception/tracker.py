"""
Görevi: ByteTrack ile aynı nesneyi frameler boyunca takip etmek.
"""

from typing import List
import supervision as sv
import numpy as np
from trackers import ByteTrackTracker
from perception.candidate import DetectionCandidate
from config import CONFIDENCE_THRESHOLD

class Tracker:
    def __init__(self):
        # supervision's own sv.ByteTrack (deprecated since 0.28.0) has a real
        # bug in this version: it activates a track on the first detection
        # and then fails to continue it on every call after, even with an
        # identical, unmoving, high-confidence detection (reproduced in
        # isolation). supervision's own docs point at the `trackers` package's
        # ByteTrackTracker as the replacement, which does not have this bug.
        #
        # Both thresholds are derived from CONFIDENCE_THRESHOLD (same value)
        # so there's no gap between "YOLO reports it" and "tracker accepts
        # it" -- a gap there previously caused marginal detections to pass
        # YOLO's filter but never actually get tracked.
        activation_threshold = max(0.01, CONFIDENCE_THRESHOLD - 0.05)
        self.tracker = ByteTrackTracker(
            track_activation_threshold=activation_threshold,
            high_conf_det_threshold=activation_threshold,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
        )

    def track_objects(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        if not candidates:
            return []

        # Convert candidates to sv.Detections
        xyxy = []
        confidences = []
        class_ids = []

        for idx, cand in enumerate(candidates):
            xyxy.append(cand.bbox)
            confidences.append(cand.confidence)
            class_ids.append(idx) # Using index as class id just for tracking mapper

        detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=float),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        tracked_detections = self.tracker.update(detections)

        # Re-map tracker IDs to candidates. tracker_id == -1 means not yet
        # confirmed as a track (first frame it's seen on).
        tracked_candidates = []
        for i in range(len(tracked_detections.xyxy)):
            tracker_id = tracked_detections.tracker_id[i]
            if tracker_id < 0:
                continue

            box = tracked_detections.xyxy[i]
            class_idx = tracked_detections.class_id[i]

            cand = candidates[class_idx]
            cand.track_id = int(tracker_id)
            cand.bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            tracked_candidates.append(cand)

        return tracked_candidates
