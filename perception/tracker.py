"""
Görevi: ByteTrack ile aynı nesneyi frameler boyunca takip etmek.
"""

from typing import List
import supervision as sv
import numpy as np
from perception.candidate import DetectionCandidate

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        
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
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
        
        tracked_detections = self.tracker.update_with_detections(detections=detections)
        
        # Re-map tracker IDs to candidates
        tracked_candidates = []
        for i in range(len(tracked_detections.xyxy)):
            box = tracked_detections.xyxy[i]
            tracker_id = tracked_detections.tracker_id[i]
            class_idx = tracked_detections.class_id[i]
            
            cand = candidates[class_idx]
            cand.track_id = int(tracker_id)
            cand.bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            tracked_candidates.append(cand)
            
        return tracked_candidates
