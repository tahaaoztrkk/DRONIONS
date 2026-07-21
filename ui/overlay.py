"""
Görevi: Ekrana kutu, etiket, FPS, hedef oku, merkez çizgisi çizer.
"""
import cv2
import numpy as np
from typing import List, Dict, Any
from perception.candidate import DetectionCandidate

def draw_overlay(frame: np.ndarray, candidates: List[DetectionCandidate], nav_decision: Dict[str, Any] = None, phase: str = "SEARCHING") -> np.ndarray:
    """
    Draws boxes, IDs, labels, confidences, and navigation instructions on the frame.
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]
    
    # Phase text
    cv2.putText(out_frame, f"PHASE: {phase}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if phase == "VLM SEARCHING" else (0, 255, 0), 2)
    
    # Draw center crosshair
    cv2.line(out_frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.line(out_frame, (0, h//2), (w, h//2), (255, 255, 255), 1)
    
    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox
        color = (0, 255, 0)
        
        # Bounding box
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        text = f"{cand.label} ID:{cand.track_id} {cand.confidence:.2f}"
            
        cv2.putText(out_frame, text, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Tracking center
        cx, cy = cand.center
        cv2.circle(out_frame, (cx, cy), 5, (0, 0, 255), -1)

    # Draw Navigation Info
    if nav_decision:
        action = nav_decision.get('action', 'NONE')
        angle = nav_decision.get('angle', 0)
        nav_text = f"CMD: {action} | ANGLE: {angle}deg"
        cv2.putText(out_frame, nav_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    return out_frame
