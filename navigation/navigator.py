"""
Görevi: Drone için nereye gidileceğini (sağ, sol, derece vb.) hesaplar.
"""

from perception.candidate import DetectionCandidate
from typing import Dict, Any

def get_navigation_decision(target: DetectionCandidate) -> Dict[str, Any]:
    """
    Computes movement commands based on bounding box position relative to image center.
    Returns a dictionary with command, angle, and speed.
    """
    if target is None:
        return {"action": "SEARCHING", "angle": 0, "speed": 0}
        
    cx, cy = target.normalized_center
    
    # Simple P-controller logic mapping
    # normalized x is 0.0 (left) to 1.0 (right). Center is 0.5.
    error_x = cx - 0.5
    
    # Max FOV roughly 60 degrees. Let's map error to degrees
    angle = int(error_x * 60)
    
    if abs(error_x) < 0.1:
        action = "FORWARD"
        angle = 0
    elif error_x > 0:
        action = "TURN_RIGHT"
    else:
        action = "TURN_LEFT"
        
    return {
        "action": action,
        "angle": angle,
        "speed": 10 if action == "FORWARD" else 0
    }
