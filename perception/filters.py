"""
Görevi: Geometri, aspect ratio, area, merkez uzaklığı ve confidence kullanarak adayları filtrelemek.
"""

from typing import List
from perception.candidate import DetectionCandidate

def filter_candidates(candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
    """
    Applies heuristic filters to reject false positives (like backgrounds)
    and scores remaining candidates to find the most likely target.
    """
    filtered = []
    
    for cand in candidates:
        # Example filters:
        # Reject if area is too large (probably background or floor instead of an object)
        if cand.relative_area > 0.8:
            continue
            
        # Reject if area is too small (likely noise)
        if cand.relative_area < 0.001:
            continue

        # Calculate a geometry score (reward being close to the center and a reasonable size)
        cand.center_score = max(0.0, 1.0 - cand.distance_to_center * 2)
        
        # We can penalize extreme aspect ratios (e.g. > 10 or < 0.1) depending on object type
        cand.geometry_score = cand.center_score * 0.5 + cand.relative_area * 0.5
        
        # Combine confidence and geometry
        cand.final_score = cand.confidence * 0.7 + cand.geometry_score * 0.3
        
        filtered.append(cand)

    # Sort descending by final score
    filtered.sort(key=lambda x: x.final_score, reverse=True)
    return filtered
