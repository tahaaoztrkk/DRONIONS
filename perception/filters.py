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

        cand.center_score = max(0.0, 1.0 - cand.distance_to_center * 2)

        # relative_area used to be rewarded here (geometry was
        # center_score*0.5 + relative_area*0.5, at 0.3 of the final score).
        # For "find this particular object" that signal points the wrong way:
        # it rewards whatever is biggest and most centred, which is usually
        # background. Measured on a real frame, it ranked a 3 m wall (conf
        # 0.532, area 0.19, centred) above the actual cardboard box (conf
        # 0.566, area 0.013, off to one side) -- 0.508 vs 0.460 -- so the
        # tracker locked onto the wall even though the detector had scored the
        # box higher. Size is a sanity gate above, not evidence of identity.
        cand.geometry_score = cand.center_score

        # Ranking is the detector's confidence and nothing else. Neither size
        # nor centredness is evidence of *identity*, and both belong to the
        # background more often than to the target: at 0.9/0.1 the wall still
        # edged the box out 0.5503 to 0.5498 purely on being nearer the middle
        # of the frame, despite the box scoring higher (0.565 vs 0.532) on the
        # only signal that actually says what the thing is. center_score and
        # geometry_score stay populated for the overlay/debug view.
        cand.final_score = cand.confidence
        
        filtered.append(cand)

    # Sort descending by final score
    filtered.sort(key=lambda x: x.final_score, reverse=True)
    return filtered
