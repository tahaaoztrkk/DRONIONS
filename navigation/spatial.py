"""
Görevi: Mekânsal analiz (örn. 'charger is next to the laptop'). İleride depth estimation eklenebilir.
"""

from typing import List, Dict, Any
from perception.candidate import DetectionCandidate

def analyze_spatial_relations(objects: List[DetectionCandidate]) -> List[Dict[str, Any]]:
    """
    Given a list of objects, describes their spatial relations.
    """
    relations = []
    if len(objects) < 2:
        return relations
        
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            obj1 = objects[i]
            obj2 = objects[j]
            
            x1, y1 = obj1.center
            x2, y2 = obj2.center
            
            if x1 < x2:
                rel = f"{obj1.label} is to the left of {obj2.label}"
            else:
                rel = f"{obj1.label} is to the right of {obj2.label}"
                
            relations.append({
                "subject": obj1.label,
                "object": obj2.label,
                "relation": rel
            })
            
    return relations
