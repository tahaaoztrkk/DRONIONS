"""
Görevi: SigLIP ve Gemini ile ikinci doğrulama katmanı.
"""

from typing import List
import numpy as np
import cv2
from perception.candidate import DetectionCandidate
from config import USE_GEMINI, GEMINI_API_KEY

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def verify_objects(frame: np.ndarray, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
    """
    Crops the objects and uses Gemini to verify if the object matches the target.
    This can be slow, so usually only run on the best candidate or in a separate thread.
    For demonstration, we run on the top candidate.
    """
    if not USE_GEMINI or not GEMINI_AVAILABLE or not candidates:
        return candidates

    # Let's only verify the top candidate to save time
    top_cand = candidates[0]
    
    if top_cand.gemini_verified:
        return candidates

    x1, y1, x2, y2 = top_cand.bbox
    
    # Ensure bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return candidates

    # Convert to PIL Image for Gemini
    from PIL import Image
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    
    prompt = f"Is this a {top_cand.prompt}? Answer with only YES or NO."
    
    try:
        response = model.generate_content([prompt, pil_img])
        answer = response.text.strip().lower()
        if "yes" in answer:
            top_cand.verified = True
            top_cand.gemini_verified = True
            top_cand.description = "Verified by Gemini"
        else:
            top_cand.verified = False
            top_cand.gemini_verified = True
            top_cand.description = "Rejected by Gemini"
    except Exception as e:
        print(f"[Verifier] Gemini API Error: {e}")
        
    return candidates
