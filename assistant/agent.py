"""
Görevi: Kullanıcının komutunu alır, kameradan fotoğraf çeker ve Gemini VLM'e göndererek nesneyi arar.
"""
import cv2
import numpy as np
import PIL.Image
from config import GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai is not installed properly.")


def capture_and_analyze(frame: np.ndarray, target: str) -> dict:
    """
    Kameradan gelen 1 kare fotoğrafı Gemini modeline göndererek hedefin yerini sorar.
    Returns: {"found": bool, "message": str}
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "message": "Gemini kütüphanesi bulunamadı."}

    print(f"[VLM SEARCHING] '{target}' aranıyor, görüntü analiz ediliyor...")
    
    if frame is None or frame.size == 0:
        return {"found": False, "message": "Geçerli bir görüntü sağlanamadı."}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(rgb_frame)
    
    prompt = f"The user is looking for '{target}'. Look at this image carefully. Is the {target} in the image? If it is, start your response with EXACTLY the word '[YES]' and then describe its location. If it is NOT in the image, start your response with EXACTLY the word '[NO]' and briefly explain."

    try:
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=[prompt, pil_image]
        )
        text = response.text.strip()
        found = "[YES]" in text.upper()
        return {"found": found, "message": text}
    except Exception as e:
        return {"found": False, "message": f"Gemini API Hatası: {e}"}
