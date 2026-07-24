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


import os

def capture_and_analyze(frame: np.ndarray, target: str, reference_img_path: str = None) -> dict:
    """
    Kameradan gelen 1 kare fotoğrafı Gemini modeline göndererek hedefin yerini sorar.
    Eğer reference_img_path verilmişse (Görsel Hafıza Bankası), o spesifik nesneyi arar.
    Returns: {"found": bool, "message": str}
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "message": "Gemini kütüphanesi bulunamadı."}

    print(f"[VLM SEARCHING] '{target}' aranıyor, görüntü analiz ediliyor...")
    
    if frame is None or frame.size == 0:
        return {"found": False, "message": "Geçerli bir görüntü sağlanamadı."}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_img = PIL.Image.fromarray(rgb_frame)
    
    contents = []
    
    if reference_img_path and os.path.exists(reference_img_path):
        print(f"[MEMORY BANK] Referans görsel eklendi: {reference_img_path}")
        ref_img = PIL.Image.open(reference_img_path)
        prompt = (
            f"You are an AI assistant. I have provided TWO images. "
            f"The FIRST image is a reference photo of my specific '{target}'. "
            f"The SECOND image is the current camera view. "
            f"Look at the second image carefully. Is my specific '{target}' (the exact one from the reference image) present in the camera view? "
            f"Ignore other generic objects of the same type. "
            f"If my exact '{target}' is in the camera image, start your response with EXACTLY the word '[YES]' and describe its location. "
            f"If it is NOT in the image, start your response with EXACTLY the word '[NO]' and briefly explain."
        )
        contents = [prompt, ref_img, camera_img]
    else:
        prompt = (
            f"The user is looking for '{target}'. Look at this image carefully. Is the {target} in the image? "
            f"If it is, start your response with EXACTLY the word '[YES]' and then describe its location. "
            f"If it is NOT in the image, start your response with EXACTLY the word '[NO]' and briefly explain."
        )
        contents = [prompt, camera_img]

    try:
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=contents
        )
        text = response.text.strip()
        found = "[YES]" in text.upper()
        return {"found": found, "message": text}
    except Exception as e:
        return {"found": False, "message": f"Gemini API Hatası: {e}"}
