"""
Görevi: Kullanıcının komutunu alır, kameradan fotoğraf çeker ve Gemini VLM'e göndererek nesneyi arar.
"""
import cv2
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


def capture_and_analyze(camera_source: int, target: str) -> str:
    """
    Kameradan 1 kare fotoğraf çeker ve Gemini modeline hedefin yerini sorar.
    """
    if not GEMINI_AVAILABLE:
        return "Gemini kütüphanesi bulunamadı."

    print(f"[{target}] aranıyor, kameradan görüntü alınıyor...")
    
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        return "Kamera açılamadı."
        
    # Kameranın aydınlanması ve netleşmesi için birkaç kare atla
    for _ in range(10):
        cap.read()
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "Görüntü alınamadı."

    # Resmi BGR'dan RGB'ye çevirip PIL formatına getir (Gemini için)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(rgb_frame)
    
    prompt = f"The user is looking for '{target}'. Look at this image carefully. Is the {target} in the image? If it is, describe its exact location relative to other objects in the scene briefly in English."

    print("Görüntü Gemini'ye gönderiliyor...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_image]
        )
        return response.text
    except Exception as e:
        return f"Gemini API Hatası: {e}"
