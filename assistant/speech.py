"""
Görevi: Ses (TTS) üretir.
"""
from config import VOICE_ENABLED
import pyttsx3
import threading
import queue
import pythoncom

speech_queue = queue.Queue()

def tts_worker():
    try:
        # Windows COM nesnelerini arka planda kullanabilmek için gerekli
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        
        # İngilizce ses seçimi (Windows'ta genelde Zira veya David vardır)
        voices = engine.getProperty('voices')
        for voice in voices:
            if "ZIRA" in voice.id.upper() or "DAVID" in voice.id.upper():
                engine.setProperty('voice', voice.id)
                break
                
        while True:
            text = speech_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[SES HATASI] Konuşma sırasında hata: {e}")
                
    except Exception as e:
        print(f"[SES SİSTEMİ HATASI] Ses motoru başlatılamadı: {e}")

if VOICE_ENABLED:
    threading.Thread(target=tts_worker, daemon=True).start()

def speak(text: str):
    """
    Ses kuyruğuna (queue) metin ekler, arka plan iş parçacığı konuşur.
    Böylece kamera döngüsü asla donmaz.
    """
    print(f"[DRONE SPEECH] {text}")
    if VOICE_ENABLED:
        speech_queue.put(text)
