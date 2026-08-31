"""
Görevi: Ses (TTS) üretir.
"""
from config import VOICE_ENABLED
import pyttsx3
import threading
import queue
import sys
if sys.platform == 'win32':
    import pythoncom

speech_queue = queue.Queue()

def tts_worker():
    try:
        # Windows COM nesnelerini arka planda kullanabilmek için gerekli
        if sys.platform == 'win32':
            pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        
        # Bir İngilizce ses seç. Önceki hâli yalnızca Windows seslerini
        # (Zira, David) arıyordu; Linux'ta ikisi de yok, dolayısıyla döngü
        # hiçbir şey yapmadan geçiyor ve motorun varsayılanı ne ise o
        # konuşuyordu -- yani hangi sesin kullanıldığı kontrolümüzde değildi.
        # Burada espeak'in İngilizce seslerinden biri açıkça seçiliyor,
        # bulunamazsa varsayılana düşülüyor.
        voices = engine.getProperty('voices')
        preferred = ("ZIRA", "DAVID", "ENGLISH-US", "EN-US", "ENGLISH")
        chosen = None
        for want in preferred:
            for voice in voices:
                blob = f"{voice.id} {getattr(voice, 'name', '')}".upper()
                if want in blob:
                    chosen = voice.id
                    break
            if chosen:
                break
        if chosen:
            engine.setProperty('voice', chosen)
                
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
