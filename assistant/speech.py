"""
Görevi: Ses (TTS) üretir.

İki motor: varsa Piper (yerel sinirsel sentez), yoksa pyttsx3/espeak.

Ayrım bir demo gereksinimi değil, projenin konusuyla ilgili. Referans çalışma
(Wei et al., CHI '26) konuşmanın doğal olmamasını kullanıcı şikâyeti olarak
kaydediyor, ve espeak 1990'ların formant sentezi -- robotik olması yapısal, ses
seçimiyle düzelmiyor. Piper yerel çalışır, yani demoda ağ, kota ve gecikme
eklemez; bulut TTS'lerin her cümlede getireceği kırılma noktası burada yok.

Model yoksa ya da Piper kurulu değilse sessizce eski motora düşer: ses
üretmemek, hiç konuşmamaktan iyidir.
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
import wave

from config import VOICE_ENABLED

if sys.platform == 'win32':
    import pythoncom

speech_queue = queue.Queue()

PIPER_VOICE = os.getenv(
    "DRONIONS_PIPER_VOICE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "voices", "en_US-amy-medium.onnx"))


def _load_piper():
    """Piper voice, or None if it is not usable here."""
    try:
        from piper import PiperVoice
    except ImportError:
        return None
    if not os.path.exists(PIPER_VOICE):
        print(f"[SES] Piper sesi yok ({PIPER_VOICE}) -- espeak kullanilacak.")
        return None
    try:
        return PiperVoice.load(PIPER_VOICE)
    except Exception as exc:                       # noqa: BLE001
        print(f"[SES] Piper yuklenemedi ({exc}) -- espeak kullanilacak.")
        return None


def _speak_with_piper(voice, text: str):
    """Synthesise to a temp wav and play it.

    A file rather than a stream: the player is a separate process, so a
    synthesis that fails cannot leave the audio device half-open, and the
    worker thread is the only thing that blocks.
    """
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fh:
            path = fh.name
        with wave.open(path, 'wb') as w:
            voice.synthesize_wav(text, w)
        subprocess.run(['aplay', '-q', path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def tts_worker():
    piper = _load_piper()
    engine = None
    if piper is None:
        try:
            import pyttsx3
            if sys.platform == 'win32':
                pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            # Pick an English voice explicitly. The earlier version looked only
            # for Windows voices (Zira, David); on Linux neither exists, so the
            # loop did nothing and whatever the engine defaulted to did the
            # talking -- which voice that was, was not under our control.
            voices = engine.getProperty('voices')
            for want in ("ZIRA", "DAVID", "ENGLISH-US", "EN-US", "ENGLISH"):
                hit = next((v.id for v in voices
                            if want in f"{v.id} {getattr(v, 'name', '')}".upper()),
                           None)
                if hit:
                    engine.setProperty('voice', hit)
                    break
        except Exception as exc:                   # noqa: BLE001
            print(f"[SES SİSTEMİ HATASI] Ses motoru baslatilamadi: {exc}")
            return

    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            if piper is not None:
                _speak_with_piper(piper, text)
            else:
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:                   # noqa: BLE001
            print(f"[SES HATASI] Konusma sirasinda hata: {exc}")


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
