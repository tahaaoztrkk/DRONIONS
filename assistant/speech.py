"""
Görevi: Ses (TTS) üretir.
"""
from config import VOICE_ENABLED

def speak(text: str):
    """
    Outputs speech. Currently mocked as print per user request.
    """
    if VOICE_ENABLED:
        # TTS Engine would be invoked here
        pass
        
    print(f"[DRONE SPEECH] {text}")
