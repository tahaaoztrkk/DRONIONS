"""
Görevi: Bütün sistemi başlatır (LLM-VLM Agentic Architecture).
Komut Dinle -> VLM (Gemini) İle Fotoğraf Analiz Et -> Konuş
"""
import time
from config import CAMERA_SOURCE
from assistant.command_parser import parse_command
from assistant.agent import capture_and_analyze
from assistant.speech import speak
from utils.logger import log_event

def main():
    log_event("Initializing DRONIONS AI (VLM Agent) pipeline...")
    speak("Sistem başlatıldı, komut bekliyorum.")
    
    while True:
        try:
            # 1. Komut Alma
            user_input = input("\n[DRONIONS] Lütfen bir komut girin ('çıkış' veya 'q' ile kapatın): ")
            if user_input.lower() in ['q', 'çıkış', 'quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            parsed_cmd = parse_command(user_input)
            target = parsed_cmd.get("target")
            
            if not target:
                # If regex fails, fallback to using the whole phrase as target
                target = user_input.replace("find", "").replace("search", "").strip()

            if not target:
                print("Hedef anlaşılamadı.")
                continue

            log_event(f"Hedef belirlendi: {target}")
            speak(f"{target} aranıyor...")
            
            # 2. VLM İle Analiz (Kameradan fotoğraf çekip sorar)
            response = capture_and_analyze(CAMERA_SOURCE, target)
            
            # 3. Sonucu Sesli/Yazılı İletme
            log_event(f"Gemini Cevabı: {response}")
            speak(response)
            
        except KeyboardInterrupt:
            print("\nSistem kapatılıyor...")
            break
        except Exception as e:
            log_event(f"Hata: {e}")
            
    log_event("System shutdown.")

if __name__ == '__main__':
    main()
