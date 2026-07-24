import speech_recognition as sr
import threading
import queue

def get_voice_input(q: queue.Queue):
    """
    Sürekli olarak ortamı dinler (İngilizce).
    Eğer bir komut algılarsa komut kuyruğuna (queue) atar.
    """
    recognizer = sr.Recognizer()
    # Eşik değerini mikrofon gürültüsüne göre ayarla
    recognizer.energy_threshold = 300 
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("\n[MİKROFON] Gürültü seviyesi kalibre ediliyor, lütfen bekleyin...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("[MİKROFON] Dinleniyor... (Komutlarınızı İngilizce söyleyebilirsiniz, örn: 'Find my laptop')")

        while True:
            try:
                # Sesi dinle (kısa aralıklarla timeout koyuyoruz ki takılı kalmasın)
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                # Google STT API (Ücretsiz, İngilizce)
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"\n[SİZ] {text}")
                
                # Sadece içinde find, look for, search for geçen veya doğrudan bir isim olan şeyleri komut kabul edebiliriz.
                # Şimdilik duyduğu her şeyi komut sırasına alalım, parse_command onu ayıklayacaktır.
                q.put(text)
                
                if "exit" in text.lower() or "quit" in text.lower() or "stop" in text.lower():
                    # Stop komutu alırsa çıkmak için q göndersin
                    q.put("q")
                    break

            except sr.WaitTimeoutError:
                # Ses duyulmadı, dinlemeye devam et
                continue
            except sr.UnknownValueError:
                # Ses algılandı ama anlaşılamadı
                pass
            except sr.RequestError as e:
                print(f"[HATA] STT Servisine ulaşılamadı: {e}")
                break
            except Exception as e:
                # Beklenmedik hata
                print(f"[HATA] Mikrofon döngüsünde hata: {e}")
                break
