import speech_recognition as sr
import threading
import queue

# Above this, a calibration is describing a broken input rather than a room.
MAX_ENERGY_THRESHOLD = 4000


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
        print("\n[MIC] Calibrating for background noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        # Report it, and refuse an impossible one.
        #
        # The calibrated threshold was never printed, so when it came back at
        # 22757 -- seventy times the default -- nothing said so and speech
        # simply never registered. The cause was the capture gain sitting at
        # +30 dB, which saturated the input: ambient noise measured a median
        # RMS of 13135 against a 16-bit ceiling of 32767, leaving speech no
        # headroom to rise into. Halving the gain brought ambient to 697.
        #
        # The clamp does not fix a saturated microphone; it makes the symptom
        # visible instead of silent, and keeps a noisy calibration from
        # switching listening off altogether.
        if recognizer.energy_threshold > MAX_ENERGY_THRESHOLD:
            print(f"[MIC] Calibrated threshold {recognizer.energy_threshold:.0f} "
                  f"is implausibly high -- the input is probably saturated. "
                  f"Capping at {MAX_ENERGY_THRESHOLD}. Check the capture gain: "
                  f"amixer sget Capture")
            recognizer.energy_threshold = MAX_ENERGY_THRESHOLD
        else:
            print(f"[MIC] Noise threshold: {recognizer.energy_threshold:.0f}")
        print("[MIC] Listening -- say commands in English, e.g. 'find my laptop'")

        while True:
            try:
                # Sesi dinle (kısa aralıklarla timeout koyuyoruz ki takılı kalmasın)
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                # Google STT API (Ücretsiz, İngilizce)
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"\n[YOU] {text}")
                
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
                print(f"[MIC] Speech service unreachable: {e}")
                break
            except Exception as e:
                # Beklenmedik hata
                print(f"[MIC] Error in the listening loop: {e}")
                break
