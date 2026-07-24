"""
Görevi: Bütün sistemi başlatır (Hibrit Mimari: VLM Arama + YOLO Takip).
Mod 1 (PHASE_SEARCH): Gemini ile nesnenin odadaki bağlamsal yerini bulur.
Mod 2 (PHASE_TRACK): YOLO-World ile nesneye kilitlenip 30FPS takip eder.
"""
import time
import os

# FFmpeg ve OpenCV arka plan loglarını (mjpeg overread vb.) tamamen kapatır.
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
# Bu ayar IP kameralarda overread (paket kaybı) loglarını susturmak için kullanılır
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

import cv2
from config import CAMERA_SOURCE, VOICE_ENABLED
from assistant.command_parser import parse_command
from assistant.agent import capture_and_analyze
from assistant.speech import speak
from assistant.listen import get_voice_input
from utils.logger import log_event

from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from perception.tracker import Tracker
from navigation.navigator import get_navigation_decision
from ui.overlay import draw_overlay

import threading
import queue

PHASE_SEARCH = "VLM SEARCHING"
PHASE_TRACK = "YOLO TRACKING"

def get_console_input(q: queue.Queue):
    while True:
        cmd = input("\n[DRONIONS] Lütfen bir komut girin ('çıkış' veya 'q' ile kapatın): ")
        q.put(cmd)
        if cmd.lower() in ['q', 'çıkış', 'quit', 'exit']:
            break

def main():
    log_event("Initializing DRONIONS AI (Hybrid VLM + YOLO) pipeline...")
    speak("System activated. I am listening for your commands.")
    
    # Initialize YOLO components but keep them idle until tracking starts
    detector = YOLOWorldDetector()
    tracker = Tracker()
    
    current_phase = PHASE_SEARCH
    target = None
    frames_lost = 0
    MAX_FRAMES_LOST = 60 # About 2 seconds at 30 fps before falling back to VLM
    last_vlm_check_time = 0
    VLM_CHECK_INTERVAL = 5.0 # Free-tier limitlerine takılmamak için 5 saniyede bir kontrol eder
    
    cmd_queue = queue.Queue()
    
    # Klavye dinleyicisi
    input_thread = threading.Thread(target=get_console_input, args=(cmd_queue,), daemon=True)
    input_thread.start()
    
    # Ses dinleyicisi (Mikrofon)
    if VOICE_ENABLED:
        voice_thread = threading.Thread(target=get_voice_input, args=(cmd_queue,), daemon=True)
        voice_thread.start()
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    def reconnect_camera(old_cap):
        print("\n[UYARI] Kamera bağlantısı zayıf veya koptu. Yeniden bağlanılıyor...")
        if old_cap:
            old_cap.release()
        time.sleep(1)
        new_cap = cv2.VideoCapture(CAMERA_SOURCE)
        return new_cap
    
    try:
        while True:
            # Phase: SEARCH (VLM)
            if current_phase == PHASE_SEARCH:
                # Ekranda kamerayı göster ve tamponu boşalt (donma ve siyah ekranı önler)
                ret, frame = cap.read()
                if not ret:
                    cap = reconnect_camera(cap)
                    continue

                if ret:
                    cv2.imshow("DRONIONS AI", draw_overlay(frame, [], None, phase=current_phase))
                    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                try:
                    user_input = cmd_queue.get_nowait()
                    if user_input.lower() in ['q', 'çıkış', 'quit', 'exit']:
                        break
                    if user_input.strip():
                        parsed_cmd = parse_command(user_input)
                        target = parsed_cmd.get("target") or user_input.strip()
                        log_event(f"Hedef belirlendi: {target}")
                        speak(f"{target} aranıyor...")
                        last_vlm_check_time = 0 # Yeni komut girildiğinde beklemeden hemen analiz et
                except queue.Empty:
                    pass
                
                if not target:
                    continue # Target yoksa bekle ve kamera akışını canlandır
                
                current_time = time.time()
                if current_time - last_vlm_check_time > VLM_CHECK_INTERVAL:
                    # Check for Visual Memory Bank image
                    target_clean = target.lower().strip()
                    target_underscore = target_clean.replace(" ", "_")
                    ref_path = None
                    for ext in [".jpg", ".png", ".jpeg"]:
                        for name in [target_clean, target_underscore]:
                            path = os.path.join("memory", name + ext)
                            if os.path.exists(path):
                                ref_path = path
                                break
                        if ref_path:
                            break
                            
                    # Call Gemini API
                    if frame is not None:
                        result = capture_and_analyze(frame, target, reference_img_path=ref_path)
                    else:
                        result = {"found": False, "message": "Kamera görüntüsü alınamadı, lütfen kameranızı kontrol edin."}
                        
                    log_event(f"Gemini Cevabı: {result['message']}")
                    speak(result['message'])
                    
                    if result['found']:
                        print("\n[!] Gemini hedefi doğruladı. YOLO takibine geçiliyor...")
                        speak("Hedef doğrulandı. Takibe geçiliyor.")
                        detector.set_target(target)
                        tracker = Tracker() # Reset tracker
                        frames_lost = 0
                        current_phase = PHASE_TRACK
                        # Hedefi bilerek sıfırlamıyoruz (target = None YAPMIYORUZ).
                        # Böylece nesne kaybolduğunda otomatik olarak buraya dönüp tekrar arayabilecek.
                    else:
                        print(f"\n[?] '{target}' bulunamadı. Aramaya devam ediliyor, lütfen kamerayı etrafta gezdirin...")
                        
                    last_vlm_check_time = current_time
                    
            # Phase: TRACK (YOLO)
            elif current_phase == PHASE_TRACK:
                ret, frame = cap.read()
                if not ret:
                    cap = reconnect_camera(cap)
                    continue
                    
                candidates = detector.detect(frame)
                filtered_candidates = filter_candidates(candidates)
                tracked_candidates = tracker.track_objects(filtered_candidates)
                
                nav_decision = None
                if tracked_candidates:
                    frames_lost = 0 # Reset lost counter
                    best_candidate = tracked_candidates[0]
                    nav_decision = get_navigation_decision(best_candidate)
                else:
                    frames_lost += 1
                    
                # If object is lost for too long, fallback to SEARCH phase
                if frames_lost > MAX_FRAMES_LOST:
                    print("\n[!] Hedef kaybedildi. VLM aramasına (Mod 1) geri dönülüyor...")
                    speak("Hedef kaybedildi. Ortam tekrar taranıyor.")
                    current_phase = PHASE_SEARCH
                    continue
                    
                out_frame = draw_overlay(frame, tracked_candidates, nav_decision, phase=current_phase)
                cv2.imshow("DRONIONS AI", out_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'): # Force manual reset
                    current_phase = PHASE_SEARCH
                    target = None
                    
    except KeyboardInterrupt:
        print("\nSistem kapatılıyor...")
        
    cap.release()
    cv2.destroyAllWindows()
    log_event("System shutdown.")

if __name__ == '__main__':
    main()
