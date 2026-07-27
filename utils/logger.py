"""
Görevi: Çalışma sırasında olan biteni loglar.
"""
import os
import datetime

def log_event(message: str):
    """
    Logs an event with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/dronions_run.log", "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")
