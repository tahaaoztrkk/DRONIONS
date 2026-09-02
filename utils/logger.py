"""
Görevi: Çalışma sırasında olan biteni loglar.

Konsola İngilizce, dosyaya Türkçe yazar. Ayrımın sebebi ikisinin iki ayrı
okuyucusu olması:

  - Dosya kanıttır ve ölçüm araçları onu ayrıştırır. `experiment_repeatability`
    "HEDEFE VARILDI", "Hedef konumu (dunya)", "ANKET r=", "Ucus anomalisi" gibi
    dizeleri arıyor. Çevrilirse hiçbiri eşleşmez ve kampanya veri toplamamış
    gibi görünür -- o betiğin kendi yorumu bu tuzağa iki kez düşüldüğünü
    yazıyor. Dosya bu yüzden olduğu gibi kalır.
  - Konsol demoda izlenir ve izleyen Türkçe bilmiyor.

Aşağıdaki tablo yalnızca demoda ekranda akan satırları kapsar, hepsini değil.
Eşleşmeyen satır olduğu gibi geçer; yani tablo eksik olabilir ama yanlış
olamaz.
"""
import os
import re
import datetime

# (kalıp, İngilizce karşılık). Sayılar ve adlar grup olarak taşınır.
_CONSOLE_EN = [
    (r"^Yeni hedef -- once bulunulan yer taraniyor\.$",
     "New target -- sweeping the current position first."),
    (r"^Arama konumu \((.*?)\) yaw=(\S+) irtifa=(\S+) hedef_wp=\((.*?)\)$",
     r"Search position (\1) yaw=\2 altitude=\3 waypoint=(\4)"),
    (r"^Referans rengi \((\d+) aci\): (.*)$",
     r"Reference colour (\1 angles): \2"),
    (r"^'(.*?)' icin (\d+) prompt: (.*)$",
     r"\2 prompts for '\1': \3"),
    (r"^Eleme: (\d+) adaydan (\d+) boyut, (\d+) renk -- (\d+) kaldi "
     r"\(toplam boyut (\d+), renk (\d+)\)\.$",
     r"Filtered: \1 candidates, \2 by size, \3 by colour -- \4 left "
     r"(\5 by size and \6 by colour so far)."),
    (r"^Gemini beklerken kayma: (\S+) m, (\S+) derece \((\S+) s surdu\)\.$",
     r"Drift while waiting for the model: \1 m, \2 degrees (took \3 s)."),
    (r"^Gemini Cevab.: (.*)$", r"Model reply: \1"),
    (r"^Gemini kirpma #(\d+) sec ti: (.*)$", r"Model chose crop #\1: \2"),
    (r"^Hedef ortalandi \((.*?)\) -- takibe geciliyor\.$",
     r"Target centred (\1) -- switching to tracking."),
    (r"^Hedef konumu \(dunya\) (.*)$", r"Target position (world) \1"),
    (r"^  \^ kaynak=(\S+) etiket=(\S+) (.*)$",
     r"  ^ source=\1 label=\2 \3"),
    (r"^Aday secildi \(Gemini konumuna en yakin, fark (\S+)\): (.*)$",
     r"Candidate selected (nearest to the model's point, gap \1): \2"),
    (r"^HEDEFE VARILDI: (\S+) irtifa (\S+), hedefe (\S+)$",
     r"ARRIVED: \1 at altitude \2, \3 from the target"),
    (r"^VARIS reddedildi: (.*)$", r"Arrival rejected: \1"),
    (r"^Yuzey bulundu: (\S+) \((.*?)\), tahmini ust (\S+) m\. "
     r"Bilinen yuzeyler: (.*)$",
     r"Surface found: \1 (\2), estimated top \3 m. Known surfaces: \4"),
    (r"^Yuzey bulundu: (.*)$", r"Surface found: \1"),
    (r"^Isinma: ilk (\S+) s boyunca tanimlama yapilmiyor, supurme suruyor\.$",
     r"Warm-up: no identification for the first \1 s, sweep continues."),
    (r"^Ucus anomalisi \((\S+)\): (\S+) m hareket (\S+) s icinde \((.*?)\) "
     r"(.*?) -- komut edilen hizin cok ustunde\. Arama durduruluyor\.$",
     r"Flight anomaly (\1): moved \2 m in \3 s (\4) \5 "
     r"-- far above the commanded speed. Stopping the search."),
    (r"^Ucus anomalisi \((\S+)\): (.*)$", r"Flight anomaly (\1): \2"),
    (r"^Gemini kotasi doldu -- (\S+)s beklenip aramaya devam edilecek\.$",
     r"Model quota exhausted -- waiting \1s, then continuing the search."),
    (r"^Egitilmis model hedefi bulmadi -- VLM'e sorulmadi \((\d+) kare\)\.$",
     r"Trained detector did not find the target -- model not asked (\1 frames)."),
    (r"^Uyari: '(.*?)' icin prompt genislemesi yok -- tek ifadeyle araniyor, "
     r"tespit zayif olabilir\.$",
     r"Warning: no prompt expansion for '\1' -- searching on the single "
     r"phrase, detection may be weak."),
    (r"^(\S+) tarandi, bulunamadi\. Kalan yuzeyler: (.*)$",
     r"Scanned the \1, not found. Remaining surfaces: \2"),
    (r"^Suprme sonucsuz -- (\S+) \((.*?)\) yuzeyine gidip yakindan bakiliyor\.$",
     r"Sweep found nothing -- going to look closely at the \1 (\2)."),
    (r"^(\S+) ustunde (\S+) m'de tarama basladi\.$",
     r"Scanning above the \1 at \2 m."),
    (r"^Son goruldugu yere donuluyor: \((.*?)\)\.$",
     r"Returning to where it was last seen: (\1)."),
    (r"^Yeni hedef -- once bulunulan yer taraniyor\.$",
     r"New target -- sweeping the current position first."),
    (r"^Arama irtifasina cikiliyor \((\S+) -> (\S+) m\) -- once yukselip "
     r"sonra soruluyor\.$",
     r"Climbing to search altitude (\1 -> \2 m) -- asking once up there."),
    (r"^PX4: (.*)$", r"PX4: \1"),
]
_CONSOLE_EN = [(re.compile(p), r) for p, r in _CONSOLE_EN]


_PHASES = {"seyir": "cruise", "kalkis": "takeoff"}


def _for_console(message: str) -> str:
    for pattern, replacement in _CONSOLE_EN:
        if pattern.match(message):
            out = pattern.sub(replacement, message)
            for tr, en in _PHASES.items():
                out = out.replace(f"({tr})", f"({en})")
            return out
    return message


def log_event(message: str):
    """Zaman damgalı bir olayı kaydeder.

    Dosyaya orijinal metin, konsola varsa İngilizce karşılığı gider.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {_for_console(message)}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/dronions_run.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
