"""
Görevi: Kullanıcının komutunu alır, kameradan fotoğraf çeker ve Gemini VLM'e göndererek nesneyi arar.
"""
import re
import cv2
import numpy as np
import PIL.Image
from config import GEMINI_API_KEY, GEMINI_MODEL

try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai is not installed properly.")


import os

def capture_and_analyze(frame: np.ndarray, target: str, reference_img_path: str = None) -> dict:
    """
    Kameradan gelen 1 kare fotoğrafı Gemini modeline göndererek hedefin yerini sorar.
    Eğer reference_img_path verilmişse (Görsel Hafıza Bankası), o spesifik nesneyi arar.
    Returns: {"found": bool, "message": str}
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "message": "Gemini kütüphanesi bulunamadı."}

    print(f"[VLM SEARCHING] '{target}' aranıyor, görüntü analiz ediliyor...")
    
    if frame is None or frame.size == 0:
        return {"found": False, "message": "Geçerli bir görüntü sağlanamadı."}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_img = PIL.Image.fromarray(rgb_frame)
    
    contents = []
    
    if reference_img_path and os.path.exists(reference_img_path):
        print(f"[MEMORY BANK] Referans görsel eklendi: {reference_img_path}")
        ref_img = PIL.Image.open(reference_img_path)
        prompt = (
            f"You are an AI assistant. I have provided TWO images. "
            f"The FIRST image is a reference photo of my specific '{target}'. "
            f"The SECOND image is the current camera view. "
            f"Look at the second image carefully. Is my specific '{target}' (the exact one from the reference image) present in the camera view? "
            f"Ignore other generic objects of the same type. "
            f"If my exact '{target}' is in the camera image, start your response with EXACTLY the word '[YES]', "
            f"then IMMEDIATELY give its centre in the exact format (x=0.00, y=0.00) "
            f"where x is the fraction across from the left edge and y is the fraction down from the top edge, "
            f"then describe its location in words. "
            f"If it is NOT in the image, start your response with EXACTLY the word '[NO]' and briefly explain."
        )
        contents = [prompt, ref_img, camera_img]
    else:
        prompt = (
            f"The user is looking for '{target}'. Look at this image carefully. Is the {target} in the image? "
            f"If it is, start your response with EXACTLY the word '[YES]', "
            f"then IMMEDIATELY give its centre in the exact format (x=0.00, y=0.00) "
            f"where x is the fraction across from the left edge and y is the fraction down from the top edge, "
            f"then describe its location in words. "
            f"If it is NOT in the image, start your response with EXACTLY the word '[NO]' and briefly explain."
        )
        contents = [prompt, camera_img]

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        text = response.text.strip()
        found = "[YES]" in text.upper()
        return {"found": found, "message": text, "point": _parse_point(text) if found else None}
    except Exception as e:
        return {"found": False, "message": f"Gemini API Hatası: {e}", "point": None}


def _parse_point(text: str):
    """Pulls the (x=0.00, y=0.00) centre out of a [YES] reply.

    This is what lets the caller pick *which* detection to follow rather than
    just whether to follow something. Returns None if the model didn't answer
    in the requested format -- callers must cope, since nothing forces it to.
    """
    m = re.search(r'x\s*=\s*([01]?\.?\d+)\s*,\s*y\s*=\s*([01]?\.?\d+)', text, re.I)
    if not m:
        return None
    try:
        x, y = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x, y)
    return None


# Maximum crops sent in one selection request. Ranked by confidence, so the
# cut only ever drops the detector's least-likely candidates.
MAX_SELECTION_CROPS = 5
# Crops are padded and upscaled before sending: a distant target can be 30 px
# across in the source frame, which is not enough for the model to judge.
CROP_PADDING_PX = 12
CROP_MIN_SIDE_PX = 160


_MATCH_TOKEN = re.compile(r'\[(MATCH\s*:\s*\d+|NONE)\]', re.I)


def _spoken_context(text: str) -> str:
    """The part of the reply meant for the user, and only that.

    The model is asked for a machine token, an English justification and a
    Turkish sentence, in that order. Only the last is read aloud; the answer
    used to be assembled by splitting on the first ']' and keeping the rest,
    which would now speak the justification too. That is the same mistake as
    reading raw API text aloud, arrived at from the other direction.
    """
    out = []
    for line in _MATCH_TOKEN.sub('', text or '').splitlines():
        line = line.strip()
        if not line or line.upper().startswith('BECAUSE'):
            continue
        out.append(line)
    return ' '.join(out).strip()


def select_candidate(frame: np.ndarray, candidates, target: str,
                     reference_img_path: str = None) -> dict:
    """Asks Gemini which of YOLO's detections is the target.

    Replaces asking "is it in this view, and where?" with "here are the N
    things the detector found -- which one is yours?". Measured over six
    calls, the where-question was answered with a coordinate on the left
    edge of the frame every single time (x=0.01-0.03) regardless of where the
    object actually was, so location answers could not be used to pick a
    detection. Choosing between a handful of cropped images asks the model to
    compare rather than to localise, which is also exactly what the reference
    photo from the memory bank is for.

    Returns {"found": bool, "index": int|None, "message": str}, where index
    points into the candidates list as passed in.
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "index": None, "message": "Gemini kütüphanesi bulunamadı."}
    if frame is None or frame.size == 0 or not candidates:
        return {"found": False, "index": None, "message": "Aday yok."}

    shortlist = list(candidates)[:MAX_SELECTION_CROPS]
    h, w = frame.shape[:2]

    contents = []
    prompt = [
        f"I am looking for my specific '{target}'.",
    ]
    if reference_img_path and os.path.exists(reference_img_path):
        print(f"[MEMORY BANK] Referans görsel eklendi: {reference_img_path}")
        prompt.append(
            "The FIRST image is a reference photo of it. "
            "The images after that are numbered crops taken from the current camera view."
        )
    else:
        prompt.append("The images below are numbered crops from the current camera view.")
    prompt.append(
        # The base rate is stated because it is measured, and because without
        # it the question degrades into a yes/no. Across 80 surveyed viewpoints
        # the object was present in 14%, and the detector usually offers a
        # single crop -- so the model is handed one picture and asked "is this
        # it", which is exactly the shape that gets a reflexive yes. A run
        # approved a bright blue distractor against a tan reference photo.
        f"Most of the time the object is NOT in view: in this system roughly "
        f"six out of seven of these requests contain no match at all, so "
        f"'[NONE]' is the ordinary answer and choosing a crop is the exception. "
        f"Do not pick the closest-looking crop. Pick one only if it clearly "
        f"matches on colour, texture and markings; if it differs in any of "
        f"those, it is a different object even when the shape is identical. "
        f"Ignore walls, floors, panels and other background surfaces.\n"
        f"Reply with EXACTLY '[MATCH:n]' where n is the number of the crop "
        f"showing my '{target}', or EXACTLY '[NONE]'.\n"
        f"If you answer [MATCH:n], first state in English which specific "
        f"feature made you certain, in the form 'BECAUSE: <feature>'. Name a "
        f"visible detail, not a category -- 'the taped seam down the middle' "
        f"rather than 'it is a box'.\n"
        f"Then add ONE short sentence IN TURKISH describing what the object is "
        f"resting on or sitting next to, addressed to the user. "
        f"This sentence is read aloud to a blind person, so do not justify your "
        f"choice there, do not mention crops, images or numbers, and do not "
        f"use English in it."
    )
    contents.append("\n".join(prompt))

    if reference_img_path and os.path.exists(reference_img_path):
        contents.append(PIL.Image.open(reference_img_path))

    for i, cand in enumerate(shortlist, start=1):
        x1, y1, x2, y2 = cand.bbox
        x1 = max(0, int(x1) - CROP_PADDING_PX)
        y1 = max(0, int(y1) - CROP_PADDING_PX)
        x2 = min(w, int(x2) + CROP_PADDING_PX)
        y2 = min(h, int(y2) + CROP_PADDING_PX)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        side = max(crop.shape[0], crop.shape[1])
        if side < CROP_MIN_SIDE_PX:
            scale = CROP_MIN_SIDE_PX / side
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        contents.append(f"Crop {i}:")
        contents.append(PIL.Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        text = response.text.strip()
        m = re.search(r'\[MATCH\s*:\s*(\d+)\]', text, re.I)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(shortlist):
                return {"found": True, "index": idx, "message": text,
                        "context": _spoken_context(text)}
            return {"found": False, "index": None,
                    "message": f"Gemini gecersiz kirpma numarasi verdi: {text}"}
        return {"found": False, "index": None, "message": text,
                "context": _spoken_context(text)}
    except Exception as e:
        return {"found": False, "index": None, "message": f"Gemini API Hatası: {e}"}


def answer_followup(frame: np.ndarray, question: str, history=None,
                    reference_img_path: str = None) -> str:
    """Answers a question about what was just found.

    Follow-ups were 35% of all user queries in the reference study -- "is there
    any wording on the box", "what else was on the shelf" -- so a single-shot
    request/response is the wrong shape for this system. The frame kept from
    the moment of the answer is re-used rather than flying again: the user is
    asking about what the drone already saw.

    Replies are asked to stay short on purpose. Hearing is a safety channel for
    a blind user, and the reference work capped response length after finding
    that too much audio was overwhelming.
    """
    if not GEMINI_AVAILABLE:
        return "Gemini kütüphanesi bulunamadı."
    if frame is None or frame.size == 0:
        return "Elimde bakabileceğim bir görüntü yok."

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam = PIL.Image.fromarray(rgb)

    turns = ""
    for role, text in (history or []):
        who = "User" if role == "user" else "You"
        turns += f"{who}: {text}\n"

    prompt = (
        "You are an assistive drone helping a blind user. The image is the view "
        "you captured a moment ago, and this is the conversation so far:\n"
        f"{turns}\n"
        f"Answer this follow-up question about that view: \"{question}\"\n"
        "Reply in Turkish, in at most two short sentences. Be concrete and "
        "specific. If the image does not show what is being asked about, say so "
        "plainly rather than guessing."
    )

    contents = [prompt]
    if reference_img_path and os.path.exists(reference_img_path):
        contents.append(PIL.Image.open(reference_img_path))
    contents.append(cam)

    try:
        r = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        return r.text.strip()
    except Exception as e:
        return f"Soruyu cevaplayamadım: {str(e)[:100]}"
