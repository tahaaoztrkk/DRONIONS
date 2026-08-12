"""
Görevi: Bir tespitin rengini, hafıza bankasındaki referans fotoğrafla karşılaştırır.

Why colour, and why here: the physical-size gate rejects anything that cannot
be the target's size, which removed every wall detection in the survey data.
It cannot help with objects that *are* the right size -- and the scenario has
two: a blue box of almost identical dimensions, and a 0.6 m sphere. Measured
across ten approach runs, those two accounted for four of the nine locks, and
Gemini approved the blue one when asked.

Shape would be the obvious answer and is the harder one: a bounding box says
almost nothing about form, and aspect ratio changes with viewing angle. Colour
is cheap, viewpoint-independent, and separates exactly the cases that are
failing -- tan against blue, green and red.

This runs before the VLM, so a rejected candidate costs no API call. It is
deliberately permissive: it answers "could this be the right colour", not
"is this the object". Deciding is still the model's job.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np

# How far the hue may differ from the reference before a candidate is rejected.
# Generous: sim lighting shifts hue noticeably across the scene, and the point
# is to exclude a different colour, not to grade a match.
MAX_HUE_DEGREES = 45.0

# Pixels below these are treated as colourless and ignored when averaging.
# Shadowed and washed-out pixels carry no reliable hue, and a bounding box is
# full of both around the object's edges.
MIN_SATURATION = 40
MIN_VALUE = 40

# Fraction of the box kept when sampling. A detection box always includes some
# background at its edges; the centre is the part most likely to be the object.
CENTRE_FRACTION = 0.6

# Below this many usable pixels the sample is not worth an opinion.
MIN_USABLE_PIXELS = 30


def _circular_mean_hue(hue_deg: np.ndarray) -> float:
    """Mean of an angular quantity. A plain average is wrong for hue: red sits
    at both 0 and 360, so averaging two reds either side of the wrap gives
    cyan."""
    radians = np.deg2rad(hue_deg.astype(np.float64))
    return math.degrees(math.atan2(np.sin(radians).mean(),
                                   np.cos(radians).mean())) % 360.0


def colour_signature(bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """(mean hue in degrees, median saturation), or None if there is no colour.

    OpenCV packs hue into 0-179 to fit a byte; it is doubled here so the number
    means what it says.
    """
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(np.float32) * 2.0, hsv[:, :, 1], hsv[:, :, 2]
    usable = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    if int(usable.sum()) < MIN_USABLE_PIXELS:
        return None
    return _circular_mean_hue(h[usable]), float(np.median(s[usable]))


def _centre_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    mx = (x2 - x1) * (1.0 - CENTRE_FRACTION) / 2.0
    my = (y2 - y1) * (1.0 - CENTRE_FRACTION) / 2.0
    cx1, cy1 = int(x1 + mx), int(y1 + my)
    cx2, cy2 = int(x2 - mx), int(y2 - my)
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    return frame[cy1:cy2, cx1:cx2]


def hue_distance(a: float, b: float) -> float:
    """Shortest way round the colour wheel, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def reference_signature(reference_img_path: str) -> Optional[Tuple[float, float]]:
    """Colour signature of the memory-bank photo, or None if unusable."""
    if not reference_img_path or not os.path.exists(reference_img_path):
        return None
    img = cv2.imread(reference_img_path)
    if img is None:
        return None
    return colour_signature(_centre_crop(img, (0, 0, img.shape[1], img.shape[0])))


def colour_plausible(frame: np.ndarray, candidate, ref_sig,
                     max_hue: float = MAX_HUE_DEGREES) -> bool:
    """Could this detection be the reference object's colour?

    True whenever there is no opinion to give -- no reference on file, no
    usable colour in either sample. Every gate in this pipeline abstains rather
    than guesses, because the cost of wrongly rejecting the target is a search
    that never succeeds, while the cost of letting one extra candidate through
    is one API call.
    """
    if not ref_sig:
        return True
    crop = _centre_crop(frame, candidate.bbox)
    sig = colour_signature(crop) if crop is not None else None
    if not sig:
        return True
    return hue_distance(sig[0], ref_sig[0]) <= max_hue
