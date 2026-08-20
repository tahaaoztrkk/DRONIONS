#!/usr/bin/env python3
"""
Görevi: Görsel Hafıza Bankası için nesne referans fotoğrafları üretir.

The memory bank is what makes the task assistive rather than generic: it is how
the system tells *your* phone from any phone. It reads memory/<target>.jpg, uses
it for the colour gate and hands it to the model when picking between crops.
Only the box had one, so for every other target both mechanisms were switched
off -- and the logs said so on every flight, in a line that scrolled past.

It cost a real failure. The room's book is a scanned "Eat to Live" whose cover
is a bright blue image, and lying flat it looks exactly like a phone on a table.
The detector labelled both the book and the phone "phone", the book ranked
higher, and the model confirmed it as "the bright blue screen on the black
smartphone". Runs that looked like the drone losing the phone were the drone
correctly approaching the book.

A reference photo is what the user would have taken beforehand. Here the object
positions are known, so the crop is taken from ground truth rather than from a
detection -- a reference built out of a detection would inherit whatever the
detector was confused by, which in this case is the whole problem.

  scripts/make_reference_photos.py             # room world + bridge up
  scripts/make_reference_photos.py --only phone
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import rclpy

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation.spatial import CAMERA_HFOV
from perception.appearance import reference_signature
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, fresh, place,
                                      project_into_frame, settled_pose,
                                      world_control)

# name -> (x, y, z, true width, how far to stand off)
#
# The standoffs are the distances the colour gate is actually applied at, not
# the distance a nice photograph would be taken from. A close-up reference of
# the laptop is its screen; at search range the crop is the whole machine plus
# the table around it, and the two do not match on colour at all -- measured,
# the laptop's own reference rejected the laptop at every viewpoint tried. What
# the reference has to represent is the object as the gate will see it.
# Standoffs are a compromise, and worth naming as one. Far enough that the crop
# resembles what the gate will see during a search; close enough that the
# detector actually finds the object, which at 1.4 m it does not for the phone
# -- measured one viewpoint in six. A reference can only be made of something
# that was detected, so the near end wins where the two disagree, and the
# multiple angles carry the slack.
TARGETS = {
    'phone':  (2.62, -0.16, 1.019, 0.160, 0.95),
    'mug':    (2.62,  0.24, 1.060, 0.166, 1.00),
    'book':   (3.05, -0.22, 1.041, 0.210, 1.05),
    'laptop': (3.05,  0.22, 1.110, 0.375, 1.40),
}
OUT_DIR = 'memory'
# Bearings around the object, and how high above it to look from. One angle is
# not a description of an object whose appearance turns with the view: measured
# in the room, the laptop reads hue 17 from behind and 238 from the front, the
# mug 174 to 354 around its rim, and each was being rejected against its own
# single reference. Flat single-coloured things -- the phone, the book -- read
# the same from everywhere, so the extra angles cost them nothing.
VIEW_ANGLES = [(180, 0.70), (135, 0.65), (225, 0.65), (90, 0.60)]
TRY_SCALES = (1.0, 1.45, 0.75)
# Padding around the object, as a fraction of its own width.
#
# 0.45 was chosen to give the model some context and it destroyed the colour
# signature: for a small dark object on a wooden table the padding *is* the
# picture, and all four references came out at hue 19-32 -- the table, not the
# object. A gate comparing wood to wood cannot separate a black phone from a
# blue book, which is the one job it had here.
#
# Tight enough that the object dominates. The model still sees context, because
# the crops it is sent are padded separately in agent.py.
PAD = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='sadece bu nesne')
    a = ap.parse_args()

    world = find_world()
    if not world or 'room' not in world:
        sys.exit(f"Oda dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('reference_photos')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")

    det = YOLOWorldDetector()
    os.makedirs(OUT_DIR, exist_ok=True)
    focal = (1280 / 2.0) / math.tan(CAMERA_HFOV / 2.0)

    for name, (ox, oy, oz, width, standoff) in TARGETS.items():
        if a.only and name != a.only:
            continue
        # Several viewpoints, taking the first where the detector actually
        # boxes the object. One fixed view is a gamble: at the book's the
        # detector found nothing, the crop fell back to geometry centred on the
        # model's pose origin -- which for a book lying flat is its spine -- and
        # the reference came out as wooden table. A reference the detector
        # cannot recognise is also the wrong reference to be comparing against.
        det.set_target(name)
        shots = []
        # Two distances per bearing. Detection of these objects is noisy rather
        # than monotonic in range -- the book was found at 1.5 m and not at
        # 1.05 m from the same side -- so a single distance per angle throws
        # away angles for no reason.
        for bearing, lift in VIEW_ANGLES:
          got_this_angle = False
          for scale in TRY_SCALES:
            if got_this_angle:
                break
            ang = math.radians(bearing)
            dist = standoff * scale
            x, y = ox + dist * math.cos(ang), oy + dist * math.sin(ang)
            z = oz + lift
            yaw = math.atan2(oy - y, ox - x)
            place(world, MODEL, x, y, z, yaw)
            img = fresh(node, g, world)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            drone = settled_pose(pose) or (x, y, z)
            uv = project_into_frame((ox, oy, oz), drone, yaw, w_img, h_img)
            if uv is None or not (0 <= uv[0] < w_img and 0 <= uv[1] < h_img):
                continue
            rng0 = math.dist(drone, (ox, oy, oz))
            px0 = width * focal / rng0
            boxed = [c for c in filter_candidates(det.detect(img))
                     if c.bbox[0] <= uv[0] <= c.bbox[2]
                     and c.bbox[1] <= uv[1] <= c.bbox[3]
                     and 0.4 <= (c.bbox[2] - c.bbox[0]) / px0 <= 2.5]
            if boxed:
                shots.append((img, drone, uv,
                              max(boxed, key=lambda c: c.confidence), bearing))
                got_this_angle = True
        if not shots:
            print(f"  {name:8} hicbir acidan tespit edilemedi")
            continue
        for idx, (img, drone, uv, hitbox, bearing) in enumerate(shots, start=1):
            h_img, w_img = img.shape[:2]
            rng = math.dist(drone, (ox, oy, oz))
            px = width * focal / rng
            # Crop the detector's box, which frames what is actually
            # visible. Geometry alone centres on the model's pose origin, and
            # the book's origin sits on its spine, so that crop came out half
            # wooden table and read as wood.
            b = hitbox
            pad = (b.bbox[2] - b.bbox[0]) * PAD
            x0, x1 = int(max(0, b.bbox[0] - pad)), int(min(w_img, b.bbox[2] + pad))
            y0, y1 = int(max(0, b.bbox[1] - pad)), int(min(h_img, b.bbox[3] + pad))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            suffix = "" if idx == 1 else f"_{idx}"
            path = os.path.join(OUT_DIR, f"{name}{suffix}.jpg")
            cv2.imwrite(path, crop)
            sig = reference_signature(path)
            print(f"  {name:8} aci {bearing:4.0f}  {crop.shape[1]}x{crop.shape[0]} px  "
                  f"guven {b.confidence:.2f}  "
                  + (f"ton {sig[0]:.0f}" if sig else "renk yok") + f"  -> {path}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
