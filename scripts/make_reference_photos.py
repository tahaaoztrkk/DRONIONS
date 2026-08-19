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
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, fresh, place,
                                      project_into_frame, settled_pose,
                                      world_control)

# name -> (x, y, z, true width, how far to stand off)
TARGETS = {
    'phone':  (2.62, -0.16, 1.019, 0.160, 0.55),
    'mug':    (2.62,  0.24, 1.060, 0.166, 0.55),
    'book':   (3.05, -0.22, 1.041, 0.210, 0.60),
    'laptop': (3.05,  0.22, 1.110, 0.375, 0.90),
}
OUT_DIR = 'memory'
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

    os.makedirs(OUT_DIR, exist_ok=True)
    focal = (1280 / 2.0) / math.tan(CAMERA_HFOV / 2.0)

    for name, (ox, oy, oz, width, standoff) in TARGETS.items():
        if a.only and name != a.only:
            continue
        # Look down at it from close, which is the view the drone will have
        # when it is deciding, and the view a user photographing the object
        # would not have. The second matters less than the first.
        x, y = ox - standoff, oy - 0.05
        z = oz + 0.55
        yaw = math.atan2(oy - y, ox - x)
        place(world, MODEL, x, y, z, yaw)
        img = fresh(node, g, world)
        if img is None:
            print(f"  {name:8} kare gelmedi")
            continue
        h_img, w_img = img.shape[:2]
        drone = settled_pose(pose) or (x, y, z)
        uv = project_into_frame((ox, oy, oz), drone, yaw, w_img, h_img)
        if uv is None or not (0 <= uv[0] < w_img and 0 <= uv[1] < h_img):
            print(f"  {name:8} kare disinda kaldi")
            continue
        rng = math.dist(drone, (ox, oy, oz))
        px = width * focal / rng
        half = max(24.0, px * (0.5 + PAD))
        x0, x1 = int(max(0, uv[0] - half)), int(min(w_img, uv[0] + half))
        y0, y1 = int(max(0, uv[1] - half)), int(min(h_img, uv[1] + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            print(f"  {name:8} kirpma bos")
            continue
        path = os.path.join(OUT_DIR, f"{name}.jpg")
        cv2.imwrite(path, crop)
        print(f"  {name:8} {crop.shape[1]}x{crop.shape[0]} px  "
              f"(nesne ~{px:.0f} px, {rng:.2f} m'den)  -> {path}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
