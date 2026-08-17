#!/usr/bin/env python3
"""
Görevi: Masadaki küçük nesnelerin hangi boyutta tespit edilebildiğini ölçer.

Two questions the scenario world cannot answer.

First: is the measured limit about size or about clutter? There the target
stands in front of a 3 m wall and past 2 m the detector returns the wall at
every range, so "the box stops being found beyond 2 m" conflates the two. Here
there is no wall. Scoring the same apparent widths in a clean scene separates
them: if a 60 px book is found here and a 60 px box was not found there, the
wall was the limit and no amount of camera work would have helped.

Second: does a table make a cup or a phone reachable? A cup on the floor needs
the drone half a metre away, closer than the airframe is wide. A table lifts it
0.75 m -- worth more than either lens change, both of which were measured and
neither of which helped.

Each object is scored on its own: its known position is projected into the
frame and a detection must cover it at roughly the predicted width. Ranking or
frame position alone would credit the table for the phone.

  scripts/experiment_tabletop.py                # needs sim + bridge on this world
  scripts/experiment_tabletop.py --samples 5
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import rclpy

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from navigation.spatial import CAMERA_HFOV, CAMERA_PITCH_DOWN
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, SIZE_HI, SIZE_LO,
                                      assert_lens_matches, fresh, place,
                                      project_into_frame, settled_pose,
                                      world_control)

TABLE_TOP = 1.015   # openrobotics/table slab top

# name -> (x, y, height above the table top, true width in metres, YOLO prompt)
# Widths are what the scoring compares against, so they are the modelled sizes
# and not a rounded-off idea of them.
# Widths are measured from the mesh vertices, not estimated: they are what a
# detection's size is checked against, so a guess here would quietly decide
# which detections count.
TARGETS = [
    ('laptop', 3.35,  0.20, 0.097, 0.375, 'laptop'),
    ('book',   3.35, -0.20, 0.026, 0.210, 'book'),
    ('mug',    2.70,  0.20, 0.048, 0.166, 'mug'),
    ('phone',  2.70, -0.20, 0.004, 0.160, 'phone'),
]

# Slant distances from the *camera* to the object. The near end is set by
# clearance rather than by optics: the drone has to stay 0.6 m above the table,
# and since the camera rides 0.24 m above the drone, that puts the closest
# flyable look at about 1.1 m. The two shorter entries are below that and are
# reported as unflyable -- kept because they answer a different question, what
# the detector could do if the airframe allowed it.
SLANTS = [0.7, 0.9, 1.1, 1.4, 1.8, 2.3, 3.0]

# Depression angle from the object up to the camera. The camera is fixed 20 deg
# down with a 42 deg half-height, so the frame reaches 62 deg below horizontal
# and 50 deg needs no camera change at all -- the object only has to be in
# frame, not on the boresight, which the first arithmetic here got wrong and
# concluded a much steeper camera was required.
#
# Measuring from the camera and not the drone matters: at 55 deg nominal the
# 0.24 m mount made the true depression 61 deg, right at the frame edge, and
# every object projected just past the bottom of the image. Every score was a
# miss and nothing said so.
DEPRESSION = 50.0
CAM_Z = 0.242           # camera mount height above the drone origin
TABLE_CLEARANCE = 0.60  # the drone, not the camera, has to keep this

BEARINGS = [-50.0, 0.0, 50.0]
OUT_CSV = 'logs/tabletop.csv'


def viewpoint(obj_xy, obj_z, slant, depression_deg, bearing_deg):
    """Drone pose putting the *camera* at the given slant and depression."""
    th = math.radians(depression_deg)
    r = slant * math.cos(th)
    ang = math.radians(bearing_deg)
    x = obj_xy[0] + r * math.cos(ang)
    y = obj_xy[1] + r * math.sin(ang)
    z = obj_z + slant * math.sin(th) - CAM_Z
    return x, y, z, math.atan2(obj_xy[1] - y, obj_xy[0] - x)


def clearance_for(slant, depression_deg, obj_z):
    """How far the drone body ends up above the table at that viewpoint."""
    return obj_z + slant * math.sin(math.radians(depression_deg)) - CAM_Z - TABLE_TOP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=3)
    a = ap.parse_args()

    world = find_world()
    if not world:
        sys.exit("Gazebo bulunamadi.")
    if 'tabletop' not in world:
        sys.exit(f"Yanlis dunya: {world}. Masa dunyasiyla baslatin:\n"
                 "  make px4_sitl gz_x500_dronions_dronions_tabletop")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('tabletop')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("/camera/image_raw'dan kare yok -- kopru calisiyor mu?")
    assert_lens_matches(node, 1280)

    det = YOLOWorldDetector()
    focal = (1280 / 2.0) / math.tan(CAMERA_HFOV / 2.0)

    print(f"masa ustu {TABLE_TOP} m, bakis acisi {DEPRESSION:.0f} derece asagi")
    print(f"{len(TARGETS)} nesne x {len(SLANTS)} mesafe x {len(BEARINGS)} yon "
          f"x {a.samples} tekrar\n")

    rows = []
    for name, ox, oy, dz, width, prompt in TARGETS:
        det.set_target(prompt)
        obj_z = TABLE_TOP + dz
        print(f"{name} ({width * 100:.1f} cm, '{prompt}')")
        print(f"{'mesafe':>8} {'gorunen':>9} {'tespit':>9} {'olculen':>9}")
        for slant in SLANTS:
            hits = trials = 0
            widths = []
            for bearing in BEARINGS:
                for _ in range(a.samples):
                    x, y, z, yaw = viewpoint((ox, oy), obj_z, slant,
                                             DEPRESSION, bearing)
                    place(world, MODEL, x, y, z, yaw)
                    img = fresh(node, g, world)
                    trials += 1
                    if img is None:
                        continue
                    h_img, w_img = img.shape[:2]
                    drone = settled_pose(pose) or (x, y, z)
                    uv = project_into_frame((ox, oy, obj_z), drone, yaw,
                                            w_img, h_img)
                    if uv is None or not (0 <= uv[0] < w_img and 0 <= uv[1] < h_img):
                        continue
                    # measured slant, not commanded: the drone settles a few
                    # centimetres low and the predicted width has to follow it
                    true_slant = math.dist(drone, (ox, oy, obj_z))
                    exp_w = width * focal / true_slant
                    for c in filter_candidates(det.detect(img)):
                        x0, y0, x1, y1 = c.bbox
                        if not (x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1):
                            continue
                        if not SIZE_LO <= (x1 - x0) / exp_w <= SIZE_HI:
                            continue
                        hits += 1
                        widths.append(x1 - x0)
                        break
            exp_w = width * focal / slant
            got = sum(widths) / len(widths) if widths else None
            clear = clearance_for(slant, DEPRESSION, obj_z)
            rows.append({'object': name, 'width_m': width, 'prompt': prompt,
                         'slant_m': slant, 'clearance_m': round(clear, 2),
                         'flyable': int(clear >= TABLE_CLEARANCE),
                         'expected_px': round(exp_w, 1),
                         'trials': trials, 'hits': hits,
                         'recall': round(hits / trials, 3) if trials else 0,
                         'measured_px': round(got, 1) if got else ''})
            print(f"{slant:7.1f}m {exp_w:8.0f} {hits:6}/{trials:<3} "
                  f"{(f'{got:.0f}' if got else '--'):>9}"
                  f"{'' if clear >= TABLE_CLEARANCE else '   (ucusa uygun degil)'}")
        print()

    os.makedirs('logs', exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT_CSV}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
