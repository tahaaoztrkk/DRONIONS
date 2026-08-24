#!/usr/bin/env python3
"""
Görevi: Hedefin hangi yüzeyde durduğu çıkarımını gerçek tespitlerle kalibre eder.

locate_target picks the surface a target is standing on by asking which plane
puts it at the distance its apparent size implies, and only leaves the floor
when a raised plane fits FLOOR_SWITCH_MARGIN times better. That margin was
never measured against objects genuinely on a table; it was set from the
scenario world, where everything is on the floor and the only question was how
often a wide detector box would wrongly lift something.

In the room it is wrong the other way. Measured in flight: the laptop, on a
1.015 m table, was placed at (3.75, -0.12) against a true (3.05, 0.22) and
announced as "on the floor" -- in the same sentence as the model's own
description of it sitting on a wooden table. Being 0.8 m out also turns the
table into an obstacle standing between the drone and a target it believes is
on the ground behind it, which is what then refused the approach.

This measures both error directions at once. For every detection of an object
whose true position is known, it records the distance implied by apparent size,
the distance to each candidate plane, and the error the current margin produces
against the error each alternative margin would have produced.

  scripts/calibrate_support_plane.py            # room world + bridge up
  scripts/calibrate_support_plane.py --samples 3
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
from navigation.spatial import (SUPPORT_HEIGHTS, camera_origin_world,
                                camera_ray_world, project_to_plane,
                                range_from_apparent_size)
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, assert_lens_matches, fresh,
                                      place, project_into_frame, settled_pose,
                                      world_control)

# name -> (x, y, z of the object, the surface it rests on, prompt)
TARGETS = [
    ('laptop', 3.05, 0.22, 1.11, 1.015, 'laptop'),
    ('book', 3.05, -0.22, 1.07, 1.015, 'book'),
    ('mug', 2.62, 0.24, 1.06, 1.015, 'mug'),
    ('floor_box', 0.10, 1.10, 0.15, 0.0, 'box'),
]

# Where the drone looks from. Spread over the room at the altitudes the search
# and the approach actually use, since the plane choice depends on how oblique
# the view is and that is what changes across an approach.
VIEWS = [(0.0, 0.0, 2.0), (0.6, -0.6, 2.0), (1.2, 0.4, 1.8),
         (1.7, -0.9, 1.6), (0.2, 1.0, 2.0), (1.4, 1.2, 1.8),
         (0.8, -1.2, 1.5), (2.0, 0.6, 1.5)]

OUT_CSV = 'logs/support_plane_calibration.csv'


def plane_hits(candidate, drone_xyz, quat):
    """Distance to each candidate support plane along the bbox-bottom ray."""
    u, v = candidate.normalized_center
    if getattr(candidate, 'image_height', 0):
        v = candidate.bbox[3] / candidate.image_height
    ray = camera_ray_world(u, v, quat)
    # From the lens, not from base_link -- see CAMERA_MOUNT. Casting from the
    # pose origin is what made this calibration's plane answers read short.
    drone_xyz = camera_origin_world(drone_xyz, quat)
    out = {}
    for z in SUPPORT_HEIGHTS:
        hit = project_to_plane(drone_xyz, ray, z)
        out[z] = (hit, math.dist(hit, drone_xyz)) if hit else (None, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=2)
    a = ap.parse_args()

    world = find_world()
    if not world or 'room' not in world:
        sys.exit(f"Oda dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('support_plane')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")
    assert_lens_matches(node, 1280)

    det = YOLOWorldDetector()
    rows = []

    for name, ox, oy, oz, surface, prompt in TARGETS:
        det.set_target(prompt)
        for vx, vy, vz in VIEWS:
            for _ in range(a.samples):
                yaw = math.atan2(oy - vy, ox - vx)
                place(world, MODEL, vx, vy, vz, yaw)
                img = fresh(node, g, world)
                if img is None:
                    continue
                h, w = img.shape[:2]
                drone = settled_pose(pose) or (vx, vy, vz)
                quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
                uv = project_into_frame((ox, oy, oz), drone, yaw, w, h)
                if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                    continue
                for c in filter_candidates(det.detect(img)):
                    x0, y0, x1, y1 = c.bbox
                    if not (x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1):
                        continue
                    expected = range_from_apparent_size(c, prompt)
                    if expected is None:
                        continue
                    hits = plane_hits(c, drone, quat)
                    row = {'object': name, 'surface': surface,
                           'expected_range': round(expected, 3)}
                    for z, (hit, rng) in hits.items():
                        row[f'r_{z}'] = round(rng, 3) if rng else ''
                        row[f'e_{z}'] = (round(math.dist(hit[:2], (ox, oy)), 3)
                                         if hit else '')
                    rows.append(row)
                    break

    if not rows:
        sys.exit("Hicbir tespit toplanamadi.")
    os.makedirs('logs', exist_ok=True)
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ('object', 'surface', 'expected_range'), k))
    with open(OUT_CSV, "w", newline="") as op:
        wtr = csv.DictWriter(op, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)

    # What each margin would have chosen, scored against the truth.
    print(f"\n{len(rows)} tespit\n")
    print(f"{'esik':>6} {'dogru yuzey':>13} {'ortanca hata':>14} {'>0.5 m sapan':>14}")
    print('-' * 52)
    for margin in (1.0, 1.5, 2.0, 3.0, 6.0, 12.0):
        right = 0
        errs = []
        for r in rows:
            exp = r['expected_range']
            best_z, best_gap = None, None
            for z in SUPPORT_HEIGHTS[1:]:
                if r.get(f'r_{z}') == '' or r.get(f'r_{z}') is None:
                    continue
                gap = abs(r[f'r_{z}'] - exp)
                if best_gap is None or gap < best_gap:
                    best_z, best_gap = z, gap
            floor_r = r.get('r_0.0')
            floor_gap = abs(floor_r - exp) if floor_r not in ('', None) else None
            if best_z is None or floor_gap is None:
                chosen = 0.0 if floor_gap is not None else best_z
            else:
                chosen = best_z if best_gap * margin < floor_gap else 0.0
            if chosen is None:
                continue
            right += (abs(chosen - r['surface']) < 0.01)
            e = r.get(f'e_{chosen}')
            if e not in ('', None):
                errs.append(e)
        errs.sort()
        med = errs[len(errs) // 2] if errs else float('nan')
        bad = sum(1 for e in errs if e > 0.5) / len(errs) * 100 if errs else 0
        mark = '  <- simdiki' if margin == 12.0 else ''
        print(f"{margin:6.1f} {right / len(rows) * 100:12.0f}% "
              f"{med:13.2f}m {bad:13.0f}%{mark}")
    print(f"\n-> {OUT_CSV}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
