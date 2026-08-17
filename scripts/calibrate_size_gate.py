#!/usr/bin/env python3
"""
Görevi: Boyut kapısının eşiğini gerçek tespitlerle kalibre eder.

size_plausible only ever asked about the floor, and that rejected a laptop on a
table three times in one flight -- each time after the model had confirmed it,
because a ray through the box's bottom edge passes over the table and lands on
the floor well beyond, inflating the range and the width with it.

Letting it ask about raised surfaces too fixes the laptop and weakens the gate,
which exists to reject the scenario wall. Whether it still does is a question
about real detections at real viewpoints, not about arithmetic on an imagined
bounding box, so this measures both populations in the same sweep: every
detection is classified as the box or not-the-box by projecting the box's known
position into the frame, and its implied width recorded on every candidate
support plane. The separation between the two populations is what a threshold
can be set from.

  scripts/calibrate_size_gate.py                # scenario world + bridge up
  scripts/calibrate_size_gate.py --samples 2
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
from navigation.spatial import (OBJECT_WIDTHS, SUPPORT_HEIGHTS, implied_width)
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, TARGET_XY, assert_lens_matches,
                                      fresh, place, project_into_frame,
                                      settled_pose, world_control)

TARGET = 'box'
TARGET_Z = 0.30
RANGES = [2.0, 3.0, 4.0, 5.0, 6.0]
BEARINGS = [-60.0, -30.0, 0.0, 30.0, 60.0]
VIEW_ALT = 2.0
OUT_CSV = 'logs/size_gate_calibration.csv'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=2)
    a = ap.parse_args()

    world = find_world()
    if not world or 'scenario' not in world:
        sys.exit(f"Senaryo dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('size_gate')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")
    assert_lens_matches(node, 1280)

    det = YOLOWorldDetector()
    det.set_target(TARGET)
    known = OBJECT_WIDTHS[TARGET]

    rows = []
    for rng in RANGES:
        for bearing in BEARINGS:
            for _ in range(a.samples):
                ang = math.radians(bearing)
                dx, dy = rng * math.cos(ang), rng * math.sin(ang)
                x, y = TARGET_XY[0] + dx, TARGET_XY[1] + dy
                yaw = math.atan2(-dy, -dx)
                place(world, MODEL, x, y, VIEW_ALT, yaw)
                img = fresh(node, g, world)
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]
                drone = settled_pose(pose) or (x, y, VIEW_ALT)
                quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
                uv = project_into_frame((TARGET_XY[0], TARGET_XY[1], TARGET_Z),
                                        drone, yaw, w_img, h_img)
                slant = math.hypot(rng, VIEW_ALT - TARGET_Z)
                exp_px = 0.63 * (w_img / 2) / math.tan(1.7453 / 2) / slant
                for c in filter_candidates(det.detect(img)):
                    x0, y0, x1, y1 = c.bbox
                    covers = uv is not None and x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1
                    # The box, not merely something covering it: the wall sits
                    # directly behind and covers the same pixels at every range.
                    is_box = covers and 0.4 <= (x1 - x0) / exp_px <= 2.5
                    row = {'range_m': rng, 'bearing_deg': bearing,
                           'is_box': int(is_box), 'conf': round(c.confidence, 3),
                           'px': round(x1 - x0, 1)}
                    best = None
                    for z in SUPPORT_HEIGHTS:
                        w = implied_width(c, drone, quat, plane_z=z)
                        row[f'w_{z}'] = round(w, 3) if w else ''
                        if w is not None:
                            r = w / known
                            if best is None or abs(math.log(r)) < abs(math.log(best)):
                                best = r
                    row['best_ratio'] = round(best, 3) if best else ''
                    rows.append(row)

    os.makedirs('logs', exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def ratios(flag):
        return sorted(r['best_ratio'] for r in rows
                      if r['is_box'] == flag and r['best_ratio'] != '')

    box, other = ratios(1), ratios(0)
    print(f"\n{len(rows)} tespit: {len(box)} kutu, {len(other)} kutu degil\n")
    for label, vals in (('kutu', box), ('kutu degil (cogunlukla duvar)', other)):
        if not vals:
            print(f"  {label}: yok")
            continue
        n = len(vals)
        print(f"  {label:32} n={n:3}  min {vals[0]:.2f}  "
              f"medyan {vals[n // 2]:.2f}  %90 {vals[int(n * 0.9)]:.2f}  "
              f"max {vals[-1]:.2f}")
    if box and other:
        print(f"\n  kutunun en yuksegi   {max(box):.2f}")
        print(f"  duvarin en dusugu    {min(other):.2f}")
        print("  -> esik bu ikisinin arasinda olmali"
              if max(box) < min(other) else
              "  -> populasyonlar ortusuyor, tek esik ayiramaz")
    print(f"\n-> {OUT_CSV}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
