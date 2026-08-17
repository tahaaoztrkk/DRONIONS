#!/usr/bin/env python3
"""
Görevi: Belirli bir bakış noktasından dronun gerçekte ne gördüğünü kaydeder.

Written because the small-object sweep returned 0/9 beyond 7 m with the narrow
lens while reporting a 719 px "detection" where geometry predicts 120 -- two
symptoms that cannot both be true of a working rig. Every measurement rig in
this project has been optimistically wrong on first construction and the way
that keeps getting caught is looking at the actual frame rather than the
summary. This dumps the frame, every candidate, and where each one projects.

  scripts/debug_view.py 11.2            # 11.2 m, dead ahead
  scripts/debug_view.py 11.2 --bearing -40
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from navigation.spatial import CAMERA_HFOV, CAMERA_PITCH_DOWN
from gz_pose import PoseReader, find_world

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_small_objects import (Grab, MODEL, SIZE_HI, SIZE_LO, TARGET_NAME,
                                      TARGET_WIDTH, TARGET_XY, TARGET_Z, VIEW_ALT,
                                      expected_px, fresh, project_into_frame,
                                      teleport, world_control)

OUT_DIR = 'logs/debug_views'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('range_m', type=float)
    ap.add_argument('--bearing', type=float, default=0.0)
    a = ap.parse_args()

    world = find_world()
    if not world:
        sys.exit("Gazebo bulunamadi.")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('debug_view')
    g = Grab(node)

    ang = math.radians(a.bearing)
    dx, dy = a.range_m * math.cos(ang), a.range_m * math.sin(ang)
    x, y = TARGET_XY[0] + dx, TARGET_XY[1] + dy
    yaw = math.atan2(-dy, -dx)
    world_control(world, 'pause: true')
    teleport(world, MODEL, x, y, VIEW_ALT, yaw)
    img = fresh(node, g, world)
    if img is None:
        sys.exit("Kare gelmedi.")

    h, w = img.shape[:2]
    vfov = 2 * math.atan(math.tan(CAMERA_HFOV / 2) * h / w)
    # Where the target sits in the frame, from geometry alone. If this is off
    # the bottom or top, the sweep was never looking at the box and no amount
    # of detector tuning would have shown it.
    depression = math.atan2(VIEW_ALT, a.range_m)
    off = depression - CAMERA_PITCH_DOWN
    row = h / 2 + (off / (vfov / 2)) * (h / 2)

    print(f"lens {math.degrees(CAMERA_HFOV):.0f} deg yatay, "
          f"{math.degrees(vfov):.0f} deg dikey; kare {w}x{h}")
    print(f"kamera egimi {math.degrees(CAMERA_PITCH_DOWN):.0f} deg asagi")
    print(f"hedefin bakis acisi {math.degrees(depression):.1f} deg asagi "
          f"-> merkezden {math.degrees(off):+.1f} deg, satir ~{row:.0f}/{h}")
    if not 0 <= row <= h:
        print("  !! hedef karenin DISINDA -- olculen sey kutu degil")
    print(f"beklenen genislik {expected_px(TARGET_WIDTH, a.range_m, VIEW_ALT, w):.0f} px\n")

    det = YOLOWorldDetector()
    det.set_target(TARGET_NAME)
    cands = filter_candidates(det.detect(img))
    drone_xyz = pose.latest() or (x, y, VIEW_ALT)
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))

    uv = project_into_frame((TARGET_XY[0], TARGET_XY[1], TARGET_Z),
                            drone_xyz, yaw, w, h)
    exp_w = expected_px(TARGET_WIDTH, a.range_m, VIEW_ALT, w)
    print(f"kutunun karedeki yeri: {uv[0]:.0f}, {uv[1]:.0f}\n" if uv
          else "kutu kare disinda\n")

    print(f"{len(cands)} aday:")
    for c in cands:
        x0, y0, x1, y1 = c.bbox
        bw = x1 - x0
        covers = uv is not None and x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1
        ratio = bw / exp_w
        ok = covers and SIZE_LO <= ratio <= SIZE_HI
        why = 'SAYILDI' if ok else ('boyut %.1fx' % ratio if covers else 'yeri tutmuyor')
        print(f"  {c.label:14} guven {c.confidence:.2f}  {bw:4.0f} px  {why}")
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 0) if ok else (0, 165, 255), 2)
    if uv:
        cv2.drawMarker(img, (int(uv[0]), int(uv[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 40, 2)


    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/r{a.range_m:.1f}_b{a.bearing:+.0f}_{round(math.degrees(CAMERA_HFOV))}deg.jpg"
    cv2.imwrite(path, img)
    print(f"\n-> {path}  (kirmizi hac: kutunun olmasi gereken yer)")

    print(f"dronun gercek irtifasi: {drone_xyz[2]:.2f} m (istenen {VIEW_ALT})")
    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
