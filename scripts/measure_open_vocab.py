#!/usr/bin/env python3
"""
Görevi: Açık sözlüğün korunup korunmadığını, eğitimde görülmemiş nesnelerle ölçer.

This exists for one decision: whether to fine-tune YOLO-World itself on the
room's objects instead of running a separate closed-set model beside it.

The trap in that decision is that the obvious measurement confirms itself. A
YOLO-World fine-tuned on laptop, book, mug and phone will of course find
laptops, books, mugs and phones -- and the closed-set model already reaches
36/36 on those, so matching it proves nothing. The only reason to fine-tune the
open model is to carry one model instead of two, and the only thing that can go
wrong is that it stops being open. A user saying "find my charger" has to get
an attempt, without anyone retraining anything; that property is the project's
premise, and it is not visible in any measurement taken on the trained classes.

So this measures the other set: objects standing in the same room, under the
same light, that the training data never contained. Recall on them before and
after is the acceptance criterion. Everything else is a tie-breaker.

Scoring is deliberately identical to compare_detectors.py -- a detection counts
only if its box covers where the object truly is, at roughly the width geometry
predicts -- so the two numbers can be read side by side.

  scripts/measure_open_vocab.py                       # stock YOLO-World
  scripts/measure_open_vocab.py --weights runs/detect/world_ft/weights/best.pt
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

from navigation.spatial import CAMERA_HFOV
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, fresh, place, placed_ok,
                                      project_into_frame, settled_pose,
                                      world_control)

# Objects the training set never contained, with their true centres and the
# largest horizontal extent a box of them should show.
#
# Read off px4/worlds/dronions_room.sdf. Furniture is included deliberately:
# "find the sofa" is exactly the kind of request the open vocabulary is for,
# and furniture is also what the surface scan depends on -- if fine-tuning
# costs the open model its furniture, it costs the search its surfaces too.
HELD_OUT = {
    "bowl":       ((2.18,  0.20, 1.015), 0.16, ["bowl"]),
    "headphones": ((0.50, -1.20, 0.580), 0.20, ["headphones", "headset"]),
    "bottle":     ((1.40, -0.60, 0.100), 0.08, ["bottle", "water bottle"]),
    "chair":      ((1.50,  0.90, 0.450), 0.55, ["chair"]),
    "table":      ((2.60,  0.00, 0.750), 1.10, ["table", "desk"]),
    "sofa":       ((0.50, -1.35, 0.450), 1.80, ["sofa", "couch"]),
    "bookshelf":  ((-1.20, 0.80, 1.000), 0.90, ["bookshelf", "shelf"]),
    "cabinet":    ((1.20,  1.60, 0.900), 0.90, ["cabinet", "sideboard"]),
}
# The four the training set did contain, kept so both halves of the trade are
# on one page rather than in two runs a day apart.
TRAINED_ON = {
    "laptop": ((3.05,  0.22, 1.112), 0.375, ["laptop"]),
    "book":   ((3.05, -0.22, 1.041), 0.210, ["book"]),
    "mug":    ((2.62,  0.24, 1.060), 0.166, ["mug", "cup"]),
    "phone":  ((2.62, -0.16, 1.019), 0.160, ["phone", "cell phone"]),
}

# The altitudes and spread the sweep actually uses, not a close-up tour.
VIEWS = [(0.0, 0.0, 2.0), (0.5, -0.5, 2.0), (0.8, 0.5, 2.0),
         (1.2, -0.8, 1.9), (1.5, 0.3, 1.9), (0.3, 0.9, 2.0),
         (1.8, -0.4, 1.8), (0.9, 1.0, 1.8)]
SIZE_LO, SIZE_HI = 0.4, 2.5
OUT_CSV = 'logs/open_vocab.csv'


def counts_as(box, uv, expected_px) -> bool:
    """Same rule as compare_detectors: covers the truth, at about the right size."""
    x0, y0, x1, y1 = box
    if not (x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1):
        return False
    return SIZE_LO <= (x1 - x0) / expected_px <= SIZE_HI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8s-world.pt',
                    help='olculecek YOLO-World agirliklari')
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--trained-too', action='store_true',
                    help='egitimde bulunan dort nesneyi de olc')
    a = ap.parse_args()

    world = find_world()
    if not world or 'room' not in world:
        sys.exit(f"Oda dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('open_vocab')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")

    from ultralytics import YOLOWorld
    model = YOLOWorld(a.weights)
    focal = (1280 / 2.0) / math.tan(CAMERA_HFOV / 2.0)

    groups = [("gorulmemis", HELD_OUT)]
    if a.trained_too:
        groups.append(("egitimde", TRAINED_ON))

    stats, skipped = {}, 0
    for _, table in groups:
        for name in table:
            stats[name] = {"seen": 0, "hit": 0}

    for view in VIEWS:
        for group, table in groups:
            for name, (centre, width, prompts) in table.items():
                ox, oy, oz = centre
                yaw = math.atan2(oy - view[1], ox - view[0])
                place(world, MODEL, *view, yaw)
                img = fresh(node, g, world)
                drone = settled_pose(pose)
                quat = pose.latest_quat()
                # A viewpoint the airframe could not actually occupy measures a
                # different, tilted viewpoint without saying so.
                if img is None or not placed_ok(drone, view, quat):
                    skipped += 1
                    continue
                h_img, w_img = img.shape[:2]
                uv = project_into_frame(centre, drone, yaw, w_img, h_img)
                if uv is None or not (0 <= uv[0] < w_img and 0 <= uv[1] < h_img):
                    continue
                rng = math.dist(drone, centre)
                exp_px = width * focal / rng
                stats[name]["seen"] += 1

                model.set_classes(prompts)
                res = model.predict(img, conf=a.conf, verbose=False)[0]
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for xyxy in boxes.xyxy.tolist():
                    if counts_as(xyxy, uv, exp_px):
                        stats[name]["hit"] += 1
                        break

    print(f"\nagirliklar: {a.weights}")
    if skipped:
        print(f"{skipped} ornek atlandi (bakis noktasina varilamadi)")
    for group, table in groups:
        rows = [(n, stats[n]) for n in table if stats[n]["seen"]]
        if not rows:
            continue
        print(f"\n  --- {group} ---")
        for name, s in rows:
            print(f"    {name:12} {s['hit']}/{s['seen']}")
        seen = sum(s["seen"] for _, s in rows)
        hit = sum(s["hit"] for _, s in rows)
        print(f"    {'TOPLAM':12} {hit}/{seen}  (%{100 * hit / seen:.0f})")

    os.makedirs('logs', exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=["weights", "object", "seen", "hit"])
        w.writeheader()
        for name, s in stats.items():
            w.writerow({"weights": a.weights, "object": name, **s})
    print(f"\n-> {OUT_CSV}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
