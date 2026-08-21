#!/usr/bin/env python3
"""
Görevi: Açık sözlüklü ve göreve özel dedektörleri aynı bakış noktalarında ölçer.

The comparison has to be at the viewpoints the search actually flies from, not
at the ones a demo would choose. Everything measured close up in this project
looked better than it turned out to be in flight, three separate times, because
the measuring was done where the object was easy to see and the flying happens
where it is not.

So this walks the same grid for both detectors and scores a detection the same
way in both cases: it counts only if it covers where the object truly is, at
roughly the width geometry predicts. Ranking or frame position alone would
credit the table for the mug, and did.

The result is a trade, not a winner. A closed-set model can only find what it
was shown, and the project's premise is that a user says what they want in
their own words. What this produces is the number that makes that trade
arguable: how much recall the open vocabulary costs, on these objects, in this
room.

  scripts/compare_detectors.py --weights runs/detect/room/weights/best.pt
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
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, fresh, place,
                                      project_into_frame, settled_pose,
                                      world_control)
from make_dataset import OBJECTS, CLASSES

# The altitudes and spread the sweep actually uses, not a close-up tour.
VIEWS = [(0.0, 0.0, 2.0), (0.5, -0.5, 2.0), (0.8, 0.5, 2.0),
         (1.2, -0.8, 2.0), (1.5, 0.3, 2.0), (0.3, 0.9, 2.0),
         (1.8, -0.4, 1.8), (0.9, 1.0, 1.8)]
SIZE_LO, SIZE_HI = 0.4, 2.5
OUT_CSV = 'logs/detector_comparison.csv'


def counts_as(box, uv, expected_px):
    """Same rule for both detectors: covers the truth, at about the right size."""
    x0, y0, x1, y1 = box
    if not (x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1):
        return False
    return SIZE_LO <= (x1 - x0) / expected_px <= SIZE_HI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/detect/room/weights/best.pt')
    ap.add_argument('--conf', type=float, default=0.15)
    a = ap.parse_args()

    trained = None
    if os.path.exists(a.weights):
        from ultralytics import YOLO
        trained = YOLO(a.weights)
    else:
        print(f"Egitilmis model yok ({a.weights}) -- sadece YOLO-World olculuyor.")

    world = find_world()
    if not world or 'room' not in world:
        sys.exit(f"Oda dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('compare')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")

    world_det = YOLOWorldDetector()
    focal = (1280 / 2.0) / math.tan(CAMERA_HFOV / 2.0)
    # Largest horizontal extent, which is what a box's width shows.
    widths = {name: max(half[0], half[1]) * 2 for name, _, half in OBJECTS}
    centres = {name: centre for name, centre, _ in OBJECTS}

    stats = {c: {"seen": 0, "world": 0, "trained": 0} for c in CLASSES}
    # Confidences of the detections that were actually right, per model.
    #
    # A hybrid has to rank candidates from both, and the ranking is confidence
    # alone. Two models' confidences are not the same quantity -- different
    # training objectives, different calibration -- so merging them naively
    # lets whichever scores higher win systematically, for no reason connected
    # to being right. Collected here because it costs nothing on a run that is
    # already putting both models on the same frames, and guessing at it would
    # be the fourth time in this project that an unmeasured assumption decided
    # something.
    confs = {"world": [], "trained": []}
    frames = []

    for vx, vy, vz in VIEWS:
        for name in CLASSES:
            ox, oy, oz = centres[name]
            yaw = math.atan2(oy - vy, ox - vx)
            place(world, MODEL, vx, vy, vz, yaw)
            img = fresh(node, g, world)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            drone = settled_pose(pose) or (vx, vy, vz)
            uv = project_into_frame((ox, oy, oz), drone, yaw, w_img, h_img)
            if uv is None or not (0 <= uv[0] < w_img and 0 <= uv[1] < h_img):
                continue
            rng = math.dist(drone, (ox, oy, oz))
            exp_px = widths[name] * focal / rng
            stats[name]["seen"] += 1

            world_det.set_target(name)
            for c in filter_candidates(world_det.detect(img)):
                if counts_as(c.bbox, uv, exp_px):
                    stats[name]["world"] += 1
                    confs["world"].append(c.confidence)
                    break

            if trained is not None:
                res = trained.predict(img, conf=a.conf, verbose=False)[0]
                idx = CLASSES.index(name)
                for b, cls in zip(res.boxes.xyxy.tolist(),
                                  res.boxes.cls.tolist()):
                    if int(cls) == idx and counts_as(b, uv, exp_px):
                        stats[name]["trained"] += 1
                        confs["trained"].append(
                            res.boxes.conf[res.boxes.xyxy.tolist().index(b)].item())
                        break
            frames.append({"object": name, "view": f"{vx},{vy},{vz}",
                           "range_m": round(rng, 2),
                           "expected_px": round(exp_px, 1)})

    print(f"\n{'nesne':8} {'gorunur':>8} {'YOLO-World':>12} {'egitilmis':>11}")
    print('-' * 44)
    for c in CLASSES:
        s = stats[c]
        if not s["seen"]:
            continue
        w = f"{s['world']}/{s['seen']}"
        t = f"{s['trained']}/{s['seen']}" if trained is not None else "-"
        print(f"{c:8} {s['seen']:8} {w:>12} {t:>11}")
    tot_seen = sum(s["seen"] for s in stats.values())
    tot_w = sum(s["world"] for s in stats.values())
    tot_t = sum(s["trained"] for s in stats.values())
    if tot_seen:
        print('-' * 44)
        print(f"{'TOPLAM':8} {tot_seen:8} {f'{tot_w}/{tot_seen}':>12} "
              + (f"{f'{tot_t}/{tot_seen}':>11}" if trained is not None else f"{'-':>11}"))
        print(f"\n  YOLO-World %{tot_w / tot_seen * 100:.0f}"
              + (f", egitilmis %{tot_t / tot_seen * 100:.0f}"
                 if trained is not None else ""))

    print("\nDogru tespitlerin guven dagilimi (hibrit siralamasi bunu bilmeli):")
    for src in ("world", "trained"):
        v = sorted(confs[src])
        if not v:
            print(f"  {src:8} veri yok")
            continue
        n = len(v)
        print(f"  {src:8} n={n:3}  min {v[0]:.2f}  ortanca {v[n // 2]:.2f}  "
              f"max {v[-1]:.2f}")
    if confs["world"] and confs["trained"]:
        mw = sorted(confs["world"])[len(confs["world"]) // 2]
        mt = sorted(confs["trained"])[len(confs["trained"]) // 2]
        print(f"  ortanca orani {mt / mw:.2f} -> "
              + ("dogrudan birlestirilebilir" if 0.7 <= mt / mw <= 1.4
                 else "normalize etmeden birlestirilemez"))

    os.makedirs('logs', exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=["object", "seen", "world", "trained"])
        wtr.writeheader()
        for c in CLASSES:
            wtr.writerow({"object": c, **stats[c]})
    print(f"\n-> {OUT_CSV}")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
