"""How accurately can the system say where a target is? (full sweep)

Scales the n=6 pilot in scripts/experiment_frame_mismatch.py into something
that can carry a claim: two targets, a full ring of viewing bearings, several
ranges and altitudes, repeated with pose jitter, reporting *signed* error
distributions rather than medians of absolute error.

No VLM is involved, so this costs no API quota and can be re-run freely.

Two guards keep the numbers honest rather than flattering:

  line of sight - viewpoints whose ray to the target crosses the scenario wall
                  are skipped, not silently measured. Looking through a wall is
                  a scene artefact.
  plausibility  - a detection is only counted if its apparent area is within a
                  factor of PLAUSIBLE_RATIO of what an object of that known
                  physical size subtends at that range. Without this the wall
                  gets measured as though it were the box: at 3 m the box should
                  cover ~0.0065 of the frame and the wall covers ~0.19.

Both outcomes are recorded per trial, so the rejection rate is itself a result.

    python3 scripts/experiment_localization_accuracy.py
    python3 scripts/experiment_localization_accuracy.py --quick
"""
import argparse
import csv
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

sys.path.insert(0, '/home/taha/DRONIONS')
from config import USER_POSITION, USER_YAW
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from navigation.spatial import (locate_target, relative_to_user,
                                CAMERA_HFOV, CAMERA_ASPECT)

WORLD = 'dronions_scenario'
DRONE = 'x500_dronions_0'
OUT_CSV = 'logs/localization_accuracy.csv'

# Scenario wall, from dronions_scenario.sdf: 0.2 x 4.0 x 3.0 at (1.75, 1.5).
WALL_X = (1.65, 1.85)
WALL_Y = (-0.5, 3.5)

# (name, world xy, physical width m, physical height m, positive prompts, negatives)
# `prompts=None` means "use the project's own configured prompt expansion for
# this target" (utils/prompts.py), which is what the real system does. A
# hand-written list here matched the wall far more strongly than the box.
TARGETS = [
    ("cardboard_box", (3.5, 2.0), 0.50, 0.30, None, "box"),
    ("green_sphere", (3.5, 3.6), 0.60, 0.60,
     (["green ball", "sphere", "green sphere", "round ball"],
      ["wall", "barrier", "panel", "box"]), None),
]

BEARINGS = [b * 30.0 for b in range(12)]
RANGES = [2.0, 3.5]
ALTITUDES = [2.0]
REPEATS = 1
JITTER_POS = 0.05        # m, stands in for pose uncertainty
JITTER_YAW = math.radians(2.0)
PLAUSIBLE_RATIO = 3.0    # accepted apparent-area factor either way

CAMERA_VFOV = 2.0 * math.atan(math.tan(CAMERA_HFOV / 2.0) / CAMERA_ASPECT)


def expected_area(width, height, rng, alt, obj_z=0.3):
    """Fraction of the frame an object of known size subtends.

    Uses the slant range, not the horizontal one: the camera is metres above
    an object sitting near the floor, so the true line-of-sight distance is
    noticeably longer and the horizontal figure over-predicts the size.

    This is deliberately coarse. Its only job is to separate a plausible
    detection from an order-of-magnitude wrong one -- measured, the wall comes
    back at 25-57x the predicted area while a genuine target lands within a
    factor of two.
    """
    slant = math.hypot(rng, max(alt - obj_z, 0.0))
    fw = (2.0 * math.atan(width / 2.0 / slant)) / CAMERA_HFOV
    fh = (2.0 * math.atan(height / 2.0 / slant)) / CAMERA_VFOV
    return max(fw * fh, 1e-9)


def segment_hits_wall(p0, p1, steps=60):
    """Cheap sampled test: does the line of sight cross the wall footprint?"""
    for i in range(steps + 1):
        t = i / steps
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if WALL_X[0] <= x <= WALL_X[1] and WALL_Y[0] <= y <= WALL_Y[1]:
            return True
    return False


def teleport(x, y, z, yaw):
    q = (math.sin(yaw / 2), math.cos(yaw / 2))
    subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000',
         '--req', f'name: "{DRONE}" position: {{x: {x}, y: {y}, z: {z}}} '
                  f'orientation: {{x: 0, y: 0, z: {q[0]}, w: {q[1]}}}'],
        capture_output=True, timeout=10)


POSE_RE = re.compile(r'\[\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\]')


def gz_true_pose():
    """Ground-truth pose from Gazebo, via the CLI.

    Ground truth has to come from Gazebo. The commanded pose is wrong because
    an unarmed airframe starts falling the moment it is placed. MAVROS's
    local_position is wrong too -- it is PX4's EKF estimate, and an EKF cannot
    track an instantaneous teleport; feeding it that more than tripled the
    measured bearing error.

    The CLI costs about 5 s per call because it reloads world state, which is
    why the grid below is small: a bridged pose topic would be far cheaper, but
    the bridged TFMessage never reached the subscriber here, and correctness of
    the reference matters more than the size of the sweep.

    Returns ((x, y, z), (qx, qy, qz, qw)) or None.
    """
    r = subprocess.run(['gz', 'model', '-m', DRONE, '-p'],
                       capture_output=True, text=True, timeout=15)
    block = r.stdout.split('Pose [ XYZ (m) ] [ RPY (rad) ]:')
    if len(block) < 2:
        return None
    nums = POSE_RE.findall(block[1])
    if len(nums) < 2:
        return None
    x, y, z = (float(v) for v in nums[0])
    roll, pitch, yaw = (float(v) for v in nums[1])
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return ((x, y, z), (sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy))


class Grab:
    def __init__(self, node):
        self.img = None
        node.create_subscription(Image, '/camera/image_raw', self._img, 10)

    def _img(self, m):
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else a


def fresh(node, g, settle_s=0.4, timeout_s=6.0):
    """A frame taken *after* the drone has actually moved.

    Spinning a fixed number of times is not enough: as the simulation slows the
    same count buys less wall-clock time and the grabbed frame can still be from
    the previous pose. That showed up as the no-detection rate climbing from 7%
    to 20% across otherwise identical runs -- a property of the harness, not of
    the system under test. Drain for a fixed duration instead, then take the
    next frame to arrive.
    """
    t0 = time.time()
    while time.time() - t0 < settle_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    g.img = None
    t0 = time.time()
    while g.img is None and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    return g.img, gz_true_pose()


def summarize(name, values, unit):
    if not values:
        print(f"  {name:<22} no data")
        return
    v = sorted(values)
    n = len(v)
    q1, med, q3 = v[n // 4], v[n // 2], v[(3 * n) // 4]
    p90 = v[min(n - 1, int(0.9 * n))]
    print(f"  {name:<22} n={n:4d}  mean {st.mean(v):+6.2f}  med {med:+6.2f}  "
          f"IQR [{q1:+.2f},{q3:+.2f}]  p90|.| {p90:+.2f} {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='coarser grid for a fast check')
    args = ap.parse_args()

    bearings = BEARINGS[::3] if args.quick else BEARINGS
    ranges = [3.0] if args.quick else RANGES
    altitudes = [2.0] if args.quick else ALTITUDES
    repeats = 1 if args.quick else REPEATS

    rclpy.init()
    node = rclpy.create_node('localization_accuracy')
    g = Grab(node)
    det = YOLOWorldDetector()
    random.seed(0)

    rows = []
    planned = 0
    for tname, txy, tw, th, custom, configured in TARGETS:
        if configured:
            det.set_target(configured)
        else:
            det.prompts, det.negative_prompts = custom
            det.model.set_classes(det.prompts + det.negative_prompts)
        true_d, true_b = relative_to_user(txy, USER_POSITION, USER_YAW)
        print(f"\n=== {tname} at {txy} | truth from user: "
              f"{true_d:.2f} m, {math.degrees(true_b):+.1f} deg")

        for bearing in bearings:
            for rng in ranges:
                a = math.radians(bearing)
                base = (txy[0] + rng * math.cos(a), txy[1] + rng * math.sin(a))
                blocked = segment_hits_wall(base, txy)
                for alt in altitudes:
                    for rep in range(repeats):
                        planned += 1
                        row = dict(target=tname, bearing_deg=bearing, range_m=rng,
                                   alt_m=alt, rep=rep, blocked=int(blocked))
                        if blocked:
                            row['outcome'] = 'blocked'
                            rows.append(row)
                            continue

                        jx = base[0] + random.uniform(-JITTER_POS, JITTER_POS)
                        jy = base[1] + random.uniform(-JITTER_POS, JITTER_POS)
                        yaw = (math.atan2(txy[1] - jy, txy[0] - jx)
                               + random.uniform(-JITTER_YAW, JITTER_YAW))
                        teleport(jx, jy, alt, yaw)
                        img, pose = fresh(node, g)
                        if img is None or pose is None:
                            row['outcome'] = 'no_frame'
                            rows.append(row)
                            continue

                        cands = filter_candidates(det.detect(img))
                        if not cands:
                            row['outcome'] = 'no_detection'
                            rows.append(row)
                            continue

                        c = cands[0]
                        exp = expected_area(tw, th, rng, alt)
                        ratio = c.relative_area / exp
                        row.update(area=round(c.relative_area, 5),
                                   area_expected=round(exp, 5),
                                   area_ratio=round(ratio, 2),
                                   conf=round(c.confidence, 3))
                        if not (1.0 / PLAUSIBLE_RATIO <= ratio <= PLAUSIBLE_RATIO):
                            row['outcome'] = 'implausible_size'
                            rows.append(row)
                            continue

                        # Real pose at capture time, not the commanded one.
                        act_xyz, act_quat = pose
                        row.update(act_x=round(act_xyz[0], 3),
                                   act_y=round(act_xyz[1], 3),
                                   act_z=round(act_xyz[2], 3),
                                   drop_m=round(alt - act_xyz[2], 3))
                        est = locate_target(c, act_xyz, act_quat, plane_z=0.0)
                        if est is None:
                            row['outcome'] = 'ray_misses_ground'
                            rows.append(row)
                            continue

                        ed, eb = relative_to_user(est[:2], USER_POSITION, USER_YAW)
                        berr = math.degrees(math.atan2(math.sin(eb - true_b),
                                                       math.cos(eb - true_b)))
                        row.update(outcome='ok',
                                   est_x=round(est[0], 3), est_y=round(est[1], 3),
                                   pos_err=round(math.hypot(est[0] - txy[0],
                                                            est[1] - txy[1]), 3),
                                   dist_err_signed=round(ed - true_d, 3),
                                   brg_err_signed=round(berr, 2))
                        rows.append(row)
            done = sum(1 for r in rows if r['target'] == tname)
            print(f"  bearing {bearing:5.0f} deg  ({done} trials so far)")

    # ---- summary ----
    ok = [r for r in rows if r.get('outcome') == 'ok']
    print(f"\n{'='*72}\ntrials {len(rows)}  usable {len(ok)}")
    from collections import Counter
    for k, v in Counter(r.get('outcome', '?') for r in rows).most_common():
        print(f"  {k:<20} {v:4d}  ({100*v/len(rows):.0f}%)")

    print("\nSIGNED errors (positive = over-estimate / to the user's left):")
    summarize("distance err", [r['dist_err_signed'] for r in ok], "m")
    summarize("bearing err", [r['brg_err_signed'] for r in ok], "deg")
    print("\nABSOLUTE:")
    summarize("|distance err|", [abs(r['dist_err_signed']) for r in ok], "m")
    summarize("|bearing err|", [abs(r['brg_err_signed']) for r in ok], "deg")
    summarize("position err", [r['pos_err'] for r in ok], "m")
    drops = [r['drop_m'] for r in ok if r.get('drop_m') is not None]
    if drops:
        summarize("altitude lost by capture", drops, "m")

    for key, label in (('range_m', 'range'), ('alt_m', 'altitude'), ('target', 'target')):
        print(f"\nby {label}:")
        for val in sorted({r[key] for r in ok}, key=str):
            sub = [r for r in ok if r[key] == val]
            d = [abs(r['dist_err_signed']) for r in sub]
            b = [abs(r['brg_err_signed']) for r in sub]
            print(f"  {str(val):<16} n={len(sub):4d}  |dist| med {st.median(d):.2f} m"
                  f"   |brg| med {st.median(b):5.1f} deg")

    os.makedirs('logs', exist_ok=True)
    keys = ['target', 'bearing_deg', 'range_m', 'alt_m', 'rep', 'blocked',
            'outcome', 'area', 'area_expected', 'area_ratio', 'conf',
            'act_x', 'act_y', 'act_z', 'drop_m',
            'est_x', 'est_y', 'pos_err', 'dist_err_signed', 'brg_err_signed']
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nwrote {OUT_CSV}")

    node.destroy_node()
    rclpy.shutdown()


main()
