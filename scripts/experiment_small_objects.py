#!/usr/bin/env python3
"""
Görevi: Tespit oranının görünen nesne boyutuyla nasıl düştüğünü ölçer.

Every indoor scenario in docs/ turns on one unmeasured number: the detector
finds the 0.63 m scenario box in 14% of viewpoints, and the objects those
scenarios call for -- a cup, keys, a phone -- are five to ten times smaller.
Whether "find my cup in this room" is buildable at all depends on what recall
does at that size, and finding out after building the room would be expensive.

Rather than introduce a cup, which would change class, texture and shape at the
same time, this varies only apparent size: the same box, viewed from further
away. Apparent size goes as 1/range, so the 0.63 m box at 10 m subtends what a
0.12 m cup would at 2 m. The curve therefore predicts the cup case directly,
with no new assets and no confound.

  scripts/experiment_small_objects.py            # needs the sim + bridge up
  scripts/experiment_small_objects.py --samples 6
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from navigation.spatial import CAMERA_HFOV, CAMERA_PITCH_DOWN
from gz_pose import PoseReader, find_world

MODEL = 'x500_dronions_0'
TARGET_NAME = 'box'
TARGET_XY = (3.5, 2.0)
TARGET_WIDTH = 0.63            # m, the scenario box with its scale applied
TARGET_Z = 0.30                # m, roughly the box's mid-height
CAM_OFFSET = (0.12, 0.03, 0.242)   # camera mount, from x500_dronions/model.sdf

# What the sweep actually varies is apparent width, so that is what is listed
# here -- the ranges are then computed from whatever lens is fitted. Listing
# distances instead only works for one field of view: narrowing the lens from
# 100 to 50 degrees makes the box 306 px at 2 m, and a sweep out to 12 m would
# never reach the regime where detection fails. Two campaigns at different
# lenses are comparable in pixels and not in metres, so pixels are the axis.
PX_TARGETS = [120, 94, 76, 63, 50, 41, 33, 28]
VIEW_ALT = 2.0

# Past this the box leaves the modelled part of the world and the wall starts
# intruding, so a wide lens simply cannot reach the smallest sizes. Better to
# drop those rows and say so than to measure something else and report it as
# apparent size.
MAX_RANGE = 30.0


# A few bearings per range, so a single unlucky viewing angle does not stand in
# for the whole distance.
BEARINGS = [-40.0, 0.0, 40.0]

# How far a detection's width may stray from what geometry predicts and still
# be counted as the box. Wide enough that a loose or clipped box still counts,
# tight enough to reject the wall, which is six times too large.
SIZE_LO, SIZE_HI = 0.4, 2.5


def ranges_for(px_targets, width_m, alt, img_w, hfov):
    """Horizontal ranges at which the box subtends each target width."""
    focal = (img_w / 2.0) / math.tan(hfov / 2.0)
    out = []
    for px in px_targets:
        slant = width_m * focal / px
        if slant <= alt:                      # directly overhead, no horizontal
            continue
        rng = math.sqrt(slant ** 2 - alt ** 2)
        if rng <= MAX_RANGE:
            out.append((round(rng, 2), px))
    return out


def out_csv(hfov):
    """One file per lens. The first version wrote a single path, so the second
    campaign would have overwritten the first -- the only copy of the baseline
    the new run exists to be compared against."""
    return f'logs/small_objects_{round(math.degrees(hfov)):d}deg.csv'


def assert_lens_matches(node, img_w):
    """Refuse to run unless the *running* camera is the lens the maths assumes.

    A whole campaign was measured against a simulator that had never been
    restarted: the model file said 100 degrees, the running camera was still at
    50, and the numbers came out plausible enough to tabulate. Verifying the
    file proves nothing -- the model is read at spawn, so only the live camera
    is evidence. camera_info carries the focal length directly.
    """
    got = {}
    sub = node.create_subscription(
        CameraInfo, '/camera/camera_info',
        lambda m: got.setdefault('fx', m.k[0]), 1)
    t0 = time.time()
    while 'fx' not in got and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if 'fx' not in got:
        sys.exit("/camera/camera_info gelmedi -- lens dogrulanamadan olcum yapilmaz.")
    want = (img_w / 2.0) / math.tan(CAMERA_HFOV / 2.0)
    if abs(got['fx'] - want) / want > 0.02:
        live = 2 * math.atan((img_w / 2.0) / got['fx'])
        sys.exit(
            f"Lens uyusmuyor. Calisan kamera {math.degrees(live):.0f} derece "
            f"(odak {got['fx']:.0f} px), matematik {math.degrees(CAMERA_HFOV):.0f} "
            f"derece bekliyor (odak {want:.0f} px).\n"
            "Model spawn aninda okunuyor: simulasyonu yeniden baslatin.")
    print(f"lens dogrulandi: calisan kamera odak {got['fx']:.0f} px")


class Grab:
    def __init__(self, node):
        self.img = None
        node.create_subscription(Image, '/camera/image_raw', self._cb, 1)

    def _cb(self, m):
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else a


def world_control(world, req):
    subprocess.run(
        ['gz', 'service', '-s', f'/world/{world}/control',
         '--reqtype', 'gz.msgs.WorldControl', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000', '--req', req],
        capture_output=True)


def teleport(world, name, x, y, z, yaw=0.0):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    subprocess.run(
        ['gz', 'service', '-s', f'/world/{world}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000',
         '--req', f'name: "{name}" position: {{x: {x}, y: {y}, z: {z}}} '
                  f'orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}'],
        capture_output=True)


# Where to touch down to zero the velocity: clear floor, not under the
# viewpoint.
#
# Grounding beneath the viewpoint puts the airframe inside the furniture
# whenever the viewpoint is over it, and the contact solver throws it out
# sideways rather than stopping it. Measured with the room world: asking for
# (2.0, 0.0, 1.4) delivered (1.54, 0.25, 1.18), half a metre away and moving
# fast enough to travel another 0.46 m during the fifteen steps a capture takes.
# The geometry stays self-consistent, because the pose is read rather than
# assumed, but the viewpoints sampled are not the ones asked for and the
# airframe arrives at them tilted.
GROUND_SPOT = (0.0, -1.0)


def place(world, name, x, y, z, yaw, ground_z=0.12, settle=60):
    """Put the drone at a viewpoint with its velocity actually zero.

    set_pose moves the model but leaves its velocity alone, and with physics
    paused every capture still advances 15 steps. So the drone kept the speed
    it had picked up in the previous capture and gained more: measured drop was
    2 cm, then 5, then 9, accelerating, and after a few hundred viewpoints it
    is nowhere near where the geometry says. Touching it down first lets the
    contact zero the velocity, after which every viewpoint holds to within half
    a centimetre of the last.
    """
    teleport(world, name, GROUND_SPOT[0], GROUND_SPOT[1], ground_z, yaw)
    world_control(world, f'pause: true, multi_step: {settle}')
    teleport(world, name, x, y, z, yaw)


def placed_ok(actual, want, quat=None, tol=0.06, max_tilt_deg=15.0):
    """Did the drone actually reach the viewpoint, level?

    A viewpoint that intersects the furniture cannot be occupied, and the
    contact solver answers by throwing the airframe somewhere else -- measured
    in the room, asking for (2.4, -0.1, 1.15), which is 0.14 m above a 1.015 m
    table, delivered (2.05, -0.13, 1.52) upside down at -143 degrees of roll.

    Nothing downstream notices. The pose is read rather than assumed, so the
    geometry stays self-consistent and the sample looks like any other; it is
    simply a measurement of a different, tilted, viewpoint. Sweeps that include
    a few of these grow a tail of outliers that reads as a localisation
    problem, which is what four estimates over half a metre in one sweep turned
    out to be. Callers should skip a sample this rejects rather than trust it.
    """
    if actual is None:
        return False
    if math.dist(actual, want) > tol:
        return False
    if quat is None:
        return True
    x, y, z, w = quat
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z),
                                   1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    return abs(roll) <= max_tilt_deg and abs(pitch) <= max_tilt_deg


def settled_pose(pose, delay=0.35):
    """Pose as of the captured frame. With physics paused the pose stream only
    moves when the world steps, and reading it the instant the frame arrives
    returns the previous viewpoint -- which projects the target onto empty
    floor and scores every detection as a miss."""
    time.sleep(delay)
    return pose.latest()


def fresh(node, g, world, steps=15):
    """The first frame rendered after the teleport, with physics paused.

    The drone is disarmed, so a teleport to 2 m is a drop: it is on the ground
    0.9 s later, and waiting for a settled frame guaranteed a settled *grounded*
    frame. Two whole campaigns were measured from ground level while every
    expected-pixel figure assumed a 2 m hover -- consistent within itself, and
    40% wrong in the near field, where it mattered most.

    Pausing first fixes the altitude and makes frame capture deterministic as
    well: with nothing being rendered, the queue can be drained down to empty,
    and the next frame to arrive is necessarily the one that was asked for.
    15 steps is about half a camera period at 30 Hz -- enough for exactly one
    frame, during which the drone falls roughly a centimetre.
    """
    deadline = time.time() + 1.0
    while time.time() < deadline:
        g.img = None
        rclpy.spin_once(node, timeout_sec=0.02)
        if g.img is None:
            break
    g.img = None
    world_control(world, f'pause: true, multi_step: {steps}')
    t0 = time.time()
    while g.img is None and time.time() - t0 < 6:
        rclpy.spin_once(node, timeout_sec=0.05)
    return g.img


def expected_px(width_m, rng, alt, img_w):
    """How wide the object should appear, from geometry alone."""
    slant = math.hypot(rng, alt)
    focal_px = (img_w / 2.0) / math.tan(CAMERA_HFOV / 2.0)
    return width_m * focal_px / slant


def project_into_frame(world_xyz, drone_xyz, yaw, img_w, img_h):
    """Where a known world point lands in the frame. Forward projection, the
    opposite direction to navigation's locate_target.

    Scoring by locate_target -- ray from the bbox down to the ground -- looked
    natural and silently destroyed the experiment. It refuses rays shallower
    than MAX_RANGE_HEIGHT_RATIO, correctly, because a ray cast from 2 m altitude
    to something 26 m away is mostly noise. So every viewpoint past 10 m scored
    zero detections whatever the camera saw: not a measurement, an artefact of
    the scoring. Here the box's position is known, so projecting it into the
    image instead has no shallow-ray problem at any range.
    """
    cam = (drone_xyz[0] + CAM_OFFSET[0] * math.cos(yaw) - CAM_OFFSET[1] * math.sin(yaw),
           drone_xyz[1] + CAM_OFFSET[0] * math.sin(yaw) + CAM_OFFSET[1] * math.cos(yaw),
           drone_xyz[2] + CAM_OFFSET[2])
    dx, dy, dz = (world_xyz[0] - cam[0], world_xyz[1] - cam[1], world_xyz[2] - cam[2])
    # into body frame (x forward, y left, z up)
    bx = dx * math.cos(yaw) + dy * math.sin(yaw)
    by = -dx * math.sin(yaw) + dy * math.cos(yaw)
    # then undo the camera's downward pitch
    cp, sp = math.cos(CAMERA_PITCH_DOWN), math.sin(CAMERA_PITCH_DOWN)
    fx = bx * cp - dz * sp
    fz = bx * sp + dz * cp
    if fx <= 0.01:
        return None
    focal = (img_w / 2.0) / math.tan(CAMERA_HFOV / 2.0)
    return (img_w / 2.0 - by / fx * focal, img_h / 2.0 - fz / fx * focal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=3,
                    help='her menzil-yon ciftinde kac tekrar')
    a = ap.parse_args()

    world = find_world()
    if not world:
        sys.exit("Gazebo bulunamadi -- simulasyon calisiyor mu?")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('small_objects')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("/camera/image_raw'dan kare yok -- kopru calisiyor mu?")

    assert_lens_matches(node, 1280)

    det = YOLOWorldDetector()
    det.set_target(TARGET_NAME)

    plan = ranges_for(PX_TARGETS, TARGET_WIDTH, VIEW_ALT, 1280, CAMERA_HFOV)
    if not plan:
        sys.exit("Bu lensle hicbir hedef boyut menzil icinde degil.")
    dropped = [px for px in PX_TARGETS if px not in [p for _, p in plan]]

    print(f"hedef {TARGET_NAME}, gercek genislik {TARGET_WIDTH} m")
    print(f"lens {math.degrees(CAMERA_HFOV):.0f} derece "
          f"({CAMERA_HFOV:.4f} rad), odak "
          f"{(1280 / 2) / math.tan(CAMERA_HFOV / 2):.0f} px")
    if dropped:
        print(f"menzil disi kalan boyutlar ({MAX_RANGE:.0f} m siniri): "
              f"{', '.join(str(p) for p in dropped)} px")
    print(f"{len(plan)} boyut x {len(BEARINGS)} yon x {a.samples} tekrar\n")
    print(f"{'menzil':>7} {'beklenen px':>12} {'tespit':>10} {'olculen px':>11}")
    print("-" * 44)

    rows = []
    for rng, _target_px in plan:
        hits = 0
        trials = 0
        widths = []
        for bearing in BEARINGS:
            for _ in range(a.samples):
                ang = math.radians(bearing)
                dx, dy = rng * math.cos(ang), rng * math.sin(ang)
                dxy = (TARGET_XY[0] + dx, TARGET_XY[1] + dy)
                yaw = math.atan2(-dy, -dx)
                place(world, MODEL, dxy[0], dxy[1], VIEW_ALT, yaw)
                img = fresh(node, g, world)
                trials += 1
                if img is None:
                    continue
                cands = filter_candidates(det.detect(img))
                # A detection counts only if it covers where the box actually
                # is *and* is roughly the right size. Position alone is not
                # enough: the drone is aimed at the box, so the wall behind it
                # covers the same pixels, and it passed every single sample --
                # 427 px measured where geometry predicts 120, an object 2.2 m
                # across. Requiring the width to match rejects it without
                # rejecting a genuinely marginal detection of the box.
                drone_xyz = settled_pose(pose) or (dxy[0], dxy[1], VIEW_ALT)
                h_img, w_img = img.shape[:2]
                uv = project_into_frame((TARGET_XY[0], TARGET_XY[1], TARGET_Z),
                                        drone_xyz, yaw, w_img, h_img)
                exp_w = expected_px(TARGET_WIDTH, rng, VIEW_ALT, w_img)
                good = []
                if uv is not None:
                    for c in cands:
                        x0, y0, x1, y1 = c.bbox
                        if not (x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1):
                            continue
                        if not SIZE_LO <= (x1 - x0) / exp_w <= SIZE_HI:
                            continue
                        good.append(c)
                if good:
                    hits += 1
                    c = max(good, key=lambda c: c.confidence)
                    widths.append(c.bbox[2] - c.bbox[0])

        exp = expected_px(TARGET_WIDTH, rng, VIEW_ALT, 1280)
        got = (sum(widths) / len(widths)) if widths else None
        rows.append({'hfov_deg': round(math.degrees(CAMERA_HFOV), 1),
                     'range_m': rng, 'expected_px': round(exp, 1),
                     'trials': trials, 'hits': hits,
                     'recall': round(hits / trials, 3) if trials else 0,
                     'measured_px': round(got, 1) if got else ''})
        print(f"{rng:7.1f} {exp:12.0f} {hits:6}/{trials:<3} "
              f"{(f'{got:.0f}' if got else '--'):>11}")

    os.makedirs('logs', exist_ok=True)
    path = out_csv(CAMERA_HFOV)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("-" * 44)
    print(f"\n-> {path}")
    print("\nBir fincan (~0.12 m) su menzillerdeki kutuyla ayni buyuklukte gorunur:")
    for d in (1.5, 2.0, 3.0):
        equiv = TARGET_WIDTH / 0.12 * d
        print(f"  fincan {d:.1f} m'de  ==  kutu {equiv:.1f} m'de")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
