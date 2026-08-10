"""Measure how accurately the system can say *where* a target is.

This is the metric the reference work could only rate subjectively: their
spatial orientation task scored 3.2/5 and they had no ground truth to compare
against. In simulation the true pose of every object is known, so the spoken
answer can be scored in metres and degrees.

Teleports the drone to a series of viewpoints around a target of known
position, runs the detector, projects the detection onto the ground plane and
compares against truth. Uses no VLM, so it costs no API quota and can be run as
often as needed.

    python3 scripts/measure_localization.py
"""
import math
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
from navigation.spatial import locate_target, relative_to_user, describe_target

WORLD = 'dronions_scenario'
MODEL = 'x500_dronions_0'

# Ground truth from the scenario world.
TARGET_NAME = 'box'
TARGET_XY = (3.5, 2.0)
TARGET_Z = 0.15

# Viewpoints: (distance from target, bearing the drone sits at, altitude).
# All on the clear side of the scenario wall (x=1.75, y=-0.5..3.5) -- viewing
# from the -x side puts the wall between camera and target, and the detector
# then locks onto the wall rather than the box, which is a test-rig artefact
# rather than a localization error.
VIEWPOINTS = [
    (1.5,   0.0, 1.0),
    (2.0,   0.0, 1.5),
    (3.0,   0.0, 2.0),
    (4.0,   0.0, 2.0),
    (3.0,  45.0, 2.0),
    (3.0, -45.0, 2.0),
    (3.0, -90.0, 2.0),
    (3.0,   0.0, 3.0),
]


def teleport(x, y, z, yaw):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000',
         '--req', f'name: "{MODEL}" position: {{x: {x}, y: {y}, z: {z}}} '
                  f'orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}'],
        capture_output=True, timeout=10)


class Grab:
    def __init__(self, node):
        self.img = None
        node.create_subscription(Image, '/camera/image_raw', self._cb, 10)

    def _cb(self, m):
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else a


def main():
    rclpy.init()
    node = rclpy.create_node('measure_localization')
    g = Grab(node)
    det = YOLOWorldDetector()
    det.set_target(TARGET_NAME)

    print(f"target '{TARGET_NAME}' truth = ({TARGET_XY[0]}, {TARGET_XY[1]}, {TARGET_Z})")
    print(f"user at {USER_POSITION}, facing {math.degrees(USER_YAW):.0f} deg\n")
    true_d, true_b = relative_to_user(TARGET_XY, USER_POSITION, USER_YAW)
    print(f"truth from user: {true_d:.2f} m, {math.degrees(true_b):+.1f} deg\n")

    print(f"{'range':>6} {'from':>6} {'alt':>5} | {'est x':>7} {'est y':>7} "
          f"| {'pos err':>8} {'dist err':>9} {'brg err':>8} | detection")
    print("-" * 100)

    errors = []
    for dist, from_deg, alt in VIEWPOINTS:
        a = math.radians(from_deg)
        dx, dy = dist * math.cos(a), dist * math.sin(a)
        # Face the target from wherever we are placed.
        yaw = math.atan2(-dy, -dx)
        teleport(TARGET_XY[0] + dx, TARGET_XY[1] + dy, alt, yaw)

        g.img = None
        t0 = time.time()
        while g.img is None and time.time() - t0 < 6:
            rclpy.spin_once(node, timeout_sec=0.2)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
        if g.img is None:
            print(f"{dist:6.1f} {from_deg:6.0f} {alt:5.1f} | no frame")
            continue

        cands = filter_candidates(det.detect(g.img))
        if not cands:
            print(f"{dist:6.1f} {from_deg:6.0f} {alt:5.1f} | no detection")
            continue

        drone_xyz = (TARGET_XY[0] + dx, TARGET_XY[1] + dy, alt)
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        est = locate_target(cands[0], drone_xyz, quat, plane_z=0.0)
        if est is None:
            print(f"{dist:6.1f} {from_deg:6.0f} {alt:5.1f} | ray misses ground")
            continue

        pos_err = math.hypot(est[0] - TARGET_XY[0], est[1] - TARGET_XY[1])
        est_d, est_b = relative_to_user(est[:2], USER_POSITION, USER_YAW)
        d_err = abs(est_d - true_d)
        b_err = abs(math.degrees(est_b - true_b))
        errors.append((pos_err, d_err, b_err))
        print(f"{dist:6.1f} {from_deg:6.0f} {alt:5.1f} | {est[0]:7.2f} {est[1]:7.2f} "
              f"| {pos_err:7.2f}m {d_err:8.2f}m {b_err:7.1f}d "
              f"| area {cands[0].relative_area:.4f} conf {cands[0].confidence:.2f}")

    if errors:
        p = [e[0] for e in errors]
        d = [e[1] for e in errors]
        b = [e[2] for e in errors]
        print("-" * 100)
        print(f"n={len(errors)}  position err  med {sorted(p)[len(p)//2]:.2f} m  max {max(p):.2f} m")
        print(f"          distance err  med {sorted(d)[len(d)//2]:.2f} m  max {max(d):.2f} m")
        print(f"          bearing  err  med {sorted(b)[len(b)//2]:.1f} deg  max {max(b):.1f} deg")
        print(f"\nexample sentence:")
        print("  " + describe_target((TARGET_XY[0], TARGET_XY[1], TARGET_Z),
                                     USER_POSITION, USER_YAW, "Kutunuz"))

    node.destroy_node()
    rclpy.shutdown()


main()
