#!/usr/bin/env python3
"""
Görevi: PX4'ün sapma kestirimi ile gerçek yönelim arasındaki sabit farkı ölçer.

The vehicle's heading estimate is not its heading. Measured in this SITL setup,
MAVROS reports a yaw 4.36 degrees away from the simulator's truth, constant to
better than a fiftieth of a degree over twelve samples, while its *position*
agrees with truth to within two centimetres.

That combination is the damaging one. A ray is cast from a position that is
right, in a direction that is wrong, so the error is lateral and grows with
range: 0.23 m at three metres. It is invisible in flight because the drone
navigates in its own frame and therefore still arrives -- what it corrupts is
the answer given to the user, whose position and facing are defined in the
world frame, and the world position an estimate is scored against.

It also means every offline accuracy figure in this project is optimistic. Those
measurements read the attitude from Gazebo, because the flight stack was not
running, so they measured the camera and the geometry with a heading that was
correct. The flight does not have that.

On real hardware this is what magnetometer calibration and declination are
about, and a real drone with a badly calibrated compass shows the same
signature. The number is therefore not a simulator artifact to be deleted but a
property of the vehicle to be measured, which is what this does.

  scripts/measure_heading_bias.py            # 20 samples
  scripts/measure_heading_bias.py -n 60
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from gz_pose import PoseReader, find_world


def yaw_degrees(q) -> float:
    """Yaw about +z, in degrees, from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return math.degrees(math.atan2(2.0 * (w * z + x * y),
                                   1.0 - 2.0 * (y * y + z * z)))


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--samples', type=int, default=20)
    ap.add_argument('--interval', type=float, default=0.5)
    a = ap.parse_args()

    world = find_world()
    if not world:
        sys.exit("Gazebo bulunamadi.")
    truth = PoseReader(world)
    truth.start()
    if truth.wait_for_pose(15) is None:
        sys.exit(f"Gercek poz gelmedi. Gorulen: {truth.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('heading_bias')
    # PX4 publishes local_position with BEST_EFFORT; a RELIABLE subscription
    # here matches nothing and waits for ever.
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    latest = {}

    def on_pose(msg: PoseStamped):
        o = msg.pose.orientation
        latest['q'] = (o.x, o.y, o.z, o.w)
        latest['p'] = (msg.pose.position.x, msg.pose.position.y,
                       msg.pose.position.z)

    node.create_subscription(PoseStamped, '/mavros/local_position/pose',
                             on_pose, qos)
    t0 = time.time()
    while 'q' not in latest and time.time() - t0 < 20:
        rclpy.spin_once(node, timeout_sec=0.2)
    if 'q' not in latest:
        sys.exit("MAVROS pozu gelmedi -- ucus yigini calisiyor mu?")

    print(f"\n{'':6} {'MAVROS':>10} {'gercek':>10} {'fark':>9} "
          f"{'konum farki':>13}")
    print('-' * 52)
    diffs, offsets = [], []
    for i in range(a.samples):
        spun = time.time()
        while time.time() - spun < 0.3:
            rclpy.spin_once(node, timeout_sec=0.1)
        gq, gp = truth.latest_quat(), truth.latest()
        if gq is None or gp is None:
            continue
        m_yaw, g_yaw = yaw_degrees(latest['q']), yaw_degrees(gq)
        d = wrap(m_yaw - g_yaw)
        off = math.dist(latest['p'][:2], gp[:2])
        diffs.append(d)
        offsets.append(off)
        print(f"{i:5}. {m_yaw:9.2f}° {g_yaw:9.2f}° {d:8.2f}° {off:12.3f} m")
        time.sleep(a.interval)

    if not diffs:
        sys.exit("Ornek toplanamadi.")
    diffs.sort()
    n = len(diffs)
    med = diffs[n // 2]
    print('-' * 52)
    print(f"n={n}  ortanca {med:+.2f}°  aralik {diffs[0]:+.2f}° .. "
          f"{diffs[-1]:+.2f}°  (yayilim {diffs[-1] - diffs[0]:.2f}°)")
    print(f"konum farki ortancasi {sorted(offsets)[n // 2]:.3f} m")
    print(f"\n3 m mesafede bu sapma {3.0 * math.tan(math.radians(abs(med))):.2f} m "
          f"yanal hata demek.")
    print(f"Uygulamak icin:  export DRONIONS_HEADING_BIAS_DEG={med:.2f}")

    truth.stop()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
