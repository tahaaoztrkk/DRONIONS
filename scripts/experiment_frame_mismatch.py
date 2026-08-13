"""Does computing the geometry beat asking the model?

The reference work (Wei et al., CHI '26) scored lowest on spatial orientation
(3.2/5) and attributed it to LLMs being weak at spatial reasoning, "further
compounded in our context, where the drone may be oriented differently from the
user (e.g., the drone's camera is opposite to the user)".

This sweeps exactly that mismatch. The target and the user stay put; the drone
views the target from a series of bearings, so the angle between where the drone
looks and where the user faces varies from about -135 to +135 degrees. At each
viewpoint three answers to "where is it, from the user's point of view?" are
scored against ground truth:

  geometric  - project the detection onto the ground plane, then convert into
               the user's frame (navigation/spatial.py)
  llm_naive  - give the VLM the drone image and ask, which is the information a
               camera-only pipeline has
  llm_posed  - additionally tell the VLM, in words, where the drone and the user
               are and which way each faces

llm_posed exists so the comparison is not a straw man: it hands the model the
same facts the geometric method uses and asks it to do the reasoning.

    python3 scripts/experiment_frame_mismatch.py            # all three
    python3 scripts/experiment_frame_mismatch.py --no-llm   # geometry only, free
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gz_pose import PoseReader
from config import USER_POSITION, USER_YAW, GEMINI_API_KEY, GEMINI_MODEL
from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from navigation.spatial import locate_target, relative_to_user, size_plausible

WORLD = 'dronions_scenario'
MODEL = 'x500_dronions_0'

TARGET_NAME = 'box'
TARGET_XY = (3.5, 2.0)

# Bearings the drone views the target from. Restricted to the side of the
# scenario wall (x=1.75, y=-0.5..3.5) with clear line of sight -- viewing
# through the wall is a scene artefact, not a spatial-reasoning result.
VIEW_BEARINGS = [-135.0, -112.5, -90.0, -67.5, -45.0, -22.5, 0.0,
                 22.5, 45.0, 67.5, 90.0, 112.5, 135.0]
# Range is swept as well as bearing. The pilot varied only bearing and got six
# samples, which is too few to claim anything -- and range matters on its own,
# because a ground-plane projection degrades with distance while a language
# model's guess does not obviously do either.
VIEW_RANGES = [2.0, 3.0, 4.0]
VIEW_ALT = 2.0

# The wall, for line-of-sight screening. A viewpoint that looks at the target
# through it measures occlusion, not spatial reasoning -- the pilot avoided
# this by hand-picking six bearings, which is also why it could not be scaled.
# Screening geometrically instead lets the sweep cover the full circle and drop
# only what is genuinely blocked.
WALL_X_SPAN = (1.65, 1.85)
WALL_Y_SPAN = (-0.5, 3.5)


def line_blocked(a, b, samples=60):
    """Does the segment a->b pass through the wall?"""
    for i in range(samples + 1):
        t = i / samples
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if WALL_X_SPAN[0] <= x <= WALL_X_SPAN[1] and \
           WALL_Y_SPAN[0] <= y <= WALL_Y_SPAN[1]:
            return True
    return False

OUT_CSV = 'logs/frame_mismatch.csv'


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15)


def teleport(x, y, z, yaw):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    run(['gz', 'service', '-s', f'/world/{WORLD}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000',
         '--req', f'name: "{MODEL}" position: {{x: {x}, y: {y}, z: {z}}} '
                  f'orientation: {{x: {q[0]}, y: {q[1]}, z: {q[2]}, w: {q[3]}}}'])


class Grab:
    def __init__(self, node):
        self.img = None
        node.create_subscription(Image, '/camera/image_raw', self._cb, 10)

    def _cb(self, m):
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else a


def fresh(node, g):
    """The first frame after the teleport, and nothing more.

    This used to keep spinning for twenty more cycles after the image arrived,
    which cost about two seconds -- and a teleported airframe is unarmed and
    falling the whole time. Measured: 0.47 m lost by 0.3 s, on the ground by
    1.0 s. The geometry was then computed from the altitude the drone had been
    *asked* for, which is the first of the four measurement flaws recorded in
    docs/ and the one that survived into this script.

    Taking the first frame keeps the delay near one camera period (~40 ms, so
    under a centimetre of fall). The caller still reads the true pose at that
    instant rather than trusting the commanded one.
    """
    g.img = None
    t0 = time.time()
    while g.img is None and time.time() - t0 < 6:
        rclpy.spin_once(node, timeout_sec=0.05)
    return g.img


ANSWER_FORMAT = (
    "Answer with EXACTLY one line in this format and nothing else:\n"
    "DIST=<metres, one decimal> DIR=<degrees from the direction the user faces, "
    "positive to the user's left, negative to the user's right>"
)


def ask_llm(client, img_bgr, prompt, model=None, retries=2):
    """One VLM query. Quota errors are returned, not raised: losing the whole
    sweep because one call hit the daily limit wastes the other answers."""
    import PIL.Image
    pil = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    model = model or GEMINI_MODEL
    text = ""
    for attempt in range(retries + 1):
        try:
            r = client.models.generate_content(model=model, contents=[prompt, pil])
            text = r.text.strip()
            break
        except Exception as e:
            msg = str(e)
            if '429' in msg and attempt < retries:
                time.sleep(50)      # free tier replenishes per minute
                continue
            return None, f"ERROR: {msg[:120]}"
    if not text:
        return None, "ERROR: empty"

    m = re.search(r'DIST\s*=\s*(-?[\d.]+).*?DIR\s*=\s*(-?[\d.]+)', text,
                  re.I | re.S)
    if not m:
        return None, text
    return (float(m.group(1)), float(m.group(2))), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-llm', action='store_true',
                    help='geometry only; costs no API quota')
    ap.add_argument('--limit', type=int, default=0,
                    help='ilk N bakis acisi (kota bolmek icin)')
    ap.add_argument('--model', default=GEMINI_MODEL,
                    help='free-tier quota is per model per day, so switching '
                         'model is the way to keep going once one is spent')
    args = ap.parse_args()

    client = None
    if not args.no_llm:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)

    pose = PoseReader()
    if not pose.start():
        sys.exit("Gazebo poz akisi bulunamadi -- simulasyon calisiyor mu?")
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Model pozu gelmedi. Gorulen modeller: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('frame_mismatch')
    g = Grab(node)
    det = YOLOWorldDetector()
    det.set_target(TARGET_NAME)

    true_d, true_b = relative_to_user(TARGET_XY, USER_POSITION, USER_YAW)
    print(f"target {TARGET_NAME} at {TARGET_XY}; user at {USER_POSITION} "
          f"facing {math.degrees(USER_YAW):.0f} deg")
    print(f"TRUTH from the user: {true_d:.2f} m, {math.degrees(true_b):+.1f} deg\n")
    print(f"model for the LLM arms: {args.model}\n" if client else "")

    hdr = (f"{'mismatch':>9} | {'geom d':>7} {'geom b':>7} | "
           f"{'naive d':>8} {'naive b':>8} | {'posed d':>8} {'posed b':>8}")
    print(hdr)
    print(f"{'(deg)':>9} | {'err m':>7} {'err d':>7} | "
          f"{'err m':>8} {'err d':>8} | {'err m':>8} {'err d':>8}")
    print("-" * len(hdr))

    rows = []
    all_vps = [(b, r) for b in VIEW_BEARINGS for r in VIEW_RANGES]
    viewpoints = []
    for b, rng in all_vps:
        aa = math.radians(b)
        dxy = (TARGET_XY[0] + rng * math.cos(aa), TARGET_XY[1] + rng * math.sin(aa))
        if not line_blocked(dxy, TARGET_XY):
            viewpoints.append((b, rng))
    blocked = len(all_vps) - len(viewpoints)
    if args.limit:
        viewpoints = viewpoints[:args.limit]
    print(f"{len(viewpoints)} bakis acisi "
          f"({len(VIEW_BEARINGS)} yon x {len(VIEW_RANGES)} menzil, "
          f"{blocked} tanesi duvar arkasinda kaldigi icin elendi)"
          + (f", ilk {args.limit} tanesi" if args.limit else "") + "\n")

    for bearing, view_range in viewpoints:
        a = math.radians(bearing)
        dx, dy = view_range * math.cos(a), view_range * math.sin(a)
        drone_xy = (TARGET_XY[0] + dx, TARGET_XY[1] + dy)
        yaw = math.atan2(-dy, -dx)                 # face the target
        mismatch = math.degrees(math.atan2(math.sin(yaw - USER_YAW),
                                           math.cos(yaw - USER_YAW)))
        teleport(drone_xy[0], drone_xy[1], VIEW_ALT, yaw)
        img = fresh(node, g)
        # Where the drone *is*, read at the moment the frame arrived, not
        # where it was told to go. The two differ because the airframe is
        # unarmed and falling; measured at 0.47 m by 0.3 s.
        actual = pose.latest() if pose else None
        drone_xyz = actual if actual else (*drone_xy, VIEW_ALT)
        row = {'mismatch_deg': round(mismatch, 1),
               'range_m': view_range,
               'alt_cmd': VIEW_ALT,
               'alt_actual': round(drone_xyz[2], 3) if actual else None,
               'alt_drop': round(VIEW_ALT - drone_xyz[2], 3) if actual else None}

        # --- geometric ---
        gd = gb = None
        if img is not None:
            cands = filter_candidates(det.detect(img))
            # The same physical-size gate the flight pipeline runs. Without it
            # this measures a system we no longer fly: the wall is the top
            # candidate in 71% of surveyed viewpoints, and a wall detection
            # scored as the box is what produced 170-190 degree errors here.
            quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            cands = [c for c in cands
                     if size_plausible(c, drone_xyz, quat, TARGET_NAME)]
            if cands:
                est = locate_target(cands[0], drone_xyz,
                                    (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                                    plane_z=0.0)
                if est:
                    ed, eb = relative_to_user(est[:2], USER_POSITION, USER_YAW)
                    gd = abs(ed - true_d)
                    gb = abs(math.degrees(eb - true_b))
                    row['geom_area'] = round(cands[0].relative_area, 4)
        row['geom_dist_err'] = gd
        row['geom_brg_err'] = gb

        # --- llm arms ---
        nd = nb = pd = pb = None
        if client is not None and img is not None:
            naive = (
                f"You are the camera of an assistive drone helping a blind user "
                f"find their {TARGET_NAME}. The user is standing at the drone's "
                f"launch point, not visible in this image. Where is the "
                f"{TARGET_NAME} relative to the user?\n{ANSWER_FORMAT}")
            got, txt_n = ask_llm(client, img, naive, args.model)
            if got:
                nd = abs(got[0] - true_d)
                nb = abs(((got[1] - math.degrees(true_b) + 180) % 360) - 180)

            posed = (
                f"You are the camera of an assistive drone helping a blind user "
                f"find their {TARGET_NAME}.\n"
                f"Scene facts (metres, x east / y north, angles CCW from +x):\n"
                f"- drone is at ({drone_xy[0]:.2f}, {drone_xy[1]:.2f}), "
                f"altitude {drone_xyz[2]:.1f}, facing {math.degrees(yaw):.0f} deg, "
                f"camera pitched 20 deg down\n"
                f"- user stands at ({USER_POSITION[0]:.2f}, {USER_POSITION[1]:.2f}), "
                f"facing {math.degrees(USER_YAW):.0f} deg\n"
                f"Using the image and these facts, where is the {TARGET_NAME} "
                f"relative to the user?\n{ANSWER_FORMAT}")
            got, txt_p = ask_llm(client, img, posed, args.model)
            if got:
                pd = abs(got[0] - true_d)
                pb = abs(((got[1] - math.degrees(true_b) + 180) % 360) - 180)
        row.update(llm_naive_dist_err=nd, llm_naive_brg_err=nb,
                   llm_posed_dist_err=pd, llm_posed_brg_err=pb)
        rows.append(row)

        def f(v, w, unit=''):
            return f"{v:{w}.1f}{unit}" if v is not None else f"{'--':>{w}}"
        print(f"{mismatch:9.0f} | {f(gd,7)} {f(gb,7)} | "
              f"{f(nd,8)} {f(nb,8)} | {f(pd,8)} {f(pb,8)}")

    print("-" * len(hdr))

    def med(key):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        return vals[len(vals) // 2] if vals else None

    for name, dk, bk in (("geometric", 'geom_dist_err', 'geom_brg_err'),
                         ("llm naive", 'llm_naive_dist_err', 'llm_naive_brg_err'),
                         ("llm posed", 'llm_posed_dist_err', 'llm_posed_brg_err')):
        d, b = med(dk), med(bk)
        n = sum(1 for r in rows if r.get(dk) is not None)
        if d is None:
            print(f"{name:>10}: no answers")
        else:
            print(f"{name:>10}: median distance err {d:5.2f} m, "
                  f"bearing err {b:5.1f} deg   (n={n}/{len(rows)})")

    os.makedirs('logs', exist_ok=True)
    # Taken from the rows rather than hardcoded. The fixed list silently
    # dropped the altitude columns added to diagnose the falling-drone problem,
    # so the run that was meant to check the fix could not be checked.
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nwrote {OUT_CSV}")

    node.destroy_node()
    rclpy.shutdown()


main()
