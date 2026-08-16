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
from collections import Counter
import os
import random
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

# The drone is deliberately *not* aimed straight at the target.
#
# Every viewpoint used to set yaw = atan2(-dy, -dx), putting the target exactly
# on the boresight, and the "posed" prompt hands the model the drone's position
# and heading. Between them those facts determine the answer by arithmetic --
# the target lies along the heading -- so that arm was measuring whether the
# model can do trigonometry on numbers it was given, not whether it can read
# space out of an image. It scored 4.3 deg against geometry's 8.3 on the first
# paired batch, which is what exact arithmetic beating noisy detection looks
# like, and the pilot's opposite result (45.3 deg, worst of three) was the same
# artefact seen through a weaker model.
#
# A fixed pseudo-random offset per viewpoint breaks that, while staying well
# inside the camera's ~50 deg horizontal half-angle so the target is still in
# frame and still detectable.
YAW_OFFSET_MAX = 0.45           # rad, ~26 deg
YAW_OFFSET_SEED = 20260815

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


def out_path_for(model, no_llm):
    """One file per model, so a multi-day arm accumulates instead of being
    overwritten. The quota is 20 calls a day and the full arm needs 74, so
    every batch has to add to the last -- a run that silently replaced the
    previous day's work would be indistinguishable from one that collected
    nothing."""
    if no_llm:
        return 'logs/frame_mismatch_geom.csv'
    safe = re.sub(r'[^a-z0-9.]+', '_', model.lower())
    return f'logs/frame_mismatch_{safe}.csv'


# Every column a row can carry. Explicit, and checked against each row before
# writing: a hardcoded subset silently dropped the altitude columns once, and
# the run that was meant to verify a fix could not be verified. Failing loudly
# on an unknown key is the opposite mistake to make.
FIELDS = ['mismatch_deg', 'range_m', 'yaw_offset_deg', 'true_dist', 'true_brg',
          'llm_naive_dist', 'llm_naive_brg', 'llm_posed_dist', 'llm_posed_brg',
          'alt_cmd', 'alt_actual', 'alt_drop',
          'geom_area', 'geom_dist_err', 'geom_brg_err', 'geom_reason',
          'llm_naive_dist_err', 'llm_naive_brg_err', 'llm_naive_note',
          'llm_posed_dist_err', 'llm_posed_brg_err', 'llm_posed_note']


class RowWriter:
    """Writes each viewpoint as it completes, not at the end.

    A 37-viewpoint sweep was interrupted after eleven had produced paired
    answers, and every one was lost: the file was only written once the loop
    finished. On a run that takes over an hour and spends a daily API quota
    doing it, all-or-nothing is the wrong contract -- and resume already knows
    how to pick up a partial file.
    """

    def __init__(self, path, prior):
        self.path = path
        self.fh = open(path, 'w', newline='')
        self.w = csv.DictWriter(self.fh, fieldnames=FIELDS, extrasaction='ignore')
        self.w.writeheader()
        for r in prior:
            self.write(r)

    def write(self, row):
        unknown = set(row) - set(FIELDS)
        if unknown:
            raise KeyError(f"CSV sutunu tanimsiz: {sorted(unknown)} -- "
                           f"FIELDS listesine ekleyin")
        self.w.writerow(row)
        self.fh.flush()

    def close(self):
        self.fh.close()


def load_done(path):
    """Viewpoints already measured, keyed by (bearing, range)."""
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as fh:
            rows = [r for r in csv.DictReader(fh)]
    done = set()
    for r in rows:
        if r.get('llm_naive_brg_err') or r.get('llm_posed_brg_err'):
            try:
                done.add((float(r['mismatch_deg']), float(r['range_m'])))
            except (KeyError, ValueError, TypeError):
                pass
    return rows, done


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
        # Depth 1. At depth 10 the queue fills with frames rendered before the
        # teleport whenever the loop pauses -- which it does for seconds at a
        # time while the LLM arms are called -- and the capture then scores the
        # previous viewpoint's image against this viewpoint's geometry.
        # Reproduced without spending quota: 62% coverage with no delay, 0%
        # with a ten-second one.
        node.create_subscription(Image, '/camera/image_raw', self._cb, 1)

    def _cb(self, m):
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        self.img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else a


def settled(pose, want_xy, timeout=2.0, tol=0.15):
    """Wait until Gazebo reports the drone actually at the commanded spot.

    The teleport is a service call and the physics step that applies it is not
    instantaneous, so the frame arriving right after it can still have been
    rendered from the previous viewpoint. That produced silent, run-to-run
    variation in yield -- 16, then 25, then 22 usable samples from the same
    code and the same scene -- because some viewpoints were scored against an
    image of somewhere else.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = pose.latest()
        if p and math.hypot(p[0] - want_xy[0], p[1] - want_xy[1]) < tol:
            return True
        time.sleep(0.01)
    return False


def fresh(node, g, skip=2):
    """The first frame after the teleport has taken effect.

    `skip` frames are discarded first: the pose can be correct while the
    renderer is still a frame or two behind it. At ~27 Hz that costs about
    80 ms, and an unarmed airframe falls under three centimetres in that time.

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
    # Drain whatever is already queued before waiting for anything new. Depth 1
    # keeps this short, but one stale frame can still be sitting there.
    for _ in range(5):
        g.img = None
        rclpy.spin_once(node, timeout_sec=0.0)
    for _ in range(skip + 1):
        g.img = None
        t0 = time.time()
        while g.img is None and time.time() - t0 < 6:
            rclpy.spin_once(node, timeout_sec=0.05)
        if g.img is None:
            return None
    return g.img


ANSWER_FORMAT = (
    "Answer with EXACTLY one line in this format and nothing else:\n"
    "DIST=<metres, one decimal> DIR=<degrees from the direction the user faces, "
    "positive to the user's left, negative to the user's right>"
)


# Minimum gap between API calls. The free tier limits requests per minute as
# well as per day, and the sweep fires two calls per viewpoint back to back --
# a whole 37-viewpoint run returned zero answers while the daily quota was
# still available, because the burst never let the per-minute window drain.
# Retrying after 50 s did not help: the next viewpoint's calls arrived
# immediately and kept the limit saturated.
MIN_CALL_GAP = 8.0      # s
_last_call = [0.0]


def ask_llm(client, img_bgr, prompt, model=None, retries=2):
    """One VLM query, paced and with its failure text preserved.

    Errors are returned rather than raised: losing a whole sweep because one
    call hit a limit wastes every other answer. The text comes back with them
    because a run once reported no LLM answers at all and nothing anywhere
    said why -- the reason had been computed and thrown away."""
    import PIL.Image
    pil = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    model = model or GEMINI_MODEL
    text = ""
    for attempt in range(retries + 1):
        gap = MIN_CALL_GAP - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        _last_call[0] = time.time()
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
    ap.add_argument('--delay', type=float, default=0.0,
                    help='bakis acilari arasina yapay bekleme (sn). LLM '
                         'cagrilarinin yarattigi gecikmeyi kota harcamadan '
                         'taklit etmek icin.')
    ap.add_argument('--repeat', type=int, default=1,
                    help='supurmeyi N kez tekrarla ve ornekleri havuzla. '
                         'Tek supurme yeterince kararli degil: ayni kod ve '
                         'ayni sahne 17 ile 24 arasinda kapsama verdi. '
                         'Geometrik kol bedava oldugu icin tekrar ucuz.')
    ap.add_argument('--limit', type=int, default=0,
                    help='N bakis acisi, esit aralikli (gunluk kotayi bolmek icin)')
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

    # Refuse to start without a camera. A batch once ran all ten viewpoints
    # against no image at all -- the bridge had failed to launch, from a
    # relative path resolved in the wrong directory -- and reported "kare yok"
    # ten times as though that were a measurement. Checking once at the top
    # turns a silently empty run into a message that says what to fix.
    if fresh(node, g, skip=0) is None:
        rclpy.shutdown()
        pose.stop()
        sys.exit("/camera/image_raw'dan kare gelmiyor -- kopru calisiyor mu?\n"
                 "  ros2 launch /home/taha/DRONIONS/ros/launch/"
                 "dronions_px4_bridge.launch.py")
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

    out_csv = out_path_for(args.model, args.no_llm)
    prior, done = load_done(out_csv)
    os.makedirs('logs', exist_ok=True)
    writer = RowWriter(out_csv, prior)
    rows = []
    sweeps = args.repeat if args.no_llm else 1
    if sweeps > 1:
        print(f"{sweeps} tekrar havuzlanacak\n")
    all_vps = [(b, r) for b in VIEW_BEARINGS for r in VIEW_RANGES]
    viewpoints = []
    for b, rng in all_vps:
        aa = math.radians(b)
        dxy = (TARGET_XY[0] + rng * math.cos(aa), TARGET_XY[1] + rng * math.sin(aa))
        if not line_blocked(dxy, TARGET_XY):
            viewpoints.append((b, rng))
    blocked = len(all_vps) - len(viewpoints)
    if done and not args.no_llm:
        before = len(viewpoints)
        # `mismatch` is what the CSV records, and it is derived from the
        # bearing, so the key has to be recomputed the same way rather than
        # compared against the raw bearing.
        def key_of(b, r):
            aa = math.radians(b)
            yw = math.atan2(-r * math.sin(aa), -r * math.cos(aa))
            mm = math.degrees(math.atan2(math.sin(yw - USER_YAW),
                                         math.cos(yw - USER_YAW)))
            return (round(mm, 1), float(r))
        viewpoints = [v for v in viewpoints if key_of(*v) not in done]
        print(f"{before - len(viewpoints)} bakis acisi onceki partilerde "
              f"olculmus, atlaniyor ({len(viewpoints)} kaldi)\n")

    if args.limit and args.limit < len(viewpoints):
        # Evenly spaced, not the first N. The list is ordered by bearing, so
        # a prefix would spend the whole quota on one side of the target and
        # measure nothing about the mismatch the experiment exists to sweep.
        step = len(viewpoints) / args.limit
        viewpoints = [viewpoints[int(i * step)] for i in range(args.limit)]
    print(f"{len(viewpoints)} bakis acisi "
          f"({len(VIEW_BEARINGS)} yon x {len(VIEW_RANGES)} menzil, "
          f"{blocked} tanesi duvar arkasinda kaldigi icin elendi)"
          + (f", ilk {args.limit} tanesi" if args.limit else "") + "\n")

    for sweep in range(sweeps):
        for bearing, view_range in viewpoints:
            a = math.radians(bearing)
            dx, dy = view_range * math.cos(a), view_range * math.sin(a)
            drone_xy = (TARGET_XY[0] + dx, TARGET_XY[1] + dy)
            # Deterministic per viewpoint, so a resumed batch reproduces the
            # same geometry as the one it is continuing.
            rng = random.Random((YAW_OFFSET_SEED, round(bearing, 1), view_range).__hash__())
            yaw_offset = rng.uniform(-YAW_OFFSET_MAX, YAW_OFFSET_MAX)
            yaw = math.atan2(-dy, -dx) + yaw_offset
            mismatch = math.degrees(math.atan2(math.sin(yaw - USER_YAW),
                                               math.cos(yaw - USER_YAW)))
            if args.delay:
                time.sleep(args.delay)
            teleport(drone_xy[0], drone_xy[1], VIEW_ALT, yaw)
            settled(pose, drone_xy)
            img = fresh(node, g)
            # Where the drone *is*, read at the moment the frame arrived, not
            # where it was told to go. The two differ because the airframe is
            # unarmed and falling; measured at 0.47 m by 0.3 s.
            actual = pose.latest() if pose else None
            drone_xyz = actual if actual else (*drone_xy, VIEW_ALT)
            row = {'mismatch_deg': round(mismatch, 1),
                   'range_m': view_range,
                   'yaw_offset_deg': round(math.degrees(yaw_offset), 1),
                   'true_dist': round(true_d, 2),
                   'true_brg': round(math.degrees(true_b), 1),
                   'alt_cmd': VIEW_ALT,
                   'alt_actual': round(drone_xyz[2], 3) if actual else None,
                   'alt_drop': round(VIEW_ALT - drone_xyz[2], 3) if actual else None}

            # --- geometric ---
            gd = gb = None
            reason = None
            if img is None:
                reason = 'kare yok'
            else:
                raw = det.detect(img)
                cands = filter_candidates(raw)
                # The same physical-size gate the flight pipeline runs. Without it
                # this measures a system we no longer fly: the wall is the top
                # candidate in 71% of surveyed viewpoints, and a wall detection
                # scored as the box is what produced 170-190 degree errors here.
                quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
                n_raw, n_filt = len(raw), len(cands)
                cands = [c for c in cands
                         if size_plausible(c, drone_xyz, quat, TARGET_NAME)]
                # Which stage lost the viewpoint, recorded per row. Sixteen of
                # thirty-seven produced an answer and nothing said why the rest
                # did not -- a yield that low changes what the experiment can
                # claim, so it has to be attributable rather than guessed at.
                if n_raw == 0:
                    reason = 'tespit yok'
                elif n_filt == 0:
                    reason = 'filtre eledi'
                elif not cands:
                    reason = 'boyut kapisi eledi'
                # Take the best candidate that actually yields a usable
                # estimate, not simply the most confident one. Confidence
                # alone hands over the wall in 71% of surveyed viewpoints, and
                # a wall's ray is shallow enough that the range guard kills it
                # -- at which point size_plausible has no width to judge and
                # abstains, so the bad candidate passes the gate and dies at
                # projection instead. Ten viewpoints in a row declined that
                # way. Checking whether the geometry closes is information the
                # system genuinely has, and using it is the same thing the
                # flight pipeline does when it asks the model to choose.
                est = None
                for cand in cands:
                    est = locate_target(cand, drone_xyz, quat, plane_z=0.0)
                    if est:
                        cands = [cand] + [c for c in cands if c is not cand]
                        break
                if cands:
                    if est:
                        ed, eb = relative_to_user(est[:2], USER_POSITION, USER_YAW)
                        gd = abs(ed - true_d)
                        gb = abs(math.degrees(eb - true_b))
                        row['geom_area'] = round(cands[0].relative_area, 4)
                    else:
                        reason = 'isin yere ulasmadi'
            row['geom_dist_err'] = gd
            row['geom_brg_err'] = gb
            row['geom_reason'] = reason

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
                    # The answer itself, not only its error. An arm that
                    # scores well by emitting one constant value is
                    # indistinguishable from one that reasons, unless what it
                    # said is on record -- the posed arm returned exact zeros
                    # often enough to need checking.
                    row['llm_naive_dist'] = round(got[0], 2)
                    row['llm_naive_brg'] = round(got[1], 1)
                    nd = abs(got[0] - true_d)
                    nb = abs(((got[1] - math.degrees(true_b) + 180) % 360) - 180)
                else:
                    row['llm_naive_note'] = txt_n[:100]

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
                    row['llm_posed_dist'] = round(got[0], 2)
                    row['llm_posed_brg'] = round(got[1], 1)
                    pd = abs(got[0] - true_d)
                    pb = abs(((got[1] - math.degrees(true_b) + 180) % 360) - 180)
                else:
                    row['llm_posed_note'] = txt_p[:100]
            row.update(llm_naive_dist_err=nd, llm_naive_brg_err=nb,
                       llm_posed_dist_err=pd, llm_posed_brg_err=pb)
            rows.append(row)
            writer.write(row)

            def f(v, w, unit=''):
                return f"{v:{w}.1f}{unit}" if v is not None else f"{'--':>{w}}"
            print(f"{mismatch:9.0f} | {f(gd,7)} {f(gb,7)} | "
                  f"{f(nd,8)} {f(nb,8)} | {f(pd,8)} {f(pb,8)}")

    print("-" * len(hdr))

    def med(key):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        return vals[len(vals) // 2] if vals else None

    # Coverage is reported alongside accuracy, because refusing to answer is a
    # design decision here, not a shortfall. The geometric method declines when
    # the ray is too shallow to trust -- a viewpoint it stays silent on is one
    # where it would have had to invent a number for someone who cannot check
    # it. A language model has no such notion and answers every time. Comparing
    # only the accuracy of the answers given would flatter whichever arm is
    # readier to guess, so both figures belong together.
    total = len(rows)
    for name, dk, bk in (("geometric", 'geom_dist_err', 'geom_brg_err'),
                         ("llm naive", 'llm_naive_dist_err', 'llm_naive_brg_err'),
                         ("llm posed", 'llm_posed_dist_err', 'llm_posed_brg_err')):
        d, b = med(dk), med(bk)
        n = sum(1 for r in rows if r.get(dk) is not None)
        cov = 100.0 * n / total if total else 0.0
        if d is None:
            print(f"{name:>10}: cevap yok            "
                  f"| kapsama {n:2}/{total} ({cov:3.0f}%)")
        else:
            print(f"{name:>10}: mesafe {d:5.2f} m, yon {b:5.1f} derece "
                  f"| kapsama {n:2}/{total} ({cov:3.0f}%)")

    declined = Counter(r.get('geom_reason') for r in rows
                       if r.get('geom_brg_err') is None and r.get('geom_reason'))
    if declined:
        print("\n  geometrik yontemin cevap vermedigi durumlar:")
        for why, k in declined.most_common():
            print(f"    {why:24}: {k}")
        print("    (cevap vermemek tasarim geregi -- kontrol edemeyecek birine "
              "uydurma sayi soylememek icin)")

    writer.close()
    rows = prior + rows
    remaining = len([1 for b in VIEW_BEARINGS for r in VIEW_RANGES]) - len(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} satir toplam)")
    if not args.no_llm and remaining > 0:
        print(f"kalan bakis acisi: ~{remaining}, "
              f"gunluk 20 cagri ile ~{(remaining * 2 + 19) // 20} gun")

    node.destroy_node()
    rclpy.shutdown()


main()
