#!/usr/bin/env python3
"""
Görevi: Gazebo'nun gerçek drone konumunu kaydeder, MAVROS'unkiyle karşılaştırmak için.

Why: a run put the drone at (-5.9, 1.2) sixteen seconds after it was at
(0.7, 0.6) -- 6.6 m -- while it was in the centring phase, which commands no
translation at all. Either it really flew there, or the position estimate
jumped. Nothing in the logs can tell those apart, because every position in
them comes from the same estimator that is under suspicion.

Gazebo knows the truth. This streams it alongside the flight so the two can be
laid side by side afterwards.

Deliberately a *stream*, not polling. `gz model -p` costs 5.1 s per call and in
an earlier campaign starved frame capture badly enough that 65% of samples had
no camera frame at all; subscribing to the pose topic costs nothing.

  scripts/log_ground_truth.py                    # runs until Ctrl-C
  scripts/log_ground_truth.py --compare          # align with the run log
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / "logs" / "ground_truth.csv"
RUN_LOG = REPO / "logs" / "dronions_run.log"

MODEL = "x500_dronions"
SAMPLE_INTERVAL = 0.5           # s between recorded samples

_NAME = re.compile(r'name:\s*"([^"]+)"')
_X = re.compile(r'^\s*x:\s*(-?[\d.eE+-]+)')
_Y = re.compile(r'^\s*y:\s*(-?[\d.eE+-]+)')
_Z = re.compile(r'^\s*z:\s*(-?[\d.eE+-]+)')


def find_world() -> str:
    """World name from gz's own topic list, rather than hardcoded."""
    try:
        out = subprocess.run(["gz", "topic", "-l"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"gz topic -l calistirilamadi: {e}")
    for line in out.splitlines():
        m = re.match(r'/world/([^/]+)/pose/info', line.strip())
        if m:
            return m.group(1)
    sys.exit("pose/info konusu bulunamadi -- simulasyon calisiyor mu? "
             "(dynamic_pose/info degil, pose/info aranıyor)")


def stream(world: str):
    topic = f"/world/{world}/pose/info"
    print(f"[gt] {topic} dinleniyor, {OUT_CSV} yaziliyor. Ctrl-C ile durur.")
    proc = subprocess.Popen(["gz", "topic", "-e", "-t", topic],
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    last_write = 0.0
    cur_name, pos = None, {}
    with open(OUT_CSV, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["wall_clock", "x", "y", "z"])
        try:
            for line in proc.stdout:
                m = _NAME.search(line)
                if m:
                    cur_name, pos = m.group(1), {}
                    continue
                if cur_name != MODEL:
                    continue
                for key, rx in (("x", _X), ("y", _Y), ("z", _Z)):
                    mm = rx.match(line)
                    if mm and key not in pos:
                        pos[key] = float(mm.group(1))
                # A pose block carries position before orientation, so three
                # components mean this one is complete.
                if len(pos) == 3:
                    now = time.time()
                    if now - last_write >= SAMPLE_INTERVAL:
                        w.writerow([datetime.now().strftime("%H:%M:%S"),
                                    f"{pos['x']:.3f}", f"{pos['y']:.3f}",
                                    f"{pos['z']:.3f}"])
                        f.flush()
                        last_write = now
                    cur_name, pos = None, {}
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
    print(f"\n[gt] -> {OUT_CSV}")


def compare():
    """Lay Gazebo truth next to what the node logged from MAVROS."""
    if not OUT_CSV.exists():
        sys.exit(f"{OUT_CSV} yok -- once kayit al.")
    truth = {}
    with open(OUT_CSV, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            truth[r["wall_clock"]] = (float(r["x"]), float(r["y"]))
    if not truth:
        sys.exit("yer gercegi bos.")

    rx = re.compile(r'^\[[\d-]+ ([\d:]+)\] Arama konumu \((-?[\d.]+), (-?[\d.]+)\)')
    rows = []
    for line in open(RUN_LOG, encoding='utf-8', errors='replace'):
        m = rx.match(line)
        if m and m.group(1) in truth:
            t = m.group(1)
            ex, ey = float(m.group(2)), float(m.group(3))
            tx, ty = truth[t]
            rows.append((t, ex, ey, tx, ty, ((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5))
    if not rows:
        sys.exit("Ortak zaman damgasi yok -- kayit ucusla ayni anda mi alindi?")

    print(f"{'saat':>9} {'MAVROS':>15} {'Gazebo':>15} {'fark':>7}")
    prev = None
    for t, ex, ey, tx, ty, d in rows:
        jump = ""
        if prev is not None and abs(d - prev) > 1.0:
            jump = "  <-- SICRAMA"
        print(f"{t:>9} {ex:7.2f},{ey:6.2f} {tx:7.2f},{ty:6.2f} {d:6.2f} m{jump}")
        prev = d
    errs = sorted(r[5] for r in rows)
    print(f"\nn={len(rows)}  medyan fark {errs[len(errs)//2]:.2f} m  "
          f"maks {errs[-1]:.2f} m")
    print("Fark kucuk ve sabitse kestirim saglam; bir anda buyuyup oyle "
          "kaliyorsa origin kaymasidir, ucus degil.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        stream(find_world())


if __name__ == "__main__":
    main()
