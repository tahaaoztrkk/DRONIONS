#!/usr/bin/env python3
"""
Görevi: Aramayı N kez tekrarlayıp güvenilirliğini ölçer.

Why this is separate from every other experiment here: the two end-to-end
failures on record were both the Gemini API -- a retired model answering 404,
and the daily quota running out -- and neither was a failure of the search. So
"how reliable is the system" and "how reliable is the search" could not be told
apart, and the free tier's 20 requests/day per model caps honest repetition at
roughly one run a day.

Running with DRONIONS_NO_VLM=1 removes the API from the loop entirely: the node
accepts YOLO's own best candidate, and the run costs nothing. What that buys is
the search measured on its own -- how long the sweep takes to put the target in
frame, and how often the detector alone locks the right object. The second
number is interesting precisely because it is expected to be poor: it is the
size of the job the VLM is doing, measured rather than asserted.

Scoring is against the scenario's known object positions. Every one of them is
static, so there is no need to query Gazebo at all -- which matters, because
`gz model -p` costs 5.1 s per call and starved frame capture badly enough in an
earlier campaign that 65% of samples had no frame.

  scripts/experiment_repeatability.py            # 20 runs, no VLM
  scripts/experiment_repeatability.py -n 5
  scripts/experiment_repeatability.py --with-vlm # spends quota; ~1 run/day
  scripts/experiment_repeatability.py --report   # re-score an existing CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_LOG = REPO / "logs" / "dronions_run.log"
OUT_CSV = REPO / "logs" / "repeatability.csv"
CHAIN = REPO / "scripts" / "run_sim_chain.sh"

# Ground truth from Tools/simulation/gz/worlds/dronions_scenario.sdf. All static.
# The wall is 4 m long, so scoring it as a point put estimates "1.1 m from the
# wall" that were in fact 0.5 m from its face; it is measured to the nearest
# point on its span instead.
TRUTH = {
    "box":       (3.5, 2.0),
    "sphere":    (3.5, 3.6),
    "blue_box":  (3.5, 0.4),
}
WALL_X = 1.75
WALL_Y_SPAN = (-0.5, 3.5)


def dist_to_wall(x, y):
    cy = min(max(y, WALL_Y_SPAN[0]), WALL_Y_SPAN[1])
    return ((x - WALL_X) ** 2 + (y - cy) ** 2) ** 0.5


# A lock is scored as correct if the estimate lands nearer the real box than
# this. Chosen from the measured localization error (0.49 m median, n=21) plus
# room for the box's own extent -- tight enough that the wall at 1.75 m or the
# blue distractor 1.6 m away cannot be mistaken for a hit.
HIT_RADIUS = 1.0

TARGET = "box"

_TS = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')


def _t(line):
    m = _TS.match(line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None


def parse_run(lines):
    """One run's log lines -> a row of metrics."""
    row = {
        "found": 0, "arrived": 0, "gave_up": 0, "aborted": 0,
        "seconds_to_answer": "", "est_x": "", "est_y": "",
        "error_m": "", "locked": "", "correct": 0,
        "climbs": 0, "strayed": 0, "vlm_calls": 0, "vlm_errors": 0,
    }
    t_start = None
    for l in lines:
        if ' aranıyor.' in l and '] > ' in l:
            t_start = _t(l)
        elif 'Engel asilamiyor' in l:
            row["climbs"] += 1
        elif 'Arama alani disinda' in l:
            row["strayed"] += 1
        elif 'Gemini Cevabı' in l:
            row["vlm_calls"] += 1
            if 'API Hatası' in l:
                row["vlm_errors"] += 1
        elif 'Hedef konumu (dunya)' in l:
            m = re.search(r'x=(-?[\d.]+) y=(-?[\d.]+)', l)
            if m and not row["est_x"]:
                ex, ey = float(m.group(1)), float(m.group(2))
                row["est_x"], row["est_y"] = f"{ex:.2f}", f"{ey:.2f}"
                dists = {k: ((ex - vx) ** 2 + (ey - vy) ** 2) ** 0.5
                         for k, (vx, vy) in TRUTH.items()}
                dists["wall"] = dist_to_wall(ex, ey)
                row["locked"] = min(dists, key=dists.get)
                err = dists[TARGET]
                row["error_m"] = f"{err:.2f}"
                row["correct"] = int(err <= HIT_RADIUS)
                row["found"] = 1
                if t_start:
                    row["seconds_to_answer"] = f"{(_t(l) - t_start).total_seconds():.0f}"
        elif 'HEDEFE VARILDI' in l:
            row["arrived"] = 1
        elif 'bulunamadı. Aradığım alanı' in l:
            row["gave_up"] = 1
        elif 'Aramayı durduruyorum' in l and 'göremiyorum' in l:
            row["aborted"] = 1
    return row


def split_runs(text):
    runs, cur = [], None
    for l in text.splitlines():
        if 'Initializing DRONIONS' in l:
            cur = []
            runs.append(cur)
        if cur is not None:
            cur.append(l)
    return runs


def one_run(idx, total, with_vlm, timeout):
    env = dict(os.environ, HEADLESS="1", DRONIONS_TARGET=TARGET, COMMAND_DELAY="45")
    if not with_vlm:
        env["DRONIONS_NO_VLM"] = "1"
    env.pop("INTERACTIVE", None)
    before = RUN_LOG.stat().st_size if RUN_LOG.exists() else 0
    print(f"[{idx}/{total}] basliyor {datetime.now():%H:%M:%S} ...", flush=True)
    t0 = time.time()
    try:
        subprocess.run(["bash", str(CHAIN)], env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    subprocess.run("pkill -9 -f 'gz sim|bin/px4|mavros_node|parameter_bridge|"
                   "dronions_ros_node_px4' 2>/dev/null", shell=True)
    time.sleep(3)
    if not RUN_LOG.exists():
        return None
    with open(RUN_LOG, encoding='utf-8', errors='replace') as f:
        f.seek(before)
        fresh = f.read()
    runs = split_runs(fresh)
    if not runs:
        print(f"[{idx}/{total}] log bulunamadi ({time.time() - t0:.0f}s)")
        return None
    row = parse_run(runs[-1])
    print(f"[{idx}/{total}] {'BULDU' if row['found'] else 'bulamadi'} "
          f"{row['seconds_to_answer'] or '-'}s kilit={row['locked'] or '-'} "
          f"hata={row['error_m'] or '-'}m tirmanma={row['climbs']}", flush=True)
    return row


def report(rows):
    if not rows:
        print("Satir yok.")
        return
    n = len(rows)
    found = [r for r in rows if int(r["found"])]
    correct = [r for r in rows if int(r["correct"])]
    times = [float(r["seconds_to_answer"]) for r in found if r["seconds_to_answer"]]
    errs = [float(r["error_m"]) for r in correct if r["error_m"]]

    print(f"\n{'='*58}\nn = {n} kosu\n{'='*58}")
    print(f"  bir seye kilitlendi : {len(found)}/{n}  ({100*len(found)/n:.0f}%)")
    print(f"  DOGRU nesneye       : {len(correct)}/{n}  ({100*len(correct)/n:.0f}%)")
    print(f"  hedefe vardi        : {sum(int(r['arrived']) for r in rows)}/{n}")
    print(f"  pes etti            : {sum(int(r['gave_up']) for r in rows)}/{n}")
    print(f"  gorus hatasi        : {sum(int(r['aborted']) for r in rows)}/{n}")
    if times:
        print(f"\n  cevaba kadar gecen sure (s): medyan {statistics.median(times):.0f}"
              f"  min {min(times):.0f}  maks {max(times):.0f}")
    if errs:
        print(f"  konum hatasi (m)           : medyan {statistics.median(errs):.2f}"
              f"  maks {max(errs):.2f}")
    wrong = {}
    for r in rows:
        if r["locked"] and not int(r["correct"]):
            wrong[r["locked"]] = wrong.get(r["locked"], 0) + 1
    if wrong:
        print("\n  yanlis kilitlenmeler: "
              + ", ".join(f"{k} x{v}" for k, v in sorted(wrong.items(), key=lambda x: -x[1])))
    strays = sum(int(r["strayed"]) for r in rows)
    print(f"\n  alan disina cikma   : {strays} olay / {n} kosu")
    print(f"  tirmanma            : {sum(int(r['climbs']) for r in rows)} olay")
    calls = sum(int(r["vlm_calls"]) for r in rows)
    print(f"  VLM cagrisi         : {calls} ({sum(int(r['vlm_errors']) for r in rows)} hata)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--with-vlm", action="store_true",
                    help="Gemini'yi de kullan (kota harcar, ~1 kosu/gun)")
    ap.add_argument("--timeout", type=int, default=420,
                    help="kosu basina saniye siniri")
    ap.add_argument("--report", action="store_true",
                    help="yeni kosu yapma, mevcut CSV'yi yeniden ozetle")
    a = ap.parse_args()

    if a.report:
        if not OUT_CSV.exists():
            sys.exit(f"{OUT_CSV} yok.")
        with open(OUT_CSV, newline='', encoding='utf-8') as f:
            report(list(csv.DictReader(f)))
        return

    if not shutil.which("bash") or not CHAIN.exists():
        sys.exit("run_sim_chain.sh bulunamadi.")

    rows = []
    fields = list(parse_run([]).keys())
    with open(OUT_CSV, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["run"] + fields)
        w.writeheader()
        for i in range(1, a.n + 1):
            r = one_run(i, a.n, a.with_vlm, a.timeout)
            if r is None:
                continue
            rows.append(r)
            w.writerow({"run": i, **r})
            f.flush()          # crash or Ctrl-C keeps what has run so far
    print(f"\n-> {OUT_CSV}")
    report(rows)


if __name__ == "__main__":
    main()
