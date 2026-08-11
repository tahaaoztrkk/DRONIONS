#!/usr/bin/env python3
"""
Görevi: Aramayı VLM'siz koşturup dedektörün tek başına ne kadar yettiğini ölçer.

Why this exists: the two end-to-end failures on record were both the Gemini API
-- a retired model answering 404, and the daily quota running out -- and neither
was a failure of the search. "How reliable is the system" and "how reliable is
the search" could not be told apart, and at 20 requests/day per model honest
repetition was capped at roughly one run a day.

DRONIONS_NO_VLM=1 puts the node in survey mode: it flies the whole sweep and
logs what the detector ranked at every check, without ever handing off. A run
costs nothing and yields a sample from every place the search actually looks.

The first design instead auto-accepted YOLO's best candidate, and measured
almost nothing. The wall fills 68% of the frame from the takeoff spot, so every
run ended one second into the search having locked onto it -- repeating that
gives copies of one viewpoint, not a distribution. Hence: never hand off, and
score every sample.

What comes out is the size of the job the VLM stage is doing, measured across
the sweep rather than asserted: how often the detector's top-ranked candidate
is the real box, how often it is the wall, and how much of the time the box is
detectable at all.

Scoring is against the scenario's static object positions, so Gazebo is never
queried -- `gz model -p` costs 5.1 s per call and starved frame capture badly
enough in an earlier campaign that 65% of samples had no frame.

  scripts/experiment_repeatability.py            # 5 runs, survey mode
  scripts/experiment_repeatability.py -n 10
  scripts/experiment_repeatability.py --report   # re-score the existing CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_LOG = REPO / "logs" / "dronions_run.log"
OUT_CSV = REPO / "logs" / "repeatability.csv"
CHAIN = REPO / "scripts" / "run_sim_chain.sh"

# Ground truth from Tools/simulation/gz/worlds/dronions_scenario.sdf. All static.
TRUTH = {
    "box":      (3.5, 2.0),
    "sphere":   (3.5, 3.6),
    "blue_box": (3.5, 0.4),
}
# The wall is 4 m long, so scoring it as a point called estimates "1.1 m from
# the wall" that were in fact 0.5 m from its face.
WALL_X, WALL_Y_SPAN = 1.75, (-0.5, 3.5)

# A detection counts as the box within this distance. From the measured
# localization error (0.49 m median, n=21) plus the box's own extent, and still
# tight enough that the wall and the blue distractor 1.6 m away cannot pass.
HIT_RADIUS = 1.0

TARGET = "box"

_TS = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
_SURVEY = re.compile(
    r'ANKET r=(\d+) conf=([\d.]+) alan=([\d.]+) '
    r'dunya=(-?[\d.]+),(-?[\d.]+) drone=(-?[\d.]+),(-?[\d.]+)')


def _t(line):
    m = _TS.match(line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None


def nearest_object(x, y):
    d = {k: ((x - vx) ** 2 + (y - vy) ** 2) ** 0.5 for k, (vx, vy) in TRUTH.items()}
    cy = min(max(y, WALL_Y_SPAN[0]), WALL_Y_SPAN[1])
    d["wall"] = ((x - WALL_X) ** 2 + (y - cy) ** 2) ** 0.5
    name = min(d, key=d.get)
    return name, d[name], d[TARGET]


def parse_survey(lines, run_id):
    """One run's log -> one row per ranked detection."""
    rows, t0 = [], None
    for l in lines:
        if ' aranıyor.' in l and '] > ' in l:
            t0 = _t(l)
            continue
        m = _SURVEY.search(l)
        if not m or t0 is None:
            continue
        rank, conf, area, wx, wy, dx, dy = (
            int(m.group(1)), float(m.group(2)), float(m.group(3)),
            float(m.group(4)), float(m.group(5)),
            float(m.group(6)), float(m.group(7)))
        name, _, box_err = nearest_object(wx, wy)
        rows.append({
            "run": run_id,
            "t": f"{(_t(l) - t0).total_seconds():.0f}",
            "rank": rank, "conf": f"{conf:.3f}", "area": f"{area:.4f}",
            "world_x": f"{wx:.2f}", "world_y": f"{wy:.2f}",
            "drone_x": f"{dx:.2f}", "drone_y": f"{dy:.2f}",
            "nearest": name, "box_err": f"{box_err:.2f}",
            "is_box": int(box_err <= HIT_RADIUS),
        })
    return rows


def split_runs(text):
    runs, cur = [], None
    for l in text.splitlines():
        if 'Initializing DRONIONS' in l:
            cur = []
            runs.append(cur)
        if cur is not None:
            cur.append(l)
    return runs


def one_run(idx, total, timeout):
    env = dict(os.environ, HEADLESS="1", DRONIONS_TARGET=TARGET,
               COMMAND_DELAY="45", DRONIONS_NO_VLM="1")
    env.pop("INTERACTIVE", None)
    before = RUN_LOG.stat().st_size if RUN_LOG.exists() else 0
    print(f"[{idx}/{total}] basliyor {datetime.now():%H:%M:%S} ...", flush=True)
    try:
        subprocess.run(["bash", str(CHAIN)], env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    subprocess.run("pkill -9 -f 'gz sim|bin/px4|mavros_node|parameter_bridge|"
                   "dronions_ros_node_px4' 2>/dev/null", shell=True)
    time.sleep(3)
    if not RUN_LOG.exists():
        return []
    with open(RUN_LOG, encoding='utf-8', errors='replace') as f:
        f.seek(before)
        runs = split_runs(f.read())
    if not runs:
        print(f"[{idx}/{total}] log bulunamadi")
        return []
    rows = parse_survey(runs[-1], idx)
    tops = [r for r in rows if r["rank"] == 0]
    hits = [r for r in tops if int(r["is_box"])]
    print(f"[{idx}/{total}] {len(rows)} ornek, {len(tops)} bakis; "
          f"en ust aday kutu: {len(hits)}/{len(tops) or 1}", flush=True)
    return rows


def report(rows):
    if not rows:
        print("Ornek yok.")
        return
    runs = sorted({r["run"] for r in rows}, key=str)
    tops = [r for r in rows if int(r["rank"]) == 0]
    if not tops:
        print("Siralanmis tespit yok.")
        return
    any_box = {}
    for r in rows:
        key = (r["run"], r["t"])
        any_box[key] = any_box.get(key, 0) or int(r["is_box"])

    print(f"\n{'='*60}")
    print(f"{len(runs)} kosu, {len(tops)} bakis, {len(rows)} siralanmis tespit")
    print('='*60)

    top_box = sum(int(r["is_box"]) for r in tops)
    print(f"\n  EN UST ADAY DOGRU MU  (VLM'in yaptigi isin buyuklugu)")
    print(f"    kutu        : {top_box}/{len(tops)}  ({100*top_box/len(tops):.0f}%)")
    for name, k in Counter(r["nearest"] for r in tops
                           if not int(r["is_box"])).most_common():
        print(f"    {name:12}: {k}/{len(tops)}  ({100*k/len(tops):.0f}%)")

    seen = sum(1 for v in any_box.values() if v)
    print(f"\n  KUTU ILK 3 ADAY ICINDE : {seen}/{len(any_box)} bakis "
          f"({100*seen/len(any_box):.0f}%)")

    firsts = []
    for run in runs:
        ts = [float(r["t"]) for r in rows if r["run"] == run and int(r["is_box"])]
        firsts.append(min(ts) if ts else None)
    got = [f for f in firsts if f is not None]
    line = f"  KUTUYU ILK GORME       : {len(got)}/{len(runs)} kosu"
    if got:
        line += (f", medyan {statistics.median(got):.0f}s "
                 f"(min {min(got):.0f}, maks {max(got):.0f})")
    print(line)

    if len(runs) > 1:
        per_run = []
        for run in runs:
            t = [r for r in tops if r["run"] == run]
            per_run.append(sum(int(r["is_box"]) for r in t) / len(t) if t else 0.0)
        print(f"\n  kosular arasi dagilim  : "
              + " ".join(f"{100*p:.0f}%" for p in per_run))

    box = [r for r in rows if int(r["is_box"])]
    wall = [r for r in rows if r["nearest"] == "wall"]
    if box and wall:
        print(f"\n  guven : kutu {statistics.median(float(r['conf']) for r in box):.3f}"
              f" | duvar {statistics.median(float(r['conf']) for r in wall):.3f}")
        print(f"  alan  : kutu {statistics.median(float(r['area']) for r in box):.4f}"
              f" | duvar {statistics.median(float(r['area']) for r in wall):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        if not OUT_CSV.exists():
            raise SystemExit(f"{OUT_CSV} yok.")
        with open(OUT_CSV, newline='', encoding='utf-8') as f:
            report(list(csv.DictReader(f)))
        return

    fields = ["run", "t", "rank", "conf", "area", "world_x", "world_y",
              "drone_x", "drone_y", "nearest", "box_err", "is_box"]
    all_rows = []
    with open(OUT_CSV, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(1, a.n + 1):
            rows = one_run(i, a.n, a.timeout)
            all_rows += rows
            w.writerows(rows)
            f.flush()          # Ctrl-C keeps what has run so far
    print(f"\n-> {OUT_CSV}")
    report(all_rows)


if __name__ == "__main__":
    main()
