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
  scripts/experiment_repeatability.py --approach -n 10   # approach phase only

--approach uses a stand-in for the VLM: the best size-plausible candidate is
accepted and the run proceeds through centring and the approach. Identification
is meaningless in that mode -- it will accept a distractor -- but everything
after confirmation is exercised for free.

That is the half that survey mode cannot reach, and the half where the failures
are intermittent: flying into the wall, announcing arrival early, and losing a
confirmation caught at the frame edge each happen on some runs and not others.
A single flight cannot tell a fix from a lucky roll.
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
APPROACH_CSV = REPO / "logs" / "approach.csv"
CHAIN = REPO / "scripts" / "run_sim_chain.sh"

# Ground truth per world. All static, so this is read from the SDF once and
# never queried at run time -- `gz model -p` costs 5.1 s a call and starved
# frame capture badly enough in an earlier campaign that 65% of samples had no
# frame.
#
# Furniture is in the tables, not just the targets. An estimate has to land
# somewhere, and "which object is it nearest" is the question that separates a
# detection of the mug from a detection of the table it stands on. Leaving the
# furniture out makes every stray estimate look like a near miss on the target.
SCENARIO_TRUTH = {
    "box":      (3.5, 2.0),
    "sphere":   (3.5, 3.6),
    "blue_box": (3.5, 0.4),
}
ROOM_TRUTH = {
    # The five labelled targets.
    "laptop":     (3.05,  0.22),
    "book":       (3.05, -0.22),
    "mug":        (2.62,  0.24),
    "phone":      (2.62, -0.16),
    "floor_box":  (0.10,  1.10),
    # Everything else an estimate can plausibly land on.
    "bowl":       (2.18,  0.20),
    "headphones": (0.50, -1.20),
    "bottle":     (1.40, -0.60),
    "table":      (2.60,  0.00),
    "sofa":       (0.50, -1.35),
    "bookshelf":  (-1.20, 0.80),
    "cabinet":    (1.20,  1.60),
    "chair":      (1.50,  0.90),
}

# The scenario world's test wall, which is 4 m long: scoring it as a point
# called estimates "1.1 m from the wall" that were in fact 0.5 m from its face.
WALL_X, WALL_Y_SPAN = 1.75, (-0.5, 3.5)

# How near an estimate has to be to be credited to the target at all.
#
# The scenario's 1.0 m came from a 0.49 m median localisation error and a box
# large enough to absorb it. The room cannot use that number: the laptop and
# the book are 0.44 m apart, so a metre-wide credit would score one as the
# other, which is the exact confusion this whole line of work was about.
# 0.35 m sits above the measured median error (0.11 m) and below half the
# spacing. Identity still comes from which object is nearest; this only stops
# an estimate out on open floor being credited to anything.
HIT_RADIUS_BY_WORLD = {"scenario": 1.0, "room": 0.35}

WORLD = os.getenv("DRONIONS_WORLD", "dronions_room")
IS_ROOM = "room" in WORLD
TRUTH = ROOM_TRUTH if IS_ROOM else SCENARIO_TRUTH
HIT_RADIUS = HIT_RADIUS_BY_WORLD["room" if IS_ROOM else "scenario"]

TARGET = os.getenv("DRONIONS_TARGET", "book" if IS_ROOM else "box")

_TS = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
# Fields are matched by name with anything allowed between them, because the
# line has gained columns twice (implied width, then hue) and a positional
# pattern silently matches nothing when that happens -- which reads exactly
# like a campaign that collected no data.
_SURVEY = re.compile(
    r'ANKET r=(?P<rank>\d+).*?conf=(?P<conf>[\d.]+).*?alan=(?P<area>[\d.]+)'
    r'.*?dunya=(?P<wx>-?[\d.]+),(?P<wy>-?[\d.]+)'
    r'.*?drone=(?P<dx>-?[\d.]+),(?P<dy>-?[\d.]+)')
_SURVEY_WIDTH = re.compile(r'genislik=(-?[\d.]+)')
_SURVEY_HUE = re.compile(r'ton=(-?[\d.]+)')


def _t(line):
    m = _TS.match(line)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None


def nearest_object(x, y):
    """(nearest object, its distance, distance to the target)."""
    d = {k: ((x - vx) ** 2 + (y - vy) ** 2) ** 0.5 for k, (vx, vy) in TRUTH.items()}
    if not IS_ROOM:
        cy = min(max(y, WALL_Y_SPAN[0]), WALL_Y_SPAN[1])
        d["wall"] = ((x - WALL_X) ** 2 + (y - cy) ** 2) ** 0.5
    name = min(d, key=d.get)
    return name, d[name], d.get(TARGET, float('inf'))


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
        g = m.groupdict()
        wx, wy = float(g["wx"]), float(g["wy"])
        mw, mh = _SURVEY_WIDTH.search(l), _SURVEY_HUE.search(l)
        name, _, box_err = nearest_object(wx, wy)
        rows.append({
            "run": run_id,
            "t": f"{(_t(l) - t0).total_seconds():.0f}",
            "rank": int(g["rank"]), "conf": f"{float(g['conf']):.3f}",
            "area": f"{float(g['area']):.4f}",
            "width": f"{float(mw.group(1)):.2f}" if mw else "",
            "hue": f"{float(mh.group(1)):.0f}" if mh else "",
            "world_x": f"{wx:.2f}", "world_y": f"{wy:.2f}",
            "drone_x": f"{float(g['dx']):.2f}", "drone_y": f"{float(g['dy']):.2f}",
            "nearest": name, "box_err": f"{box_err:.2f}",
            # Nearest *and* near enough. Nearest alone credits an estimate on
            # empty floor to whatever happens to be least far from it.
            "is_box": int(name == TARGET and box_err <= HIT_RADIUS),
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


APPROACH_EVENTS = {
    "arrived":        ("HEDEFE VARILDI",        "vardi"),
    "arrival_denied": ("VARIS reddedildi",      "erken varis reddi"),
    "contact":        ("TEMAS RISKI",           "temas riski"),
    "blocked":        ("Yaklasilamiyor",        "engel nedeniyle birakildi"),
    "size_drop":      ("Takip birakildi",       "boyut nedeniyle birakildi"),
    "center_fail":    ("Ortalama basarisiz",    "ortalama basarisiz"),
    "lost":           ("Hedef kaybedildi",      "hedef kaybedildi"),
    "strayed":        ("Arama alani disinda",   "alan disina cikti"),
    "gave_up":        ("bulunamadı. Aradığım",  "pes etti"),
    # Kept separate from every other outcome: a run that ended in a crash
    # measured nothing about the search, and counting it as a search failure
    # is how these looked like a colour-filter regression.
    "pose_broken":    ("Ucus anomalisi",        "UCUS ANOMALISI (carpma)"),
}


_LOCK = re.compile(r'Hedef konumu \(dunya\) x=(-?[\d.]+) y=(-?[\d.]+) '
                   r'\| drone x=(-?[\d.]+) y=(-?[\d.]+)')


OBSTACLES = {"table": (2.60, 0.00), "sofa": (0.50, -1.35),
             "bookshelf": (-1.20, 0.80), "cabinet": (1.20, 1.60),
             "chair": (1.50, 0.90)}


def _obstacle_gap(drone, tgt):
    """Closest the drone->target line passes to something solid, in metres.

    In the scenario world that is the test wall. In the room it is the
    furniture, because the room's walls are its boundary rather than an
    obstacle on any approach -- they were never what the drone had to get past.

    Recorded per run because a campaign whose runs all fly the same line
    measures nothing about geometry-dependent failures, and the first one did
    exactly that without saying so: ten runs, ten identical approaches, the
    wall never nearer than 2.1 m to any of them, and a clean sheet that looked
    like the wall-strike fix working.
    """
    (px, py), (qx, qy) = drone, tgt
    dx, dy = qx - px, qy - py
    L = dx * dx + dy * dy

    def gap_to(wx, wy):
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((wx - px) * dx + (wy - py) * dy) / L))
        return ((px + t * dx - wx) ** 2 + (py + t * dy - wy) ** 2) ** 0.5

    if IS_ROOM:
        return min(gap_to(ox, oy) for ox, oy in OBSTACLES.values())
    # The wall is 4 m long, so it is sampled rather than treated as a point.
    return min(gap_to(WALL_X, WALL_Y_SPAN[0]
                      + i * (WALL_Y_SPAN[1] - WALL_Y_SPAN[0]) / 40)
               for i in range(41))


def parse_approach(lines, run_id):
    """One run's log -> a row of approach-phase outcomes."""
    row = {"run": run_id}
    for key, (needle, _) in APPROACH_EVENTS.items():
        row[key] = sum(1 for l in lines if needle in l)
    row["handoffs"] = sum(1 for l in lines if "takibe geciliyor" in l)

    row.update({"locked": "", "tgt_x": "", "tgt_y": "",
                "drone_x": "", "drone_y": "", "obstacle_gap": ""})
    for l in lines:
        m = _LOCK.search(l)
        if m:
            tx, ty, dx, dy = (float(m.group(i)) for i in (1, 2, 3, 4))
            name, _, _ = nearest_object(tx, ty)
            row.update({"locked": name,
                        "tgt_x": f"{tx:.2f}", "tgt_y": f"{ty:.2f}",
                        "drone_x": f"{dx:.2f}", "drone_y": f"{dy:.2f}",
                        "obstacle_gap": f"{_obstacle_gap((dx, dy), (tx, ty)):.2f}"})
            break
    return row


def report_approach(rows):
    if not rows:
        print("Kosu yok.")
        return
    n = len(rows)
    print(f"\n{'='*58}\n{n} kosu, yaklasma fazi\n{'='*58}")
    print(f"  onaydan sonra takibe gecis : {sum(r['handoffs'] for r in rows)}")
    broken = sum(1 for r in rows if int(r.get("pose_broken", 0)))
    good = n - broken
    print(f"  hedefe varis               : {sum(int(r['arrived']) for r in rows)}"
          f"  ({sum(1 for r in rows if int(r['arrived']))}/{n} kosu)")
    if broken:
        print(f"  !! {broken}/{n} kosuda ucus anomalisi (muhtemelen carpma) -- "
              f"o kosular aramayi olcmez.")
        print(f"     saglikli kosularda varis: "
              f"{sum(1 for r in rows if int(r['arrived']))}/{good}")
    print()
    for key, (_, label) in APPROACH_EVENTS.items():
        if key == "arrived":
            continue
        total = sum(int(r[key]) for r in rows)
        runs_hit = sum(1 for r in rows if int(r[key]))
        flag = "  <-- " if key in ("contact",) and total else ""
        print(f"  {label:28}: {total:3} olay, {runs_hit}/{n} kosu{flag}")

    # Without this the outcomes above cannot be read: a clean sheet means
    # "the fixes held" only if the runs differed enough to test them.
    geo = [r for r in rows if r.get("obstacle_gap")]
    print(f"\n  --- yaklasma cesitliligi ---")
    if not geo:
        print("    kilit kaydi yok")
        return
    gaps = sorted(float(r["obstacle_gap"]) for r in geo)
    blocked = sum(1 for g in gaps if g < 0.5)
    print(f"    engel yol uzerinde   : {blocked}/{len(geo)} kosu "
          f"(<0.5 m); acikliklar {gaps[0]:.1f} - {gaps[-1]:.1f} m")
    print(f"    kilitlenilen nesne   : "
          + ", ".join(f"{k} x{v}" for k, v in
                      Counter(r["locked"] for r in geo).most_common()))
    # Clustering, not spread. Max pairwise distance is dominated by a single
    # outlier -- it read 9.5 m for a campaign in which nine of ten runs flew
    # from the same spot, which is exactly the case this is meant to catch.
    pts = [(float(r["drone_x"]), float(r["drone_y"])) for r in geo]
    med = (statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts))
    near = sum(1 for p in pts
               if ((p[0] - med[0]) ** 2 + (p[1] - med[1]) ** 2) ** 0.5 < 1.0)
    print(f"    yaklasma noktalari   : {len(pts)} kosudan {near} tanesi ayni "
          f"noktada (medyanin 1 m yakininda)")
    if near >= max(2, int(0.7 * len(pts))):
        print("    !! Kosular buyuk olcude ayni yerden yaklasmis. Geometriye "
              "bagli hatalar (duvarin yolda olmasi gibi) sinanmamis demektir; "
              "yukaridaki temiz sonuclar onlar hakkinda bilgi vermez.")


def one_run(idx, total, timeout, approach=False):
    env = dict(os.environ, HEADLESS="1", DRONIONS_TARGET=TARGET,
               WORLD=WORLD, COMMAND_DELAY="45")
    if approach:
        env["DRONIONS_FAKE_VLM"] = "1"
        env.pop("DRONIONS_NO_VLM", None)
        # Start each run at a different waypoint. Without this the sweep is
        # deterministic and ten runs produced ten copies of one approach --
        # every single one from (4.9, -2.5) to (4.0, 1.3), with the wall never
        # between the drone and the target. The conditions the approach fixes
        # exist for simply never arose, so "10/10 clean" said nothing about
        # them. Cycling the start walks the target's first sighting around the
        # area instead.
        env["DRONIONS_SWEEP_START"] = str(idx - 1)
    else:
        env["DRONIONS_NO_VLM"] = "1"
        env.pop("DRONIONS_FAKE_VLM", None)
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
    if approach:
        row = parse_approach(runs[-1], idx)
        print(f"[{idx}/{total}] takibe gecis={row['handoffs']} varis={row['arrived']} "
              f"temas={row['contact']} engel={row['blocked']} "
              f"erken_varis_reddi={row['arrival_denied']}", flush=True)
        return [row]
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
    print(f"    {TARGET:16}: {top_box}/{len(tops)}  "
          f"({100*top_box/len(tops):.0f}%)")
    for name, k in Counter(r["nearest"] for r in tops
                           if not int(r["is_box"])).most_common():
        # An estimate can be nearest the target and still not be credited to
        # it, because it fell outside HIT_RADIUS. Printed under the target's own
        # name that reads as a second, contradictory score for the same thing.
        label = f"{name} (>{HIT_RADIUS:.2f}m)" if name == TARGET else name
        print(f"    {label:16}: {k}/{len(tops)}  ({100*k/len(tops):.0f}%)")

    seen = sum(1 for v in any_box.values() if v)
    print(f"\n  HEDEF ILK 3 ADAY ICINDE: {seen}/{len(any_box)} bakis "
          f"({100*seen/len(any_box):.0f}%)")

    firsts = []
    for run in runs:
        ts = [float(r["t"]) for r in rows if r["run"] == run and int(r["is_box"])]
        firsts.append(min(ts) if ts else None)
    got = [f for f in firsts if f is not None]
    line = f"  HEDEFI ILK GORME       : {len(got)}/{len(runs)} kosu"
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

    # Hue, so the colour gate's threshold can be set from rendered detections
    # rather than from the reference photo and the world's material
    # definitions -- the same way the size bound was derived from measured
    # widths instead of guessed at.
    hued = [r for r in rows if r.get("hue")]
    if hued:
        print("\n  --- ton dagilimi (renk kapisi icin) ---")
        by = {}
        for r in hued:
            by.setdefault(TARGET if int(r["is_box"]) else r["nearest"], []).append(
                float(r["hue"]))
        for name, vals in sorted(by.items(), key=lambda kv: -len(kv[1])):
            vals.sort()
            print(f"    {name:9} n={len(vals):3}  medyan {statistics.median(vals):5.0f} "
                  f"derece  [{vals[0]:.0f} - {vals[-1]:.0f}]")
        ref = by.get(TARGET)
        if ref:
            rm = statistics.median(ref)
            print(f"    referans olarak {TARGET} medyani {rm:.0f} derece alinirsa:")
            for name, vals in sorted(by.items()):
                if name == TARGET:
                    continue
                far = sum(1 for v in vals
                          if abs((v - rm + 180) % 360 - 180) > 45)
                print(f"      {name:9}: {far}/{len(vals)} tanesi 45 derece disinda "
                      f"-> elenirdi")

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
    ap.add_argument("--approach", action="store_true",
                    help="VLM yerine boyut filtresini kullan, yaklasma fazini olc")
    a = ap.parse_args()

    if a.report:
        if not OUT_CSV.exists():
            raise SystemExit(f"{OUT_CSV} yok.")
        with open(OUT_CSV, newline='', encoding='utf-8') as f:
            report(list(csv.DictReader(f)))
        return

    if a.approach:
        fields = (["run", "handoffs"] + list(APPROACH_EVENTS)
                  + ["locked", "tgt_x", "tgt_y", "drone_x", "drone_y", "obstacle_gap"])
    else:
        fields = ["run", "t", "rank", "conf", "area", "width", "hue",
                  "world_x", "world_y", "drone_x", "drone_y",
                  "nearest", "box_err", "is_box"]
    all_rows = []
    out = APPROACH_CSV if a.approach else OUT_CSV
    with open(out, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(1, a.n + 1):
            rows = one_run(i, a.n, a.timeout, approach=a.approach)
            all_rows += rows
            w.writerows(rows)
            f.flush()          # Ctrl-C keeps what has run so far
    print(f"\n-> {out}")
    (report_approach if a.approach else report)(all_rows)


if __name__ == "__main__":
    main()
