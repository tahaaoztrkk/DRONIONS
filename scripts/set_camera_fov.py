#!/usr/bin/env python3
"""
Görevi: Kameranın görüş açısını değiştirir ve etkilerini gösterir.

The field of view decides how small an object the drone can find, so it is a
variable here rather than a fixed property. Recall was measured against apparent
object width (docs/PLAN_EVERYDAY_SCENARIOS.md 5f): about 120 px gives 8/9,
76 px gives half that, and below ~35 px nothing is found at all. Widening the
lens buys sweep coverage and spends reach; narrowing it does the reverse.

At the stock 1.74 rad (100 deg) a 0.12 m cup only reaches 76 px from 0.85 m --
closer than the airframe is wide, and closer than the flight envelope allows.
That is the whole reason for this script.

  scripts/set_camera_fov.py                # sadece mevcut durumu goster
  scripts/set_camera_fov.py 50             # 50 dereceye ayarla
  scripts/set_camera_fov.py 1.74 --radians

Changing it means rebuilding PX4 and restarting the simulation -- the model is
read at spawn. The script says so at the end rather than leaving it implied.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF = os.path.join(REPO, 'px4', 'models', 'dronions_cam', 'model.sdf')
PATTERN = re.compile(r'(<horizontal_fov>\s*)([0-9.]+)(\s*</horizontal_fov>)')

# Object widths worth checking a lens against, and the measured pixel
# thresholds they have to clear.
OBJECTS = [('senaryo kutusu', 0.63), ('sirt cantasi', 0.35), ('laptop', 0.33),
           ('kupa', 0.12), ('telefon', 0.075), ('anahtar', 0.06)]
THRESHOLDS = [(120, '%89'), (76, '%56'), (35, 'esik')]
IMAGE_WIDTH = 1280.0

# Below this the airframe cannot go: the x500 is 0.68 m across and the flight
# envelope keeps 0.60 m of clearance, which two crashes starting from 1.05 m
# say is not conservative. A detection distance under this is unreachable.
SAFE_STANDOFF = 1.6


def read_fov() -> float:
    m = PATTERN.search(open(SDF).read())
    if not m:
        sys.exit(f"horizontal_fov bulunamadi: {SDF}")
    return float(m.group(2))


def write_fov(rad: float) -> None:
    text = open(SDF).read()
    open(SDF, 'w').write(PATTERN.sub(lambda m: f"{m.group(1)}{rad:.4f}{m.group(3)}", text))


def report(rad: float) -> None:
    focal = (IMAGE_WIDTH / 2.0) / math.tan(rad / 2.0)
    print(f"\nHFOV {rad:.4f} rad ({math.degrees(rad):.0f} derece), "
          f"odak {focal:.0f} px\n")
    head = ''.join(f"{lbl:>10}" for _, lbl in THRESHOLDS)
    print(f"{'nesne':<16}{'genislik':>9}{head}   erisilebilir?")
    print('-' * 62)
    for name, width in OBJECTS:
        cells = ''
        reach = None
        for px, _ in THRESHOLDS:
            d = width * focal / px
            cells += f"{d:9.2f}m"
            if px == 76:
                reach = d
        ok = 'evet' if reach >= SAFE_STANDOFF else 'HAYIR (cok yakin)'
        print(f"{name:<16}{width:8.3f}m{cells}   {ok}")
    print('-' * 62)
    print(f"'erisilebilir' = %56 tespit mesafesi guvenli yaklasma "
          f"({SAFE_STANDOFF} m) disinda mi")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fov', nargs='?', type=float, help='yeni gorus acisi')
    ap.add_argument('--radians', action='store_true',
                    help='deger radyan (varsayilan: derece)')
    ap.add_argument('--no-install', action='store_true',
                    help='PX4 tarafina kopyalama')
    a = ap.parse_args()

    current = read_fov()
    if a.fov is None:
        report(current)
        return

    new = a.fov if a.radians else math.radians(a.fov)
    if not 0.1 < new < 3.0:
        sys.exit(f"{new:.3f} rad makul degil -- dereceyi radyanla karistirdiniz mi?")

    write_fov(new)
    print(f"{SDF}\n  {current:.4f} -> {new:.4f} rad "
          f"({math.degrees(current):.0f} -> {math.degrees(new):.0f} derece)")
    report(new)

    if not a.no_install:
        subprocess.run([os.path.join(REPO, 'scripts', 'install_px4_assets.sh')],
                       check=False)

    print("\nModel spawn aninda okunuyor -- simulasyonu yeniden baslatmadan "
          "bu degisiklik gecerli olmaz.")
    print("navigation/spatial.py bu dosyadan okuyor, ayrica ayar gerekmez.")


if __name__ == '__main__':
    main()
