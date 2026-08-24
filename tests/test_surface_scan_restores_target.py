#!/usr/bin/env python3
"""
Görevi: Yüzey taramasının dedektörü hedefte bıraktığını doğrular.

The surface scan repurposes the one shared detector: it swaps the prompts to
furniture, reads a frame, and swaps them back. That makes it the only place in
the search loop that can leave global state changed behind it, and it did --
twice, in flights a day apart, for the same reason.

A mis-indentation put the search-area check outside the loop over candidates,
so its `continue` applied to the main flight loop and skipped the line that
restored the prompts. The detector then went on boxing tables, sofas and chairs
while the rest of the pipeline believed it was hunting a laptop. Measured at
the viewpoint both flights failed from: a chair at confidence 0.877 covering
10.4% of the frame, localised 0.47 m from a drone standing three metres from
its target. Everything downstream then worked correctly on that: the model was
shown a crop of the table and approved it, describing the laptop standing on
it, which was true, and a box that size implies a near object, so the drone
announced it had arrived without leaving its start position.

Nothing about that failure looked like state leaking out of a scan, which is
why it took two flights to find. This reads the block out of the node and runs
it on every path through -- surface inside the area, outside it, none found,
localisation failed, unknown label -- and checks the one property that has to
hold on all of them.

  tests/test_surface_scan_restores_target.py
"""
from __future__ import annotations

import os
import sys
import textwrap
import time

NODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'ros', 'dronions_ros_node_px4.py')
MARKER = 'if target and time.time() > surface_scan_after:'
TARGET = 'laptop'


def surface_scan_block(path: str = NODE) -> str:
    """The scan, lifted out of the node as it is actually written.

    Read from the file rather than restated here: a test that restates the code
    only checks the transcription. It is also what catches the original bug --
    that block does not compile on its own, because the `continue` was legal
    only by belonging to the flight loop it had been indented out into.
    """
    lines = open(path, encoding='utf-8').read().split('\n')
    start = next(i for i, l in enumerate(lines) if MARKER in l)
    end = next(i for i, l in enumerate(lines[start:], start) if l.strip() == '')
    return textwrap.dedent('\n'.join(lines[start:end]))


class _Detector:
    """Records which mode the prompts were left in."""

    def __init__(self):
        self.mode = '(hic ayarlanmadi)'

    def set_target_classes(self, classes):
        self.mode = 'YUZEY'

    def set_target(self, target):
        self.mode = target

    def detect(self, frame):
        return self.candidates


class _Candidate:
    def __init__(self, label):
        self.label = label


class _Node:
    def pose_xyz(self):
        return (0.0, 0.0, 2.0)

    def orientation(self):
        return (0.0, 0.0, 0.0, 1.0)


class _Surfaces:
    def note(self, *_):
        return False

    def summary(self):
        return ''


def run_case(block, candidates, where):
    det = _Detector()
    det.candidates = candidates
    env = dict(
        target=TARGET, time=time, surface_scan_after=0.0, frame=None,
        detector=det, filter_candidates=lambda c: c, node=_Node(),
        surfaces=_Surfaces(), log_event=lambda m: None,
        SURFACE_SCAN_INTERVAL=3.0,
        SURFACE_TOPS={'desk': 0.75, 'sofa': 0.45, 'chair': 0.45},
        SEARCH_AREA_X=(-1.0, 5.5), SEARCH_AREA_Y=(-3.0, 5.0),
        locate_target=lambda *a, **k: where,
    )
    exec(compile(block, '<yuzey taramasi>', 'exec'), env)
    return det.mode


# (name, candidates, what locate_target returns)
#
# The second is the path that cost the two flights: furniture seen high in the
# frame projects onto the floor a long way out -- the block's own comment says
# so -- which puts it outside the searched area.
CASES = [
    ('yuzey alan icinde',            [_Candidate('desk')],  (2.0, 0.0, 0.0)),
    ('yuzey alan disinda',           [_Candidate('sofa')],  (7.4, 0.0, 0.0)),
    ('hic yuzey adayi yok',          [],                    (0.0, 0.0, 0.0)),
    ('konumlandirma basarisiz',      [_Candidate('chair')], None),
    ('bilinmeyen mobilya etiketi',   [_Candidate('lamba')], (2.0, 0.0, 0.0)),
    ('iki aday, ikisi de disarida',  [_Candidate('sofa'),
                                      _Candidate('desk')],  (7.4, 0.0, 0.0)),
]


def main() -> int:
    block = surface_scan_block()
    failures = 0
    for name, candidates, where in CASES:
        try:
            mode = run_case(block, candidates, where)
            note = ''
        except Exception as exc:                          # noqa: BLE001
            mode, note = '(istisna)', f'  {type(exc).__name__}: {exc}'
        ok = mode == TARGET
        failures += not ok
        print(f"  {'OK  ' if ok else 'HATA'} {name:30} -> dedektor "
              f"'{mode}'{note}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} yol dedektoru hedefte birakti")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
