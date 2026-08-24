"""
Görevi: Gazebo'nun gerçek model pozunu canlı olarak okur.

Why this exists as its own module: two measurement scripts need the same thing
and both got it wrong in their own way. The frame-mismatch sweep used the pose
it *commanded* a teleport to, which is not where the drone is -- an unarmed
airframe starts falling the instant it is placed, so every projection was
computed from an altitude the vehicle no longer had. The ground-truth logger
looked for the model under the wrong name and recorded nothing at all.

Streaming, not polling. `gz model -p` costs about five seconds a call and in an
earlier campaign starved frame capture badly enough that 65% of samples arrived
with no image; subscribing to the pose topic costs nothing and is always
current.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Optional, Tuple

# PX4 spawns the model with an instance suffix -- "x500_dronions_0" -- so this
# is matched as a prefix. An exact-match filter silently recorded nothing
# across four flights.
DEFAULT_PREFIX = "x500_dronions"

_NAME = re.compile(r'name:\s*"([^"]+)"')
_FIELD = re.compile(r'^\s*([xyzw]):\s*(-?[\d.eE+-]+)')
_BLOCK = re.compile(r'^\s*(position|orientation)\s*\{')
# Gazebo prints protobuf text format, which omits any field at its default, so
# an identity rotation arrives as a bare "w: 1" and a drone at the origin as an
# empty "position {}". Missing means zero, never "not received".


def find_world() -> Optional[str]:
    """World name from gz's own topic list rather than hardcoded."""
    try:
        out = subprocess.run(["gz", "topic", "-l"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        m = re.match(r'/world/([^/]+)/pose/info', line.strip())
        if m:
            return m.group(1)
    return None


class PoseReader:
    """Latest true pose of the model, kept current by a background reader.

    `latest()` returns (x, y, z) and `latest_quat()` (x, y, z, w), or None.
    None means the stream has not produced a matching pose yet -- callers should treat that as "no ground
    truth available" rather than substituting a guess, which is the mistake
    this module exists to stop.
    """

    def __init__(self, world: str = None, prefix: str = DEFAULT_PREFIX):
        self.prefix = prefix
        self.world = world or find_world()
        self._pos = None
        self._quat = None
        self._names = set()
        self._stop = threading.Event()
        self._proc = None
        self._thread = None

    def start(self) -> bool:
        if not self.world:
            return False
        topic = f"/world/{self.world}/pose/info"
        self._proc = subprocess.Popen(["gz", "topic", "-e", "-t", topic],
                                      stdout=subprocess.PIPE, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return True

    def _read(self):
        cur, block = None, None
        pos, orient = {}, {}
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _NAME.search(line)
            if m:
                cur, block = m.group(1), None
                pos, orient = {}, {}
                self._names.add(cur)
                continue
            if cur is None or not cur.startswith(self.prefix):
                continue
            b = _BLOCK.match(line)
            if b:
                block = b.group(1)
                continue
            f = _FIELD.match(line)
            if f and block:
                (pos if block == 'position' else orient)[f.group(1)] = \
                    float(f.group(2))
                continue
            # A closing brace at the outer indent ends this model's block.
            if line.startswith('}'):
                if pos or orient:
                    self._pos = (pos.get('x', 0.0), pos.get('y', 0.0),
                                 pos.get('z', 0.0))
                    q = (orient.get('x', 0.0), orient.get('y', 0.0),
                         orient.get('z', 0.0), orient.get('w', 0.0))
                    self._quat = q if any(q) else (0.0, 0.0, 0.0, 1.0)
                cur, block = None, None

    def latest(self) -> Optional[Tuple[float, float, float]]:
        return self._pos

    def latest_quat(self) -> Optional[Tuple[float, float, float, float]]:
        """(x, y, z, w) of the model's true orientation, or None.

        This was not read at all until now, and every offline measurement that
        needed a camera ray synthesised a yaw-only quaternion instead -- which
        assumes an airframe that is exactly level. A drone that has just been
        teleported is not: measured against the true attitude, the assumed ray
        sat about 4 degrees steeper than the direction to the object, and the
        plane projection it fed came out 20-35% short.
        """
        return self._quat

    def wait_for_pose(self, timeout: float = 10.0) -> Optional[Tuple[float, float, float]]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._pos is not None:
                return self._pos
            time.sleep(0.1)
        return None

    def seen_names(self):
        """What the stream actually contained, for when nothing matched."""
        return sorted(self._names)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
