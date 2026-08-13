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
_X = re.compile(r'^\s*x:\s*(-?[\d.eE+-]+)')
_Y = re.compile(r'^\s*y:\s*(-?[\d.eE+-]+)')
_Z = re.compile(r'^\s*z:\s*(-?[\d.eE+-]+)')


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

    `latest()` returns (x, y, z) or None. None means the stream has not
    produced a matching pose yet -- callers should treat that as "no ground
    truth available" rather than substituting a guess, which is the mistake
    this module exists to stop.
    """

    def __init__(self, world: str = None, prefix: str = DEFAULT_PREFIX):
        self.prefix = prefix
        self.world = world or find_world()
        self._pos = None
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
        cur, pos = None, {}
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _NAME.search(line)
            if m:
                cur, pos = m.group(1), {}
                self._names.add(cur)
                continue
            if cur is None or not cur.startswith(self.prefix):
                continue
            for key, rx in (("x", _X), ("y", _Y), ("z", _Z)):
                mm = rx.match(line)
                if mm and key not in pos:
                    pos[key] = float(mm.group(1))
            # Position comes before orientation in a pose block, so three
            # components mean this one is complete.
            if len(pos) == 3:
                self._pos = (pos["x"], pos["y"], pos["z"])
                cur, pos = None, {}

    def latest(self) -> Optional[Tuple[float, float, float]]:
        return self._pos

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
