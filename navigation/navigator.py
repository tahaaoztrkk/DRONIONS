"""
Görevi: Drone için nereye gidileceğini (sağ, sol, derece vb.) hesaplar.
"""

from perception.candidate import DetectionCandidate
from typing import Dict, Any

# Target considered "reached" once its bounding box covers this fraction of
# the frame -- stops forward motion instead of driving into it forever.
# Measured empirically from a live approach: with this camera's mount height
# and FOV, the box's tracked relative_area plateaus around 0.07 and then the
# detection collapses (goes out of frame / loses context from being too
# close) well before it could ever reach a "large in frame" 0.3+ style
# threshold. So this has to be tuned to what's actually reachable, not to
# how full the frame "looks" arrived.
ARRIVAL_RELATIVE_AREA = 0.06

# Only start braking once the target's relative area passes this fraction of
# ARRIVAL_RELATIVE_AREA -- keeps full speed for the bulk of the approach
# instead of easing off from the very first (still-far-away) detection.
BRAKING_ZONE_START = 0.3

# Below this normalized x-error the target counts as centered (no steering).
CENTER_DEADBAND = 0.1


def get_navigation_decision(target: DetectionCandidate) -> Dict[str, Any]:
    """
    Computes movement commands based on bounding box position/size relative
    to the image. Returns a dict with a human-readable `action` (used for
    the on-screen overlay) plus continuous `turn` (-1..1, + = right) and
    `throttle` (0..1) values a Twist/velocity publisher can scale directly.
    """
    if target is None:
        return {"action": "SEARCHING", "angle": 0, "speed": 0, "turn": 0.0, "throttle": 0.0}

    if target.relative_area >= ARRIVAL_RELATIVE_AREA:
        return {"action": "ARRIVED", "angle": 0, "speed": 0, "turn": 0.0, "throttle": 0.0}

    cx, cy = target.normalized_center

    # normalized x is 0.0 (left) to 1.0 (right). Center is 0.5.
    error_x = cx - 0.5

    # Max FOV roughly 60 degrees -- kept for the overlay/debug text.
    angle = int(error_x * 60)

    if abs(error_x) < CENTER_DEADBAND:
        action = "FORWARD"
    elif error_x > 0:
        action = "TURN_RIGHT"
    else:
        action = "TURN_LEFT"

    # Proportional steering: full deflection (+-1) once the target is at the
    # frame edge (|error_x| approaches 0.5), scales down as it centers.
    turn = max(-1.0, min(1.0, error_x / 0.5))

    # Ease off forward speed while turning sharply. This needs to be strong:
    # combining a big turn with real forward motion sweeps the target across
    # the frame fast enough (rotation + translation both changing its screen
    # position) that ByteTrack's frame-to-frame IOU match breaks and the
    # track is lost. Close to a stationary pivot turn at max deflection,
    # full speed once nearly centered.
    turn_penalty = 1.0 - min(1.0, abs(turn)) * 0.85

    # Full speed until the braking zone, then ramp down to the floor right
    # as we approach ARRIVAL_RELATIVE_AREA (avoids ramming the target).
    brake_start = ARRIVAL_RELATIVE_AREA * BRAKING_ZONE_START
    if target.relative_area <= brake_start:
        proximity_factor = 1.0
    else:
        remaining = ARRIVAL_RELATIVE_AREA - brake_start
        proximity_factor = max(0.3, 1.0 - (target.relative_area - brake_start) / remaining)

    throttle = turn_penalty * proximity_factor

    return {
        "action": action,
        "angle": angle,
        "speed": int(throttle * 10),
        "turn": turn,
        "throttle": throttle,
    }
