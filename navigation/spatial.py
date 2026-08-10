"""
Görevi: Mekânsal analiz (örn. 'charger is next to the laptop'). İleride depth estimation eklenebilir.
"""

from typing import List, Dict, Any
from perception.candidate import DetectionCandidate

def analyze_spatial_relations(objects: List[DetectionCandidate]) -> List[Dict[str, Any]]:
    """
    Given a list of objects, describes their spatial relations.
    """
    relations = []
    if len(objects) < 2:
        return relations
        
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            obj1 = objects[i]
            obj2 = objects[j]
            
            x1, y1 = obj1.center
            x2, y2 = obj2.center
            
            if x1 < x2:
                rel = f"{obj1.label} is to the left of {obj2.label}"
            else:
                rel = f"{obj1.label} is to the right of {obj2.label}"
                
            relations.append({
                "subject": obj1.label,
                "object": obj2.label,
                "relation": rel
            })
            
    return relations


# ---------------------------------------------------------------------------
# Geometric localization: where is the target, in the user's frame?
#
# The reference work (Wei et al., CHI '26) scored lowest on its spatial
# orientation task (3.2/5) and attributed that to LLMs being weak at spatial
# reasoning, "further compounded in our context, where the drone may be oriented
# differently from the user". That mismatch is geometry, not language. The drone
# knows its own pose, so the relation between target, drone and user can be
# computed exactly, leaving the language model only what it is actually good at:
# phrasing the result and saying what the thing looks like.
# ---------------------------------------------------------------------------

import math
from typing import Optional, Tuple

# Intrinsics read from the model, not assumed:
#   mono_cam/model.sdf     -> horizontal_fov 1.74 rad (99.7 deg), 1280x960
#   x500_dronions/model.sdf-> camera mounted pitched down by 0.35 rad
CAMERA_HFOV = 1.74
CAMERA_PITCH_DOWN = 0.35
CAMERA_ASPECT = 4.0 / 3.0

# No pitch calibration is applied, and that is a measured decision rather than
# an omission.
#
# Over 196 trials the estimate came out a consistent +0.30 m too far from the
# user, which looked like a camera-pitch offset. Steepening the ray by the
# amount that should have cancelled it made things worse (+0.45 m, and the
# median absolute error rose from 0.43 to 0.50 m), because the two are not the
# same axis: steepening pulls the estimate toward the *drone*, while the error
# is measured from the *user*. With the wall blocking 18% of viewpoints -- and
# blocking precisely those between user and target -- the surviving viewpoints
# sit mostly beyond the target, so pulling toward the drone pushes away from
# the user.
#
# The real effect underneath is that a ray through the bottom of the box meets
# the ground at the object's near edge, not its centre, so the estimate sits
# roughly half an object-depth short. Correcting that properly needs the
# object's physical size, which the system does not know at run time. Left
# uncorrected and documented instead of papered over.
CAMERA_PITCH_CALIB = 0.0

STEP_LENGTH = 0.75      # m, for expressing distance in paces


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def camera_ray_world(u_norm: float, v_norm: float, quat,
                     hfov: float = CAMERA_HFOV,
                     pitch_down: float = CAMERA_PITCH_DOWN + CAMERA_PITCH_CALIB,
                     aspect: float = CAMERA_ASPECT):
    """World-frame direction of the ray through a normalized pixel.

    u_norm/v_norm are 0..1 across the image with v downwards -- exactly the
    `normalized_center` a DetectionCandidate already carries. The full
    orientation quaternion is used rather than yaw alone, so the pitch the
    airframe takes while translating does not quietly bias the estimate.
    """
    f = 1.0 / math.tan(hfov / 2.0)
    a = (u_norm - 0.5) * 2.0                # right of centre, in half-widths
    b = (v_norm - 0.5) * 2.0 / aspect       # down from centre, same units

    ct, st = math.cos(pitch_down), math.sin(pitch_down)
    # Camera axes in the body frame (x forward, y left, z up).
    fwd = (ct, 0.0, -st)
    right = (0.0, -1.0, 0.0)
    down = (-st, 0.0, -ct)

    ray = [fwd[i] * f + right[i] * a + down[i] * b for i in range(3)]
    n = math.sqrt(sum(c * c for c in ray)) or 1.0
    return _quat_rotate(quat, [c / n for c in ray])


def project_to_plane(drone_xyz, ray_world, plane_z: float = 0.0
                     ) -> Optional[Tuple[float, float, float]]:
    """Where the ray meets a horizontal plane, or None if it never does.

    Assuming a plane is the honest limitation: one camera cannot recover range
    by itself. For a target resting on the floor (plane_z = 0) this is exact;
    for something on a table the caller must supply that height, otherwise the
    estimate lands beyond the object.
    """
    dz = ray_world[2]
    if dz >= -1e-6:                 # level or upward: never meets the plane
        return None
    t = (plane_z - drone_xyz[2]) / dz
    if t <= 0:
        return None
    return (drone_xyz[0] + t * ray_world[0],
            drone_xyz[1] + t * ray_world[1],
            plane_z)


def locate_target(candidate, drone_xyz, drone_quat, plane_z: float = 0.0,
                  use_bottom: bool = True):
    """DetectionCandidate + drone pose -> estimated world position.

    By default the ray is cast through the *bottom* edge of the box rather than
    its centre. An object standing on the floor touches the ground there, so
    that ray genuinely meets the ground plane at the object; a ray through the
    centre passes through a point part-way up the object and therefore lands
    beyond it. Measured on the scenario box, centre-ray estimates overshot by a
    consistent 0.5-0.8 m.
    """
    u, v = candidate.normalized_center
    if use_bottom and getattr(candidate, "image_height", 0):
        v = candidate.bbox[3] / candidate.image_height
    return project_to_plane(drone_xyz, camera_ray_world(u, v, drone_quat), plane_z)


def relative_to_user(target_xy, user_xy, user_yaw: float) -> Tuple[float, float]:
    """(distance, bearing) of a target from the user.

    Bearing is measured from the direction the user faces: 0 straight ahead,
    positive to their left, negative to their right.
    """
    dx = target_xy[0] - user_xy[0]
    dy = target_xy[1] - user_xy[1]
    bearing = math.atan2(dy, dx) - user_yaw
    return math.hypot(dx, dy), math.atan2(math.sin(bearing), math.cos(bearing))


def describe_direction(bearing: float) -> str:
    deg = math.degrees(bearing)
    a = abs(deg)
    if a <= 15:
        return "tam önünüzde"
    side = "solunuzda" if deg > 0 else "sağınızda"
    if a <= 50:
        return f"hafif {side}"
    if a <= 115:
        return side
    if a <= 150:
        return f"arkanıza doğru, {side}"
    return "tam arkanızda"


def describe_distance(distance: float) -> str:
    """Metres and paces. Orientation and mobility training uses steps, and
    'about three steps' is easier to act on than '2.1 metres'."""
    if distance < 0.6:
        return "hemen yanınızda"
    steps = max(1, round(distance / STEP_LENGTH))
    return f"yaklaşık {distance:.1f} metre ({steps} adım) ötede"


def height_class(z: float) -> str:
    if z < 0.25:
        return "yerde"
    if z < 1.1:
        return "masa yüksekliğinde"
    return "raf seviyesinde"


def describe_target(target_xyz, user_xy, user_yaw: float,
                    label: str = "Hedef", context: str = "") -> str:
    """The sentence the user actually hears.

    Geometry first and always; `context` carries what only the vision model can
    supply ("next to a blue mug") and is appended, never relied upon for where.
    """
    distance, bearing = relative_to_user(target_xyz[:2], user_xy, user_yaw)
    text = (f"{label} {describe_direction(bearing)}, "
            f"{describe_distance(distance)}, {height_class(target_xyz[2])}")
    if context:
        text += f". {context.strip().rstrip('.')}"
    return text + "."
