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
import os
import re
from typing import Optional, Tuple

# Intrinsics read from the model, not assumed:
#   dronions_cam/model.sdf  -> horizontal_fov, 1280x960
#   x500_dronions/model.sdf -> camera mounted pitched down by 0.35 rad
#
# The field of view is read out of the model file rather than written here as a
# constant, because it is a variable in this project: it sets how small an
# object the drone can find, and the lens gets changed to measure that. Two
# copies of the number is one copy too many -- if they drift, nothing raises,
# every projection is quietly skewed, and the error looks like a localization
# bug. Env override for callers with no repo checkout; 1.74 is the stock lens.
CAMERA_MODEL_SDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'px4', 'models', 'dronions_cam', 'model.sdf')


def _camera_hfov(default: float = 1.74) -> float:
    env = os.getenv('DRONIONS_CAMERA_HFOV')
    if env:
        return float(env)
    try:
        with open(CAMERA_MODEL_SDF) as fh:
            m = re.search(r'<horizontal_fov>\s*([0-9.]+)\s*</horizontal_fov>',
                          fh.read())
        if m:
            return float(m.group(1))
    except OSError:
        pass
    return default


CAMERA_HFOV = _camera_hfov()
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


# Horizontal range is allowed to reach this multiple of the drone's height
# above the plane, and no further. Rejecting only upward rays is not enough:
# a ray descending a tenth of a degree still meets the plane, hundreds of
# metres away, and range error grows as h/sin^2(theta) -- so near the horizon a
# one-degree pointing error becomes tens of metres. At this ratio a 1 deg error
# costs well under a metre, which is the accuracy the rest of the pipeline was
# measured at. A run once projected a target to (18.9, -12.5) this way.
MAX_RANGE_HEIGHT_RATIO = 5.0


def project_to_plane(drone_xyz, ray_world, plane_z: float = 0.0
                     ) -> Optional[Tuple[float, float, float]]:
    """Where the ray meets a horizontal plane, or None if it never does.

    Assuming a plane is the honest limitation: one camera cannot recover range
    by itself. For a target resting on the floor (plane_z = 0) this is exact;
    for something on a table the caller must supply that height, otherwise the
    estimate lands beyond the object.

    Returns None as well when the intersection is too far out to be worth
    believing -- see MAX_RANGE_HEIGHT_RATIO. Refusing to answer is the right
    failure here: the caller logs "position could not be computed" and keeps
    searching, where a number would be spoken to someone who cannot check it.
    """
    dz = ray_world[2]
    if dz >= -1e-6:                 # level or upward: never meets the plane
        return None
    height = drone_xyz[2] - plane_z
    if height <= 0:                 # at or below the plane: nothing to see
        return None
    t = (plane_z - drone_xyz[2]) / dz
    if t <= 0:
        return None
    hit = (drone_xyz[0] + t * ray_world[0],
           drone_xyz[1] + t * ray_world[1],
           plane_z)
    reach = math.hypot(hit[0] - drone_xyz[0], hit[1] - drone_xyz[1])
    if reach > MAX_RANGE_HEIGHT_RATIO * height:
        return None
    return hit


# Typical real-world width of a target, in metres. Only used to sanity-check
# which surface an object is standing on -- never as the position itself, since
# a bounding box is far too noisy to range from directly. Values are the
# widest face the object usually presents; the scenario box is 0.63 x 0.40 m,
# so 0.5 is its mean aspect.
OBJECT_WIDTHS = {
    "box": 0.50, "backpack": 0.35, "laptop": 0.33, "keyboard": 0.35,
    "bottle": 0.08, "mug": 0.10, "phone": 0.075, "wallet": 0.10,
    "keys": 0.06, "charger": 0.07, "mouse": 0.06,
}

# Heights the target might be resting on. Floor first: when the size check
# cannot separate two planes, the floor is the safer answer, because it is
# where most things are and where an over-estimate is smallest.
SUPPORT_HEIGHTS = (0.0, 0.75, 0.45)     # floor, table/counter, seat/low shelf

# A raised surface must fit the apparent size this many times better than the
# floor before it is believed.
#
# Chosen from the gap between the two cases rather than by taste. Objects
# genuinely on a 0.75 m surface produce evidence ratios of 17-56 (mug 41,
# bottle 17, laptop 56, phone 44, keys 37), while the scenario box -- which is
# on the floor and which the pipeline already localizes to 0.01 m -- never
# exceeds 9.2 even with its detector box 3x too wide. Anywhere in that gap
# separates them; the middle of it leaves room for both to be noisier than
# measured.
FLOOR_SWITCH_MARGIN = 12.0


def range_from_apparent_size(candidate, target: str,
                             hfov: float = CAMERA_HFOV) -> Optional[float]:
    """Distance implied by how wide the object looks, or None.

    Deliberately crude. Its job is not to measure range -- a detector box is
    much too unstable for that -- but to tell two candidate support surfaces
    apart, where the answers differ by metres rather than centimetres.
    """
    width_m = OBJECT_WIDTHS.get((target or "").lower().strip())
    img_w = getattr(candidate, "image_width", 0)
    if not width_m or not img_w:
        return None
    px = candidate.bbox[2] - candidate.bbox[0]
    if px <= 1:
        return None
    focal_px = (img_w / 2.0) / math.tan(hfov / 2.0)
    return width_m * focal_px / px


# How far the physical size implied by a detection may stray from the target's
# expected width before it is thrown out.
#
# Measured over 106 real detections from five survey flights. The scenario box
# implies 0.53-1.12 m across (it is 0.63 m), while the wall implies 1.26-4.90 m
# (it is 4.0 m) -- a clean gap. An upper bound of 2.5x the expected width sits
# inside it: every one of the 69 wall detections is rejected and 81% of box
# detections survive.
SIZE_MIN_RATIO = 0.30
SIZE_MAX_RATIO = 2.50


def implied_width(candidate, drone_xyz, drone_quat,
                  plane_z: float = 0.0) -> Optional[float]:
    """How wide the detection would really be, given where its base sits.

    The range comes from the same ground-plane projection used to locate the
    target, so this costs nothing extra and is available before any model is
    called.
    """
    hit = project_to_plane(
        drone_xyz,
        camera_ray_world(candidate.normalized_center[0],
                         candidate.bbox[3] / candidate.image_height
                         if getattr(candidate, "image_height", 0)
                         else candidate.normalized_center[1],
                         drone_quat),
        plane_z)
    if hit is None or not getattr(candidate, "image_width", 0):
        return None
    rng = math.dist(hit, drone_xyz)
    frac = (candidate.bbox[2] - candidate.bbox[0]) / candidate.image_width
    return frac * 2.0 * rng * math.tan(CAMERA_HFOV / 2.0)


def size_plausible(candidate, drone_xyz, drone_quat, target: str) -> bool:
    """Could a thing that size be the target at all?

    This is the filter the negative prompts were supposed to be and are not.
    YOLO-World labels the scenario wall "cardboard box", so a negative class of
    "wall" never fires; measured over 80 viewpoints, the wall was the
    top-ranked candidate in 71% of them. What separates them is not the label
    but the physics: a detection covering a third of the frame at three metres
    is several metres across, and the user asked for a box.

    Generalises past this scenario, which is the point -- in any real room
    there will be something larger than the target in view, and rejecting it
    should not depend on the detector naming it correctly.
    """
    expected = OBJECT_WIDTHS.get((target or "").lower().strip())
    if not expected:
        return True                 # nothing on file: no opinion
    w = implied_width(candidate, drone_xyz, drone_quat)
    if w is None:
        return True                 # no range: no opinion
    return expected * SIZE_MIN_RATIO <= w <= expected * SIZE_MAX_RATIO


def locate_target(candidate, drone_xyz, drone_quat, plane_z: Optional[float] = None,
                  use_bottom: bool = True, target: str = ""):
    """DetectionCandidate + drone pose -> estimated world position.

    The ray is cast through the *bottom* edge of the box rather than its
    centre. An object standing on a surface touches it there, so that ray
    genuinely meets the surface at the object; a ray through the centre passes
    part-way up the object and lands beyond it. Measured on the scenario box,
    centre-ray estimates overshot by a consistent 0.5-0.8 m.

    Which surface, though, was previously assumed to be the floor. Indoors that
    is wrong for most things worth finding -- a mug on a table projected onto
    the floor lands well past the table, and the error grows with how oblique
    the view is. Passing plane_z keeps the old fixed behaviour; leaving it None
    picks among SUPPORT_HEIGHTS by asking which one puts the object at the
    distance its apparent size implies. With no size on file the floor is used,
    which is exactly what the code did before.
    """
    u, v = candidate.normalized_center
    if use_bottom and getattr(candidate, "image_height", 0):
        v = candidate.bbox[3] / candidate.image_height
    ray = camera_ray_world(u, v, drone_quat)

    if plane_z is not None:
        return project_to_plane(drone_xyz, ray, plane_z)

    # Measured tolerance of this choice: the right surface is still picked with
    # the box up to 3x too wide, but only ~10% too narrow before it falls back
    # towards the floor. The asymmetry is the safe way round -- a narrow box
    # makes the object look far away, which agrees with the floor, so a bad
    # size estimate degrades to the behaviour this replaced rather than
    # inventing a new answer. Worth knowing that detector boxes shrink under
    # occlusion, so that is the direction real errors will take.
    expected = range_from_apparent_size(candidate, target)
    floor_hit = project_to_plane(drone_xyz, ray, SUPPORT_HEIGHTS[0])
    if expected is None:
        return floor_hit

    floor_gap = (abs(math.dist(floor_hit, drone_xyz) - expected)
                 if floor_hit else None)
    best, best_gap = None, None
    for z in SUPPORT_HEIGHTS[1:]:
        hit = project_to_plane(drone_xyz, ray, z)
        if hit is None:
            continue
        gap = abs(math.dist(hit, drone_xyz) - expected)
        if best_gap is None or gap < best_gap:
            best, best_gap = hit, gap

    if best is None:
        return floor_hit
    if floor_gap is None:
        return best
    # Leaving the floor takes clear evidence, not a marginal win. The floor is
    # right for most things and is where the pipeline's accuracy was measured;
    # a slightly over-wide detector box otherwise lifts an object that is
    # genuinely on the ground onto an imaginary table and shortens the distance
    # the user is told. Being wrong in the familiar direction beats being wrong
    # in a new one.
    return best if best_gap * FLOOR_SWITCH_MARGIN < floor_gap else floor_hit


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
