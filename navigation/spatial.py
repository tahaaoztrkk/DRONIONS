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
# Widest side the object can present, not its nominal width. A detector's box
# spans whichever dimension faces the camera, and a thing lying flat on a table
# is usually seen across its long axis -- so a phone entered at its 0.075 m
# width is 2.1x narrower than the 0.16 m the camera actually sees. Measured in
# flight: the model identified the phone correctly ("the bright blue screen and
# black bezel"), the gate computed 0.6 m against a 0.19 m ceiling, and threw the
# confirmation away. Entering the narrow side does not make the gate stricter,
# it makes it wrong.
#
# The four marked below are measured from the mesh vertices of the models
# actually in the tabletop world rather than recalled.
OBJECT_WIDTHS = {
    "box": 0.50, "backpack": 0.35, "keyboard": 0.35,
    "bottle": 0.08, "wallet": 0.10, "charger": 0.07, "mouse": 0.06,
    "keys": 0.08,
    "laptop": 0.375,        # measured, open
    # Absent before, which meant the gate had no opinion on books at all and
    # passed anything the detector called one.
    "book": 0.210,          # measured, lying flat
    "mug": 0.166,           # measured, including the handle
    "phone": 0.160,         # measured, long side
}

# Heights the target might be resting on. Floor first: when the size check
# cannot separate two planes, the floor is the safer answer, because it is
# where most things are and where an over-estimate is smallest.
# floor, table/counter, desk/worktop, seat/low shelf. The 1.0 m entry is the
# table in the tabletop world, measured rather than assumed -- without it the
# nearest plane on file was 0.75 and everything on that table was placed a
# quarter of a metre too near.
SUPPORT_HEIGHTS = (0.0, 0.75, 1.0, 0.45)

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
# Measured in the room, 38 detections of four objects at known positions
# (scripts/calibrate_support_plane.py). Median position error by margin:
#
#         laptop  book   mug   floor box
#   <= 6   0.31   0.21   0.17    0.67
#   >= 8   0.68   1.15   0.87    0.67
#
# There is no trade here, which is what the old value assumed there was. The
# floor object's error is 0.67 m at every margin -- lowering the threshold
# never lifts it onto a table that is not there -- while every object genuinely
# on a table gets three to five times closer. 12 was protecting nothing
# measurable and costing the room's targets their position: in flight the
# laptop was placed 0.8 m out and announced as "on the floor", in the same
# sentence as the model describing it on a wooden table.
#
# Anything from 1 to 6 scores identically and the cliff is between 6 and 8, so
# this sits in the middle of the flat region rather than at its edge.
FLOOR_SWITCH_MARGIN = 4.0

# How far the plane answer may disagree with the apparent-size answer before
# the plane is disbelieved altogether.
#
# Projecting onto a plane needs the drone to be meaningfully above it, and on
# an approach it stops being. Descending towards a mug on a 1.015 m table, at
# 1.1 m altitude the table plane can only be projected 0.43 m out -- so it
# drops away, the floor is all that is left, and the ray lands far past
# everything. Measured in flight: a mug 0.88 m away was placed at 3.2 m and
# announced as "on the floor", in the same breath as the model saying it was on
# a table.
#
# Apparent size gives an independent range that does not degrade this way. Over
# 38 detections the plane answer sits between 0.90 and 1.17 times it; the
# failure above is 3.6. Anything past this ratio is the geometry breaking down,
# and the size answer is used instead -- which also gives a height, without
# having to pick a plane at all.
RANGE_TRUST_RATIO = 2.0


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
    A negative class only fires if the detector names the thing, and it does
    not: YOLO-World labels a large flat surface "cardboard box" and a table
    "laptop". What separates them is not the label but the physics -- a
    detection covering a third of the frame at three metres is several metres
    across, and the user asked for a box.

    A caution about the numbers quoted below and elsewhere. Most were taken in
    the scenario world, where the confuser that dominates is the test wall --
    and that wall is a fixture for exercising obstacle avoidance, deliberately
    placed between the drone and the target. It is not what a room looks like.
    Real rooms have furniture and other objects on the same table, and their
    walls are boundaries rather than things standing in the way. So "rejects
    the wall" is evidence the gate works on one large confuser, not evidence it
    is calibrated for indoor use. That calibration has to be redone in a
    furnished scene, and until it is, the figures here should be read as a
    lower bound on the problem rather than a description of it.

    Generalises past this scenario, which is the point -- in any real room
    there will be something larger than the target in view, and rejecting it
    should not depend on the detector naming it correctly.

    Asking only about the floor is what this used to do, and it rejected a
    laptop sitting on a table three separate times in one flight, each time
    after the model had already confirmed it. The ray through the box's bottom
    edge passes over the table and travels on to the floor well beyond, so the
    range comes out too large and the width with it -- 0.8, 0.9 and 1.0 m for a
    0.375 m laptop. locate_target had already learned about raised surfaces;
    this had not. The question here is whether a thing that size could be the
    target *at all*, so it is enough to be plausible on any surface it could
    credibly be resting on.

    That relaxation could have gutted the gate, so it was measured rather than
    argued: 181 real detections across 25 viewpoints in the scenario world,
    each classified as the box or not by projecting the box's known position
    into the frame. The box keeps passing 100% of the time either way, and
    rejection of everything else moves from 30.2% to 32.1% passing -- two
    detections out of 106. The wall sits at a width ratio of 3.3 and is
    rejected on every surface; the 30% that pass are the blue distractor, which
    really is box-sized and which no size test can separate. That is what the
    colour gate and the model are for.

    What it does cost is measured too, and is worth stating rather than
    burying. A raised plane always shrinks the implied width -- that is the
    point of it -- so it always loosens the upper bound. Asked for a phone, the
    floor alone rejected 106 of 106 non-target detections and also the phone;
    with raised planes 30% of them get through, all of them the blue distractor
    seen obliquely, which from one camera genuinely could be a small object on
    a table. Monocular geometry cannot separate those two, and pretending
    otherwise would mean going back to rejecting the target as well. Identity
    is the colour gate's job and the model's.
    """
    expected = OBJECT_WIDTHS.get((target or "").lower().strip())
    if not expected:
        return True                 # nothing on file: no opinion
    measured = False
    for plane_z in SUPPORT_HEIGHTS:
        w = implied_width(candidate, drone_xyz, drone_quat, plane_z=plane_z)
        if w is None:
            continue
        measured = True
        if expected * SIZE_MIN_RATIO <= w <= expected * SIZE_MAX_RATIO:
            return True
    return not measured             # no range anywhere: no opinion


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
    chosen = best if best_gap * FLOOR_SWITCH_MARGIN < floor_gap else floor_hit
    return _sanity_against_size(chosen, drone_xyz, ray, expected)


def _sanity_against_size(hit, drone_xyz, ray, expected: Optional[float]):
    """Disbelieve a plane answer that the object's own apparent size contradicts.

    See RANGE_TRUST_RATIO. When they disagree badly the plane is the one that
    has failed -- it needs the drone to be well above it and on an approach that
    stops being true -- so the target is placed along the same ray at the range
    its size implies. The height falls out of that rather than being chosen.
    """
    if hit is None or expected is None:
        return hit
    rng = math.dist(hit, drone_xyz)
    if rng <= 0:
        return hit
    ratio = rng / expected
    if 1.0 / RANGE_TRUST_RATIO <= ratio <= RANGE_TRUST_RATIO:
        return hit
    return (drone_xyz[0] + ray[0] * expected,
            drone_xyz[1] + ray[1] * expected,
            drone_xyz[2] + ray[2] * expected)


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
