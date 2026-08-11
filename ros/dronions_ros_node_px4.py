#!/usr/bin/env python3
"""
ROS 2 wrapper for the DRONIONS hybrid VLM+YOLO pipeline -- PX4 SITL
quadrotor variant. Same Gemini/YOLO/navigation/UI pipeline as
ros/dronions_ros_node.py (the ground-rig node); only the actuation layer
changes, per the isolation that node's navdecision_to_twist() was
deliberately designed for. New here: a real arm/OFFBOARD/takeoff state
machine, altitude hold, and a synthetic GCS heartbeat PX4 requires to pass
its "GCS connected" arming check.

Run (four cooperating processes, each in its own terminal):
    Terminal 1 (PX4 SITL + Gazebo -- its own px4venv, NOT this venv):
        cd ~/PX4-Autopilot
        source px4venv/bin/activate
        make px4_sitl gz_x500_dronions_dronions_scenario
    Terminal 2 (MAVROS, connects to PX4's "Onboard" mavlink instance):
        source /opt/ros/jazzy/setup.bash
        ros2 launch mavros px4.launch fcu_url:="udp://:0@127.0.0.1:14580"
    Terminal 3 (camera + lidar bridge -- PX4 spawns its own Gazebo, so this
    bridges sensors only; nothing here actuates the vehicle):
        cd ~/DRONIONS
        source /opt/ros/jazzy/setup.bash
        ros2 launch ros/launch/dronions_px4_bridge.launch.py
    Terminal 4 (this node):
        cd ~/DRONIONS
        source /opt/ros/jazzy/setup.bash
        source venv/bin/activate
        python3 ros/dronions_ros_node_px4.py

On this machine (hybrid AMD iGPU + NVIDIA dGPU) Gazebo does not pick the
NVIDIA card on its own: Mesa opens the NVIDIA DRM node, finds no driver it
can use ("libEGL warning: ... driver (null)" -> "failed to create dri2
screen") and falls back. The fallback renderer is slow enough that, with the
GUI up, the sim stalls PX4 through lockstep and MAVROS drops the link. Use
scripts/run_px4_sim.sh for terminal 1, or export these first -- measured
5 EGL warnings and no GPU process before, 0 warnings and gz on the RTX after:
    __NV_PRIME_RENDER_OFFLOAD=1
    __GLX_VENDOR_LIBRARY_NAME=nvidia
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

Why a fake GCS heartbeat: PX4's rcAndDataLinkCheck arming gate requires a
GCS-recognized link, and specifically checks PX4's "Normal" mavlink
instance (port 18570) -- not the "Onboard" instance (14580) MAVROS itself
connects to for control. A real companion-computer setup has a separate GCS
on the Normal link; SITL has none, so this node sends bare heartbeats on
18570 itself. Confirmed empirically: without this, arming is silently
rejected (result=1) with no console message explaining why.
"""
import os
import sys
import threading
import queue
import time
import math
import random

import numpy as np
import cv2

# main.py / utils/logger.py use CWD-relative paths (memory/, logs/) -- make
# sure they resolve the same way regardless of where this script is launched
# from.
DRONIONS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DRONIONS_ROOT)
os.chdir(DRONIONS_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, NavSatFix
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.msg import State, ManualControl, StatusText
from mavros_msgs.srv import CommandBool, SetMode, ParamSetV2
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue
from rcl_interfaces.srv import SetParameters

from pymavlink import mavutil

from config import VOICE_ENABLED, USER_POSITION, USER_YAW

# Survey mode: skip the VLM, log what the detector ranked at every check, and
# keep sweeping instead of handing off. Set by
# scripts/experiment_repeatability.py so the search can be measured without
# spending API quota, which otherwise caps honest repetition at about one run
# a day. Never a flight mode -- without the model there is nothing telling
# *your* box from any box, which is the whole assistive premise.
SURVEY = os.getenv("DRONIONS_NO_VLM", "") not in ("", "0")

# Stand-in for the VLM: accept the best size-plausible candidate and carry on
# through centring and the approach. Identification is meaningless here -- it
# will happily accept a distractor -- but everything *after* confirmation is
# exercised, for free, as many times as needed.
#
# This only became possible once the size gate could reject the wall. The
# earlier attempt at a VLM-free end-to-end run accepted whatever ranked first,
# which was the wall from the takeoff spot, and never got further.
#
# It exists because the failures in the approach are intermittent: flying into
# the wall, announcing arrival early and losing an edge-of-frame confirmation
# all happen on some runs and not others, so a single flight cannot show
# whether a fix worked or the dice fell differently.
FAKE_VLM = os.getenv("DRONIONS_FAKE_VLM", "") not in ("", "0")

# Waypoint the sweep starts from. Only the campaign sets it; a normal flight
# always starts at the first, so behaviour stays reproducible.
try:
    SWEEP_START = int(os.getenv("DRONIONS_SWEEP_START", "0"))
except ValueError:
    SWEEP_START = 0
from assistant.command_parser import parse_command
from assistant.agent import capture_and_analyze, select_candidate, answer_followup
from assistant.dialogue import Dialogue
from assistant.speech import speak
from assistant.listen import get_voice_input
from utils.logger import log_event
from utils.prompts import get_prompts

from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from perception.tracker import Tracker
from navigation.navigator import get_navigation_decision
from navigation.spatial import (locate_target, describe_target,
                                describe_direction, relative_to_user,
                                size_plausible, implied_width)
from ui.overlay import draw_overlay

PHASE_SEARCH = "VLM SEARCHING"
PHASE_CENTER = "CENTERING"
PHASE_TRACK = "YOLO TRACKING"

# Horizontal pursuit speeds -- identical meaning to the ground rig
# (navigator.py's turn/throttle output is unchanged). Interpreted in
# BODY_NED frame (set on /mavros/setpoint_velocity at startup) so "forward"
# means the drone's own nose direction, matching what the navigator assumes
# -- MAVROS defaults this topic to LOCAL_NED (world-frame), which would
# silently break navigation without the frame override.
MAX_LINEAR = 0.3
MAX_ANGULAR = 0.6
EXPLORE_LINEAR = 0.25
OBSTACLE_SAFE_DISTANCE = 1.0

# Half the width the airframe actually sweeps, from the model rather than
# assumed: x500_base puts the rotors at +/-0.174 m and the props are 13 inch
# (1345_prop_*.stl), so 0.174 + 0.165 = 0.339 m -- a vehicle 0.68 m across.
AIRFRAME_HALF_WIDTH = 0.339
# Clearance either side before a return counts as being in the way. Small on
# purpose: household doorways are 0.8-0.9 m, so there is very little to spend.
CORRIDOR_CLEARANCE = 0.06
CORRIDOR_HALF_WIDTH = AIRFRAME_HALF_WIDTH + CORRIDOR_CLEARANCE
AVOID_ANGULAR = 0.35
# How long a turn is committed to after touching an obstacle. Randomized so
# repeated contacts with the same wall don't produce the same escape path.
AVOID_TURN_SECONDS = (2.0, 5.0)
# Lawnmower search area, in the local ENU frame with the launch point at the
# origin. Sized to cover the object row (x=3.5, y=0.4..3.6) and the ground
# either side of the wall (x=1.75, y=-0.5..3.5) with margin, without sending
# the drone off into empty world.
SEARCH_AREA_X = (-1.0, 5.5)
SEARCH_AREA_Y = (-3.0, 5.0)
SEARCH_ROW_SPACING = 2.0    # m between sweep rows
WAYPOINT_RADIUS = 0.8       # m; close enough to count as reached
# A waypoint sitting behind the wall can never be reached. Abandon it rather
# than let one blocked point stall the whole sweep.
WAYPOINT_TIMEOUT = 25.0     # s
# Consecutive obstacle contacts while pursuing one waypoint before giving up
# on it. 2 = one retry from a different random turn direction.
BLOCKED_HITS_BEFORE_SKIP = 2

# Multiplier on the sweep's own traversal estimate before the search is
# abandoned. The estimate assumes clear air and perfect tracking; avoidance
# turns cost AVOID_TURN_SECONDS each, a climb over the wall restarts a
# waypoint, and every VLM check holds station for the length of an API call.
SEARCH_TIMEOUT_MARGIN = 1.6

# How often the sweep records where it actually is. There was no position
# trace at all, so a nineteen-metre excursion left nothing in the log to
# diagnose it from -- only the moment tracking noticed, long afterwards.
POSE_LOG_INTERVAL = 10.0        # s
SIZE_LOG_INTERVAL = 20.0        # s

# Lidar range at which the approach is judged to have come dangerously close.
# The airframe sweeps 0.339 m either side, so a return at half a metre is
# already within a body length of contact.
CONTACT_RANGE = 0.5
CONTACT_LOG_INTERVAL = 5.0      # s
STRAY_REPORT_INTERVAL = 5.0     # s
# Before abandoning a blocked waypoint, try going *over* the obstacle. This is
# the quadrotor's one real advantage over the ground rig and nothing in the
# search used it: the sweep flew at a fixed altitude, so a wall taller than
# cruise height was not merely hard to pass, it was unrepresented as an option.
# Climbing also clears the obstacle by itself -- the lidar is horizontal and
# rides with the airframe, so once above the wall it stops seeing it.
SEARCH_CLIMB_STEP = 1.5     # m gained per blocked attempt
MAX_SEARCH_ALTITUDE = 5.5   # m ceiling for the climb-and-look-over

# Altitude hold / takeoff. HOVER_ALTITUDE is tunable -- revisit alongside
# obstacle_wall's height (Milestone B) since a hovering quad can simply fly
# over a wall sized for the ground rig's ~0.7m camera.
HOVER_ALTITUDE = 2.0        # m
ALT_HOLD_KP = 0.8
ALT_HOLD_VZ_MAX = 0.6

# While tracking, the hold altitude stops being a fixed number and follows the
# target's vertical position in frame. The camera is fixed and looks straight
# ahead, so a ground-level object sinks out of the bottom of the frame as the
# drone closes in -- which is exactly where tracking was being lost and the
# search restarted. Descending to bring it back to the middle of the frame
# keeps it visible all the way to arrival.
MIN_TRACK_ALTITUDE = 0.7    # m -- floor, don't descend into the scenery
# Absolute ceiling for any commanded hold altitude. Has to cover the search
# climb, not just tracking: clamping at the old 4.0 m would have silently
# capped the climb-over-obstacle at less than it asks for.
MAX_TRACK_ALTITUDE = MAX_SEARCH_ALTITUDE
TRACK_VERTICAL_DEADBAND = 0.12   # normalized; ignore small framing errors
TRACK_ALT_RATE = 0.6        # m/s of hold-altitude adjustment at full error
# Consecutive frames the navigator must keep saying ARRIVED before it is
# believed. A single frame is not evidence: one detection of the wall filling
# the frame was enough to declare arrival 3.2 m from the real target.
ARRIVAL_CONFIRM_FRAMES = 5

# Furthest the geometric estimate may put the target while still believing an
# ARRIVED from frame coverage. Generous: the estimate carries ~0.5 m of error
# and the approach stops short anyway, so this only has to catch the case where
# the two disagree by metres -- which is what a wall filling the frame looks
# like.
ARRIVAL_MAX_DISTANCE = 2.5
# How far (normalized image units) a YOLO detection may sit from the centre
# Gemini reported and still count as the same object. Beyond this the two
# perception layers simply disagree about what is in the frame.
GEMINI_POINT_MAX_DIST = 0.25
# How much bigger or smaller than the crop Gemini approved a detection may be
# and still count as the same object at hand-off. Generous, because the drone
# closes distance during centring and apparent area grows as 1/d^2 -- but the
# case this exists to reject was off by a factor of 165, so generous is enough.
GEMINI_AREA_RATIO_MAX = 6.0
# The sweep usually catches the target at the edge of the frame -- the one
# confirmed in flight sat at (0.11, 0.93), a corner. Handing that straight to
# tracking meant driving forward while barely holding it in view, and it was
# lost within seconds every time. Turn to face it first, moving nothing else,
# and only start the approach once it is comfortably inside the frame.
CENTER_OK = 0.20            # normalized offset from frame centre, both axes
CENTERING_MAX_SECONDS = 8.0 # give up and go back to searching

# Yaw rate used to bring a target back into view when centring loses it.
# Slower than the avoidance turn: the object is near the frame edge and a brisk
# turn sweeps it straight out the other side.
SEARCH_RECOVER_ANGULAR = 0.15
# Tracking had no spatial bound at all, only the search did. A lock onto
# something far away never grows enough in frame to satisfy the arrival test,
# so the drone drove forward indefinitely -- measured 131 m from the scenario,
# still going. Give up on a pursuit that leaves the search area by this margin.
TRACK_LEASH_MARGIN = 4.0    # m beyond SEARCH_AREA before a pursuit is dropped
# Empirically, ~1 m/s was not reliably enough to leave the ground in this
# sim (stayed under 0.2m for 4+ seconds); 3 m/s clearly was. Use the
# confirmed-working value for the initial climb, then ease onto the P
# controller once close, to avoid overshooting past HOVER_ALTITUDE.
TAKEOFF_CLIMB_VZ = 3.0
# Ramp up to TAKEOFF_CLIMB_VZ over this long instead of stepping straight to
# it -- an instant full-speed step was observed tripping a SITL battery
# failsafe (voltage sag under the sudden current draw reads as critical for
# a moment, even though true charge is unaffected) right at liftoff,
# triggering an automatic RTL/land within seconds of arming.
TAKEOFF_RAMP_SECONDS = 2.0
ALT_TOLERANCE = 0.3         # m

GCS_HEARTBEAT_PORT = 18570   # PX4's "Normal" mavlink instance (see docstring)


def return_to_area_twist(x: float, y: float, yaw: float) -> Twist:
    """Fly straight back to the middle of the search area.

    Deliberately not a waypoint: whatever let the drone leave in the first
    place is still in play, so this aims at the one point furthest from every
    edge and translates only once pointed at it, exactly like the sweep does.
    """
    cx = 0.5 * (SEARCH_AREA_X[0] + SEARCH_AREA_X[1])
    cy = 0.5 * (SEARCH_AREA_Y[0] + SEARCH_AREA_Y[1])
    err = math.atan2(math.sin(math.atan2(cy - y, cx - x) - yaw),
                     math.cos(math.atan2(cy - y, cx - x) - yaw))
    t = Twist()
    t.angular.z = max(-AVOID_ANGULAR, min(AVOID_ANGULAR, 1.5 * err))
    t.linear.x = EXPLORE_LINEAR if abs(err) < 0.6 else 0.0
    return t


def vlm_failure_message(msg: str) -> str:
    """What to tell the user when the vision model stops answering.

    The raw text is an English API payload and used to be spoken verbatim, so
    a 404 body was read aloud as the drone's reply. It carries a real
    distinction worth keeping, though: whether waiting would help.
    """
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return ("Görüş servisinin günlük kullanım hakkı doldu, şu anda "
                "göremiyorum. Aramayı durduruyorum, yarın tekrar deneyebiliriz.")
    if "404" in msg or "NOT_FOUND" in msg:
        return ("Görüş modeline erişemiyorum, ayarlarda güncellenmesi gerekiyor. "
                "Aramayı durduruyorum.")
    return ("Görüş servisine şu anda ulaşamıyorum, bu yüzden göremiyorum. "
            "Aramayı durduruyorum, birazdan tekrar deneyebilirsiniz.")


def ref_path_for(target):
    """Reference photo for a target from the Visual Memory Bank, or None.

    Factored out of the search loop because follow-up questions need the same
    picture: "is there any writing on it?" is about the user's own object, so
    the model should still have the reference in hand.
    """
    if not target:
        return None
    clean = target.lower().strip()
    for name in (clean, clean.replace(" ", "_")):
        for ext in (".jpg", ".png", ".jpeg"):
            path = os.path.join("memory", name + ext)
            if os.path.exists(path):
                return path
    return None


def compute_altitude_vz(current_z, target_z=HOVER_ALTITUDE):
    """P-controller for vertical velocity. Used both to hold hover altitude
    during search/track and for the final approach of the takeoff climb."""
    error = target_z - current_z
    vz = ALT_HOLD_KP * error
    return max(-ALT_HOLD_VZ_MAX, min(ALT_HOLD_VZ_MAX, vz))


def navdecision_to_twist(nav_decision, current_z):
    """Maps navigation.navigator's action dict + current altitude to a
    Twist. Horizontal terms are the exact ground-rig mapping (kept isolated
    there specifically for this swap); vertical is the altitude-hold P
    term, computed independently of horizontal.
    """
    twist = Twist()
    twist.linear.z = compute_altitude_vz(current_z)
    if not nav_decision or nav_decision.get("action") in ("SEARCHING", "ARRIVED"):
        return twist
    turn = nav_decision.get("turn", 0.0)
    throttle = nav_decision.get("throttle", 0.0)
    twist.linear.x = MAX_LINEAR * throttle
    twist.angular.z = -MAX_ANGULAR * turn
    return twist


def hold_altitude_twist(current_z: float) -> Twist:
    """Zero horizontal motion, altitude-hold on Z. Used whenever the loop
    has nothing else to command (no frame yet, no target set) -- a bare
    Twist() here would zero out linear.z too, silently dropping altitude
    control (confirmed causing an uncommanded climb/drift in testing)."""
    twist = Twist()
    twist.linear.z = compute_altitude_vz(current_z)
    return twist


class SearchPattern:
    """Systematic lawnmower sweep of the search area.

    Replaces a purely reactive wander, which had no notion of where the
    scenario was or which parts had already been looked at. Tuning its
    curvature could not fix that: tight, it re-flew one closed 4 m lap in
    front of the wall forever; loose, it drifted 30 m out and never came
    back. Coverage has to be planned, not emergent.

    Rows run along x and step in y, so the forward-facing camera sweeps
    along each row rather than across it. Obstacle contact still interrupts
    with a committed turn (the wall sits inside the pattern and has to be
    driven around), and a waypoint that cannot be reached in
    WAYPOINT_TIMEOUT is abandoned -- otherwise a blocked waypoint behind the
    wall would stall the whole sweep.
    """

    def __init__(self, start_index: int = 0):
        self._waypoints = self._build_lawnmower()
        # Where in the loop the sweep begins. Deterministic by default; the
        # campaign varies it because leaving it fixed made ten runs into ten
        # copies of one. Every approach came from (4.9, -2.5) to (4.0, 1.3),
        # so conditions that depend on geometry -- the wall lying between the
        # drone and the target, above all -- never arose and the fixes for
        # them were never actually exercised.
        self._idx = start_index % len(self._waypoints) if self._waypoints else 0
        self._deadline = time.time() + WAYPOINT_TIMEOUT
        self._turn_until = 0.0
        self._turn_dir = 1.0
        self._blocked_hits = 0
        self._altitude = HOVER_ALTITUDE
        self._climb_note = None

    def search_altitude(self) -> float:
        """Altitude the sweep currently wants. Rises when the way is blocked
        and drops back to cruise once past, since detail matters more than
        reach when nothing is in the way."""
        return self._altitude

    def take_climb_note(self):
        """One-shot message for the caller to log, or None."""
        note, self._climb_note = self._climb_note, None
        return note

    @staticmethod
    def _build_lawnmower():
        x0, x1 = SEARCH_AREA_X
        y0, y1 = SEARCH_AREA_Y
        pts = []
        y = y0
        left_to_right = True
        while y <= y1 + 1e-6:
            pts.append((x0 if left_to_right else x1, y))
            pts.append((x1 if left_to_right else x0, y))
            y += SEARCH_ROW_SPACING
            left_to_right = not left_to_right
        return pts

    def current_waypoint(self):
        return self._waypoints[self._idx]

    def sweep_seconds(self) -> float:
        """Lower bound on one full pass: every leg flown, every corner turned,
        nothing in the way.

        Exists because the give-up timeout has to be derived from the pattern
        rather than picked. A fixed 180 s was tried and is shorter than this
        floor (198 s for the current area), so the search was cut off before it
        could finish even one clean sweep and reported "not found" for ground
        it had never flown over.
        """
        pts = self._waypoints
        travel = sum(math.hypot(pts[i + 1][0] - pts[i][0],
                                pts[i + 1][1] - pts[i][1])
                     for i in range(len(pts) - 1))
        turned = 0.0
        for i in range(1, len(pts) - 1):
            a = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
            b = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
            turned += abs(math.atan2(math.sin(b - a), math.cos(b - a)))
        return travel / EXPLORE_LINEAR + turned / AVOID_ANGULAR

    def _advance(self):
        self._idx = (self._idx + 1) % len(self._waypoints)
        self._deadline = time.time() + WAYPOINT_TIMEOUT
        self._blocked_hits = 0
        # Back down to cruise for the next leg: height buys reach but costs
        # apparent object size, and the target is small to begin with.
        self._altitude = HOVER_ALTITUDE

    def twist(self, obstacle_ahead: bool, current_z: float,
              x: float, y: float, yaw: float) -> Twist:
        t = Twist()
        t.linear.z = compute_altitude_vz(current_z)
        now = time.time()

        if obstacle_ahead and now >= self._turn_until:
            self._turn_dir = random.choice((-1.0, 1.0))
            self._turn_until = now + random.uniform(*AVOID_TURN_SECONDS)
            self._blocked_hits += 1
            # Hitting the same obstacle repeatedly while chasing one waypoint
            # means that waypoint is behind it. Drop it immediately instead of
            # shuttling into the wall until WAYPOINT_TIMEOUT expires -- that
            # cost two ~50 s stalls in a single measured sweep.
            if self._blocked_hits >= BLOCKED_HITS_BEFORE_SKIP:
                if self._altitude + SEARCH_CLIMB_STEP <= MAX_SEARCH_ALTITUDE:
                    # Try over it before giving up on the waypoint.
                    self._altitude += SEARCH_CLIMB_STEP
                    self._blocked_hits = 0
                    self._deadline = now + WAYPOINT_TIMEOUT
                    self._climb_note = (
                        f"Engel asilamiyor -- {self._altitude:.1f}m'ye tirmanip "
                        f"ustunden bakiliyor.")
                else:
                    self._advance()

        if now < self._turn_until:
            # Yaw in place, no forward component, so a committed avoidance
            # turn can never drive further into whatever triggered it.
            t.angular.z = AVOID_ANGULAR * self._turn_dir
            return t

        wx, wy = self.current_waypoint()
        if math.hypot(wx - x, wy - y) < WAYPOINT_RADIUS or now > self._deadline:
            self._advance()
            wx, wy = self.current_waypoint()

        bearing = math.atan2(wy - y, wx - x)
        err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
        t.angular.z = max(-AVOID_ANGULAR, min(AVOID_ANGULAR, 1.5 * err))
        # Translate only once roughly pointed at the waypoint, so the drone
        # tracks the row instead of arcing across it.
        t.linear.x = EXPLORE_LINEAR if abs(err) < 0.6 else 0.0
        return t


def get_console_input(q: queue.Queue):
    while True:
        try:
            cmd = input("\n[DRONIONS] Komut, onay için Enter ('q' ile kapatın): ")
        except EOFError:
            # Input piped from a script rather than typed. Stop reading instead
            # of raising in a thread nobody is watching -- the flight carries on
            # under whatever commands did arrive.
            print("\n[DRONIONS] Girdi akışı bitti, konsol komutları kapandı.")
            return
        q.put(cmd)
        if cmd.lower() in ['q', 'çıkış', 'quit', 'exit']:
            break


def gcs_heartbeat_loop(stop_event: threading.Event):
    """Bare MAVLink heartbeats to PX4's Normal mavlink instance so its
    rcAndDataLinkCheck arming gate sees a connected GCS. See module
    docstring for why this is necessary."""
    conn = mavutil.mavlink_connection(
        f'udpout:127.0.0.1:{GCS_HEARTBEAT_PORT}',
        source_system=255,
        source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
    )
    while not stop_event.is_set():
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(0.5)


class DronionsRosNodePX4(Node):
    """PX4/MAVROS actuated variant of DronionsRosNode. cap_read()/
    obstacle_ahead() mirror the ground-rig node's interface; publish_twist()/
    current_altitude() are the PX4-specific additions.
    """

    def __init__(self):
        super().__init__('dronions_ros_node_px4')
        self._frame = None
        self._lock = threading.Lock()
        self._min_range = float('inf')
        self._z = 0.0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._quat = (0.0, 0.0, 0.0, 1.0)
        self.state = State()
        # EKF readiness. state.connected only means "MAVLink is talking", not
        # "the estimator has converged". Arming before the EKF has a local
        # pose and a global origin is what produces the
        # "PositionTargetGlobal failed because no origin" warning, and leaves
        # the position estimate free to jump once the origin finally latches.
        self._pose_count = 0
        self._has_global_origin = False

        # PX4 drops OFFBOARD if the setpoint stream stalls for more than a
        # few hundred ms. The main loop cannot guarantee that: a Gemini call,
        # a YOLO inference, a speak() or a slow cv2.imshow all block it for
        # seconds at a time. So the stream lives in its own thread that
        # republishes the last commanded twist at a fixed rate, independent
        # of how slow perception is. The main loop only ever *sets* a twist.
        self._desired_twist = Twist()
        self._target_z = HOVER_ALTITUDE
        self._sp_lock = threading.Lock()
        self._sp_stop = threading.Event()
        self._sp_thread = None

        self.create_subscription(Image, '/camera/image_raw', self._on_image, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(State, '/mavros/state', self._on_state, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                  self._on_pose, qos_profile_sensor_data)
        # PX4's own console messages (the "Failsafe activated ...", "Landing
        # detected", low-battery warnings etc.) arrive here as MAVLink
        # STATUSTEXT. Logging them into our own log means a failure is
        # self-diagnosing: the exact PX4 reason lands in the same timeline as
        # our own actions, instead of having to correlate two terminals by eye.
        # sensor_data QoS (BEST_EFFORT), not the default depth-10 RELIABLE:
        # MAVROS publishes statustext best-effort, and a RELIABLE subscription
        # is silently incompatible -- it connects but never delivers a single
        # message ("offering incompatible QoS ... RELIABILITY").
        self.create_subscription(StatusText, '/mavros/statustext/recv',
                                  self._on_statustext, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, '/mavros/global_position/global',
                                  self._on_global, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.mc_pub = self.create_publisher(ManualControl, '/mavros/manual_control/send', 10)

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.param_client = self.create_client(ParamSetV2, '/mavros/param/set')

    def _on_image(self, msg: Image):
        if msg.encoding not in ('bgr8', 'rgb8'):
            self.get_logger().warn(f"Desteklenmeyen goruntu encoding'i: {msg.encoding}")
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frame = arr.copy()

    def cap_read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def _on_scan(self, msg: LaserScan):
        # Keep each return's lateral offset, not just the nearest range. A
        # single min() cannot tell a wall from a doorway: both put something
        # inside OBSTACLE_SAFE_DISTANCE, and only the *shape* of the scan says
        # whether there is a way through.
        self._min_range = float('inf')
        self._blocked = False
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            self._min_range = min(self._min_range, r)
            if r >= OBSTACLE_SAFE_DISTANCE:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            # How far to the side of the flight path this return sits: a door
            # frame well off the centreline is not in the way, a wall straight
            # ahead is.
            #
            # Currently this changes nothing, and that was worth measuring
            # rather than assuming. The lidar fan is only +/-0.35 rad, which at
            # 0.9 m spans +/-0.33 m -- narrower than the corridor itself -- so
            # every return that is close enough to matter is already inside it.
            # The test is kept because it is the correct rule and becomes load
            # bearing the moment the fan is widened, which the doorway work
            # needs anyway: at present the frame edges are invisible until the
            # drone is too close to avoid them.
            if abs(r * math.sin(angle)) < CORRIDOR_HALF_WIDTH:
                self._blocked = True

    def obstacle_ahead(self) -> bool:
        return self._blocked

    def min_range(self) -> float:
        return self._min_range

    def _on_state(self, msg: State):
        self.state = msg

    def _on_pose(self, msg: PoseStamped):
        self._z = msg.pose.position.z
        self._x = msg.pose.position.x
        self._y = msg.pose.position.y
        q = msg.pose.orientation
        # Full orientation is kept, not just yaw: projecting a detection to a
        # world position uses the whole rotation, and the airframe pitches
        # noticeably while translating.
        self._quat = (q.x, q.y, q.z, q.w)
        self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose_count += 1

    def horizontal_pose(self):
        """(x, y, yaw) in the local ENU frame, for the search leash."""
        return self._x, self._y, self._yaw

    def pose_xyz(self):
        return (self._x, self._y, self._z)

    def orientation(self):
        return self._quat

    def _on_global(self, msg: NavSatFix):
        # MAVROS only publishes this once PX4 has a global position estimate,
        # which is exactly the origin the setpoint plugins complain about.
        if math.isfinite(msg.latitude) and math.isfinite(msg.longitude):
            self._has_global_origin = True

    def ekf_ready(self) -> bool:
        return self._pose_count >= 20 and self._has_global_origin

    def current_altitude(self) -> float:
        return self._z

    def _on_statustext(self, msg: StatusText):
        sev = {0: 'EMERGENCY', 1: 'ALERT', 2: 'CRITICAL', 3: 'ERROR',
               4: 'WARNING', 5: 'NOTICE', 6: 'INFO', 7: 'DEBUG'}.get(msg.severity, '?')
        line = f"PX4[{sev}] {msg.text}"
        log_event(line)
        if msg.severity <= 4:  # WARNING and worse: surface on the console too
            print(f"\n[PX4] {sev}: {msg.text}")

    def publish_twist(self, twist: Twist):
        self.cmd_pub.publish(twist)
        self.mc_pub.publish(ManualControl())  # keep the manual-control signal fresh too

    def set_target_altitude(self, z: float):
        """Move the altitude the setpoint thread holds. Clamped so visual
        servoing can never walk the drone into the ground or the ceiling."""
        self._target_z = max(MIN_TRACK_ALTITUDE, min(MAX_TRACK_ALTITUDE, z))

    def target_altitude(self) -> float:
        return self._target_z

    def set_desired_twist(self, twist: Twist):
        """Hand a twist to the setpoint thread. Returns immediately -- the
        actual publishing happens at a fixed rate in _setpoint_loop, so a
        caller that then blocks for seconds cannot stall the stream."""
        with self._sp_lock:
            self._desired_twist = twist

    def _setpoint_loop(self, rate_hz: float = 30.0):
        period = 1.0 / rate_hz
        while not self._sp_stop.is_set():
            with self._sp_lock:
                src = self._desired_twist
            twist = Twist()
            twist.linear.x = src.linear.x
            twist.linear.y = src.linear.y
            twist.angular.z = src.angular.z
            # Vertical is recomputed here rather than reused from the stored
            # twist: while the main loop is blocked in Gemini/YOLO its stored
            # vz would be frozen at a stale value, and a frozen non-zero vz is
            # exactly what caused the uncommanded climb seen earlier. Reading
            # the live altitude each tick keeps the hold honest.
            twist.linear.z = compute_altitude_vz(self._z, self._target_z)
            self.publish_twist(twist)
            time.sleep(period)

    def start_setpoint_stream(self):
        self._sp_thread = threading.Thread(target=self._setpoint_loop, daemon=True)
        self._sp_thread.start()

    def stop_setpoint_stream(self):
        self._sp_stop.set()
        if self._sp_thread:
            self._sp_thread.join(timeout=2.0)

    def set_px4_param(self, param_id, *, integer_value=None, double_value=None):
        req = ParamSetV2.Request()
        req.param_id = param_id
        pv = ParameterValue()
        if integer_value is not None:
            pv.type = 2  # PARAMETER_INTEGER
            pv.integer_value = integer_value
        else:
            pv.type = 3  # PARAMETER_DOUBLE
            pv.double_value = double_value
        req.value = pv
        future = self.param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        ok = bool(future.result() and future.result().success)
        if not ok:
            self.get_logger().warn(f"PX4 parametresi ayarlanamadi: {param_id}")
        return ok


def wait_for(node, predicate, timeout_sec, tick_fn=None):
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.05)
        if tick_fn:
            tick_fn()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def startup_sequence(node: DronionsRosNodePX4):
    """Connect -> configure params -> pre-stream -> OFFBOARD -> arm -> climb
    to hover altitude. Runs once before the SEARCH/TRACK loop. Raises
    RuntimeError if any step doesn't complete in time."""

    log_event("PX4: FCU baglantisi bekleniyor...")
    if not wait_for(node, lambda: node.state.connected, 15):
        raise RuntimeError("MAVROS PX4'e baglanamadi (connected=False)")
    log_event("PX4: baglandi.")

    # mav_frame=BODY_NED so linear.x/angular.z mean "forward/yaw relative to
    # the drone's own heading" -- MAVROS defaults this to LOCAL_NED (world
    # frame), which would make the navigator's steering meaningless.
    # The result is checked, and the flight refused if it did not stick. This
    # was set and then ignored, which is the same silent-failure shape as the
    # safety-parameter race that caused takeoff-then-land: nothing in the log
    # distinguished "frame set" from "frame quietly rejected", and under
    # LOCAL_NED every steering command means a compass direction instead of
    # "forward". Refusing to fly is the safe end of that.
    frame_client = node.create_client(SetParameters, '/mavros/setpoint_velocity/set_parameters')
    if not frame_client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError(
            "MAVROS setpoint_velocity parametre servisi yok -- mav_frame "
            "ayarlanamaz, BODY_NED olmadan yon komutlari anlamsiz.")
    req = SetParameters.Request()
    p = ParameterMsg()
    p.name = 'mav_frame'
    p.value.type = 4  # PARAMETER_STRING
    p.value.string_value = 'BODY_NED'
    req.parameters = [p]
    future = frame_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    res = future.result()
    ok = bool(res and res.results and res.results[0].successful)
    if not ok:
        reason = (res.results[0].reason if res and res.results else "cevap yok")
        raise RuntimeError(
            f"mav_frame=BODY_NED ayarlanamadi ({reason}). LOCAL_NED altinda "
            f"'ileri' komutu kuzeye gitmek demek -- ucus reddediliyor.")
    log_event("PX4: setpoint cercevesi BODY_NED olarak ayarlandi.")

    # Wait for the estimator, not just the link. Starting the node seconds
    # after PX4 (rather than a couple of minutes later) otherwise arms on an
    # unconverged EKF: the setpoint plugins report "no origin", and the
    # position estimate can jump the moment the origin latches -- right
    # during the climb, which is where the takeoff-then-land failsafe was
    # being observed.
    log_event("PX4: EKF/konum kestirimi bekleniyor...")
    if not wait_for(node, node.ekf_ready, 60):
        raise RuntimeError(
            "EKF hazir olmadi (yerel poz / global origin gelmedi) -- "
            "PX4'e biraz daha zaman tanimak gerekiyor"
        )
    log_event("PX4: EKF hazir (yerel poz + global origin mevcut).")

    log_event("PX4: guvenli ucus parametreleri ayarlaniyor...")
    safety_params = [
        # No real GCS on the Normal link in SITL -> don't RTL/failsafe on data-link loss.
        ('NAV_DLL_ACT', dict(integer_value=0)),
        # Our "GCS" is a plain python heartbeat thread, so it is only as
        # punctual as the GIL lets it be -- a CPU-heavy frame can delay it.
        # The 10 s default turns such a hiccup straight into
        # "Connection to ground station lost" -> RTL; 120 s keeps the link
        # judged on whether the thread is actually alive, not on scheduling.
        ('COM_DL_LOSS_T', dict(integer_value=120)),
        # Tolerate gaps in our python-side setpoint stream a bit more generously.
        ('COM_OF_LOSS_T', dict(double_value=30.0)),
        # 4 = "Ignores all sources", NOT 1 = "MAVLink only". Under 1, PX4 still
        # expects valid MANUAL_CONTROL messages over MAVLink and raises
        # manual_control_signal_lost when they don't arrive -- captured live
        # via `px4-listener failsafe_flags`, that flag flips False->True at the
        # exact instant of takeoff and is what fires the "entering Hold for 5
        # seconds" failsafe (offboard/geofence/battery/position all stayed
        # clean, and the setpoint stream held a steady 45 Hz throughout). This
        # vehicle is fully autonomous with no RC and no joystick, so the
        # correct answer is for PX4 not to expect manual control at all.
        ('COM_RC_IN_MODE', dict(integer_value=4)),
        # SITL's simulated battery can report a transient critical/emergency
        # reading under a sudden current draw even though true charge is
        # unaffected (confirmed: battery read 100% again right after the
        # RTL/land this triggered). 0 = warn only, never auto RTL/land.
        ('COM_LOW_BAT_ACT', dict(integer_value=0)),
        # Same idea for the separate "estimated remaining flight time low"
        # check, which reacts to current draw rate rather than charge.
        ('COM_FLTT_LOW_ACT', dict(integer_value=0)),
        # Geofence violation action defaults to Hold mode -- matches the
        # exact "Failsafe activated: entering Hold for 5 seconds" seen
        # right after every takeoff. Likely spurious: the "no origin"
        # warning from mavros.guided_target suggests the GPS/EKF origin
        # isn't reliably settled yet at takeoff, which would make a
        # distance-from-home geofence check read garbage. 0 = None, fence
        # checks don't trigger any action.
        ('GF_ACTION', dict(integer_value=0)),
    ]
    # MAVROS refuses param/set until it has downloaded PX4's full parameter
    # list, and that sync takes ~15-30s after the link comes up. Called any
    # earlier, every set fails *instantly* (all six inside 5 ms) and, before
    # this retry loop existed, the node just logged AYARLANAMADI and took off
    # anyway -- with GF_ACTION still at its default of 2 (Hold mode), which is
    # precisely the "Failsafe activated: entering Hold for 5 seconds" ->
    # RTL -> Land seen right after every takeoff. It only ever reproduced when
    # the three terminals were started back-to-back; a couple of minutes of
    # incidental delay was enough to hide it completely.
    PARAM_SET_TIMEOUT = 120.0
    deadline = time.time() + PARAM_SET_TIMEOUT
    pending = list(safety_params)
    warned = False
    while pending and time.time() < deadline:
        still_pending = []
        for param_id, kwargs in pending:
            if node.set_px4_param(param_id, **kwargs):
                log_event(f"PX4: {param_id} ayarlandi")
            else:
                still_pending.append((param_id, kwargs))
        pending = still_pending
        if pending:
            if not warned:
                log_event("PX4: parametre listesi henuz senkronize degil, "
                          "MAVROS'un indirmesi bekleniyor...")
                warned = True
            wait_for(node, lambda: False, 3.0)

    if pending:
        names = ", ".join(p for p, _ in pending)
        raise RuntimeError(
            f"Guvenlik parametreleri ayarlanamadi ({names}). Bunlar olmadan "
            f"kalkis yapilmiyor: GF_ACTION varsayilani Hold modudur ve "
            f"kalkistan hemen sonra RTL/inise yol acar."
        )

    log_event("PX4: setpoint akisi baslatiliyor...")
    t0 = time.time()
    while time.time() - t0 < 3:
        node.publish_twist(Twist())
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)

    node.arm_client.wait_for_service(timeout_sec=10)
    node.mode_client.wait_for_service(timeout_sec=10)

    log_event("PX4: OFFBOARD moduna geciliyor...")
    req = SetMode.Request()
    req.custom_mode = "OFFBOARD"
    future = node.mode_client.call_async(req)
    ok = wait_for(node, lambda: future.done(), 10, tick_fn=lambda: node.publish_twist(Twist()))
    if not ok or not future.result().mode_sent:
        raise RuntimeError("OFFBOARD moduna gecilemedi")
    if not wait_for(node, lambda: node.state.mode == "OFFBOARD", 5,
                     tick_fn=lambda: node.publish_twist(Twist())):
        raise RuntimeError("OFFBOARD modu onaylanmadi")

    log_event("PX4: arm ediliyor...")
    # Retried: the GCS heartbeat thread occasionally hasn't sent enough
    # heartbeats yet for PX4's GCS-link arming check to be satisfied at the
    # exact moment of the first attempt (observed intermittently in
    # testing) -- a short retry loop is more robust than tuning the exact
    # warm-up delay.
    armed_ok = False
    for attempt in range(4):
        req = CommandBool.Request()
        req.value = True
        future = node.arm_client.call_async(req)
        ok = wait_for(node, lambda: future.done(), 10, tick_fn=lambda: node.publish_twist(Twist()))
        if ok and future.result().success:
            armed_ok = True
            break
        log_event(f"PX4: arm denemesi {attempt + 1} basarisiz, tekrar deneniyor...")
        wait_for(node, lambda: False, 2, tick_fn=lambda: node.publish_twist(Twist()))
    if not armed_ok:
        raise RuntimeError("Arm edilemedi (saglik kontrolleri / GCS heartbeat calisiyor mu kontrol et)")

    log_event(f"PX4: kalkis - {HOVER_ALTITUDE}m hedefleniyor...")
    speak("Kalkış yapılıyor.")

    climb_start = time.time()

    def climb_tick():
        twist = Twist()
        if node.current_altitude() < HOVER_ALTITUDE - ALT_TOLERANCE * 2:
            # Ramp 0 -> TAKEOFF_CLIMB_VZ instead of stepping straight to it
            # (see TAKEOFF_RAMP_SECONDS).
            ramp = min(1.0, (time.time() - climb_start) / TAKEOFF_RAMP_SECONDS)
            twist.linear.z = TAKEOFF_CLIMB_VZ * ramp
        else:
            twist.linear.z = compute_altitude_vz(node.current_altitude())  # smooth final approach
        node.publish_twist(twist)

    if not wait_for(node, lambda: abs(node.current_altitude() - HOVER_ALTITUDE) < ALT_TOLERANCE,
                     30, tick_fn=climb_tick):
        raise RuntimeError(f"Hedef irtifaya ulasilamadi (mevcut: {node.current_altitude():.1f}m)")

    # A SITL failsafe (e.g. a transient sensor/battery reading) can RTL and
    # auto-disarm the vehicle during the climb without our loop noticing --
    # confirmed happening intermittently in testing. Catch it here rather
    # than silently starting SEARCH/TRACK on the ground, disarmed.
    if not node.state.armed:
        raise RuntimeError("Kalkis sirasinda disarm oldu (muhtemelen bir PX4 failsafe/RTL tetiklendi)")

    log_event(f"PX4: hedef irtifada ({node.current_altitude():.1f}m). Arama basliyor.")


def main():
    rclpy.init()
    node = DronionsRosNodePX4()

    stop_heartbeat = threading.Event()
    threading.Thread(target=gcs_heartbeat_loop, args=(stop_heartbeat,), daemon=True).start()
    time.sleep(1.0)  # let the heartbeat thread's first few beats go out before anything needs them

    log_event("Initializing DRONIONS ROS node (PX4 SITL, Hybrid VLM + YOLO)...")
    speak("System activated. I am listening for your commands.")

    # The whole sequence gets retried a few times: SITL has shown occasional
    # transient failures (arm rejected because the GCS heartbeat hadn't
    # warmed up yet, or a failsafe RTL-ing and disarming mid-climb) that a
    # fresh attempt from a clean disarmed state reliably clears.
    MAX_STARTUP_ATTEMPTS = 3
    started = False
    for attempt in range(1, MAX_STARTUP_ATTEMPTS + 1):
        try:
            startup_sequence(node)
            started = True
            break
        except Exception as e:
            log_event(f"PX4 baslangic dizisi basarisiz (deneme {attempt}/{MAX_STARTUP_ATTEMPTS}): {e}")
            print(f"\n[HATA] PX4 baslangic dizisi basarisiz (deneme {attempt}/{MAX_STARTUP_ATTEMPTS}): {e}")
            if attempt < MAX_STARTUP_ATTEMPTS:
                time.sleep(3.0)

    if not started:
        speak("Kalkış başarısız oldu.")
        stop_heartbeat.set()
        node.destroy_node()
        rclpy.shutdown()
        return

    # From here on the main thread is no longer trusted to keep either the
    # setpoint stream or the subscriptions alive:
    #  - rclpy.spin() moves to its own thread so pose/state/image/scan
    #    callbacks keep flowing while the main loop blocks in Gemini or YOLO
    #    (a stale self._z would otherwise make altitude hold act on old data),
    #  - the setpoint thread keeps OFFBOARD fed at a fixed rate.
    # Both start *before* YOLOWorldDetector() rather than after: loading the
    # model takes seconds, and until now that happened after takeoff with
    # nothing publishing, leaving a silent gap right where the drone was
    # observed dropping out of OFFBOARD and landing a few seconds after
    # reaching altitude.
    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
    node.start_setpoint_stream()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    detector = YOLOWorldDetector()
    tracker = Tracker()

    current_phase = PHASE_SEARCH
    target = None
    camera_warned = False
    last_alt_update = time.time()
    announced_arrival = False
    arrived_frames = 0
    gemini_point = None
    gemini_area = 0.0
    center_point = None
    center_deadline = 0.0
    dialogue = Dialogue(speak, log_event)
    # The frame and target as they were when an answer was given, so a
    # follow-up can be answered from what the drone actually saw.
    last_answer_frame = None
    target_last = None
    found_context = ""
    last_report = ""
    vlm_errors = 0
    pose_log_after = 0.0
    size_log_after = 0.0
    contact_log_after = 0.0
    size_rejected_total = 0
    stray_report_after = 0.0
    locked_track_id = None
    locked_point = (0.5, 0.5)
    wanderer = SearchPattern(SWEEP_START)
    frames_lost = 0
    MAX_FRAMES_LOST = 60
    last_vlm_check_time = 0
    VLM_CHECK_INTERVAL = 15.0
    RATE_LIMIT_BACKOFF = 30.0
    # Consecutive API failures tolerated before the search is abandoned and the
    # user is told the drone cannot see. Above one, so a single network blip
    # does not end a flight; low enough that minutes are not spent flying blind.
    VLM_ERROR_LIMIT = 3

    cmd_queue = queue.Queue()
    threading.Thread(target=get_console_input, args=(cmd_queue,), daemon=True).start()
    if VOICE_ENABLED:
        threading.Thread(target=get_voice_input, args=(cmd_queue,), daemon=True).start()

    try:
        while rclpy.ok():
            # No spin_once here any more -- a background thread owns spinning,
            # so callbacks keep arriving even while this loop is blocked.

            # Command intake runs in every phase, before the phase dispatch.
            #
            # It used to live inside the SEARCH branch, so from the moment a
            # target was confirmed until tracking ended the queue was never
            # read: typed commands piled up unanswered and the prompt just
            # kept re-appearing. That leaves the user unable to correct a
            # wrong target, ask about the answer, or call the drone off while
            # it flies at something -- which is precisely when being able to
            # interrupt matters most.
            #
            # (The same block was moved above the camera-frame check earlier,
            # for the same reason: a `continue` upstream of the queue read is
            # indistinguishable, to the user, from being ignored.)
            try:
                act = dialogue.submit(cmd_queue.get_nowait())
                if act['action'] == 'quit':
                    break
                if act['action'] == 'start':
                    target = act['target']
                    # Give up only after a full sweep has actually been
                    # flown, plus room for the things the estimate leaves
                    # out: avoidance turns, climbs over the wall, and the
                    # station-keeping during each VLM call.
                    dialogue.start_search(
                        target,
                        timeout=wanderer.sweep_seconds() * SEARCH_TIMEOUT_MARGIN)
                    # Arm YOLO now, not at the hand-off to tracking: it is
                    # the screening layer during the search from here on.
                    detector.set_target(target)
                    # A target with no entry in the prompt database falls back
                    # to itself, and YOLO-World is markedly weaker on one
                    # phrasing than on four. That degradation used to be
                    # invisible: a whole flight searched for "key" on a single
                    # prompt while the "keys" entry sat unused.
                    prompts = get_prompts(target)
                    if len(prompts) == 1:
                        log_event(f"Uyari: '{target}' icin prompt genislemesi yok "
                                  f"-- tek ifadeyle araniyor, tespit zayif olabilir.")
                    else:
                        log_event(f"'{target}' icin {len(prompts)} prompt: "
                                  f"{', '.join(prompts)}")
                    last_vlm_check_time = 0
                    # A new target supersedes whatever is being chased, so the
                    # old lock has to go with it -- otherwise the drone keeps
                    # flying at the previous object under a new name.
                    current_phase = PHASE_SEARCH
                    locked_track_id = None
                    gemini_point = None
                    gemini_area = 0.0
                    vlm_errors = 0
                    node.set_target_altitude(HOVER_ALTITUDE)
                elif act['action'] == 'cancel':
                    target = None
                    current_phase = PHASE_SEARCH
                    locked_track_id = None
                    gemini_point = None
                    gemini_area = 0.0
                    node.set_target_altitude(HOVER_ALTITUDE)
                    node.set_desired_twist(
                        hold_altitude_twist(node.current_altitude()))
                elif act['action'] == 'followup':
                    # Answered from the frame kept at the moment of the
                    # result -- the user is asking about what the drone
                    # already saw, not asking it to go and look again.
                    node.set_desired_twist(
                        hold_altitude_twist(node.current_altitude()))
                    reply = answer_followup(last_answer_frame, act['question'],
                                            dialogue.context(), ref_path_for(target_last))
                    dialogue.say(reply)
            except queue.Empty:
                pass

            if current_phase == PHASE_SEARCH:
                # A search that never ends is not an answer anyone can act on.
                if dialogue.search_expired():
                    dialogue.give_up()
                    target = None
                    node.set_target_altitude(HOVER_ALTITUDE)
                    node.set_desired_twist(
                        hold_altitude_twist(node.current_altitude()))
                    continue

                ret, frame = node.cap_read()
                if not ret:
                    if not camera_warned:
                        camera_warned = True
                        msg = ("Kamera goruntusu gelmiyor (/camera/image_raw "
                               "yayinlanmiyor). Tespit baslayamaz -- "
                               "ros_gz_bridge kamera koprusu calisiyor mu?")
                        log_event(msg)
                        print(f"\n[!] {msg}")
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    time.sleep(0.03)
                    continue

                if not target:
                    cv2.imshow("DRONIONS AI",
                               draw_overlay(frame, [], None, phase=current_phase))
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    # Throttle. Without this sleep the idle "waiting for a
                    # target" path spins as fast as the CPU allows, running
                    # draw_overlay + imshow on every frame. That saturates the
                    # machine and starves the pure-python GCS heartbeat thread
                    # for longer than COM_DL_LOSS_T, so PX4 declares
                    # "Connection to ground station lost" and RTLs -- confirmed
                    # as the single flag that flips (gcs_connection_lost) while
                    # offboard/geofence/battery/position all stayed clean.
                    time.sleep(0.03)
                    continue

                sx, sy, syaw = node.horizontal_pose()

                # The pursuit leash covered only PHASE_TRACK, so nothing bounded
                # the sweep itself: a run ended with the drone at (18.9, -12.5),
                # nineteen metres outside an area 6.5 m across, and the excursion
                # was only noticed once tracking began. Recovery did eventually
                # happen -- the waypoints are all inside the area, so the
                # controller does aim back -- but slowly, and blind to the fact
                # that it was searching ground nobody asked about.
                strayed = (sx < SEARCH_AREA_X[0] - TRACK_LEASH_MARGIN
                           or sx > SEARCH_AREA_X[1] + TRACK_LEASH_MARGIN
                           or sy < SEARCH_AREA_Y[0] - TRACK_LEASH_MARGIN
                           or sy > SEARCH_AREA_Y[1] + TRACK_LEASH_MARGIN)
                if strayed:
                    if time.time() > stray_report_after:
                        log_event(f"Arama alani disinda ({sx:.1f}, {sy:.1f}) -- "
                                  f"geri donuluyor.")
                        stray_report_after = time.time() + STRAY_REPORT_INTERVAL
                    node.set_desired_twist(return_to_area_twist(sx, sy, syaw))
                    node.set_target_altitude(HOVER_ALTITUDE)
                    time.sleep(0.03)
                    continue

                if time.time() > pose_log_after:
                    log_event(f"Arama konumu ({sx:.1f}, {sy:.1f}) yaw={syaw:.2f} "
                              f"irtifa={node.current_altitude():.1f} "
                              f"hedef_wp={wanderer.current_waypoint()}")
                    pose_log_after = time.time() + POSE_LOG_INTERVAL

                node.set_desired_twist(
                    wanderer.twist(node.obstacle_ahead(), node.current_altitude(), sx, sy, syaw))

                # Say roughly where the search has got to. Rate-limited inside
                # Dialogue: a running commentary would compete with the hearing
                # a blind user needs for their own safety.
                if target:
                    wx, wy = wanderer.current_waypoint()
                    _, wbrg = relative_to_user((wx, wy), USER_POSITION, USER_YAW)
                    dialogue.narrate(f"{describe_direction(wbrg)} arıyorum.")
                # The sweep owns its own altitude now: it climbs to look over
                # what it cannot get around, and drops back to cruise after.
                node.set_target_altitude(wanderer.search_altitude())
                climb_note = wanderer.take_climb_note()
                if climb_note:
                    log_event(climb_note)
                    print(f"\n[^] {climb_note}")

                # Cheap layer first. YOLO is already on the GPU and costs
                # nothing per frame, so it screens every frame at camera rate
                # where Gemini only ever saw one frame per VLM_CHECK_INTERVAL
                # -- the drone passed 0.87 m from the box in a measured sweep
                # without a single look landing on it.
                #
                # This is an AND gate, not a replacement: Gemini still decides,
                # and the time floor still applies, so it can only ever reduce
                # the number of API calls. The floor is what stops a distractor
                # parked in frame (the blue box) from firing a request on every
                # frame -- YOLO's job here is recall, not precision.
                search_candidates = filter_candidates(detector.detect(frame))

                # Throw out anything whose physical size cannot be the target.
                # The negative prompts were meant to do this and cannot: YOLO
                # labels the wall "cardboard box", so a negative class of
                # "wall" never matches. Size does not depend on the detector
                # naming things correctly -- measured over 106 detections, this
                # rejects every one of the 69 wall hits while keeping 81% of
                # real box hits. It also saves API calls, since a viewpoint
                # left with no candidates never reaches the model at all.
                # Not in survey mode: that exists to record what the detector
                # produces unfiltered, and applying the gate first would both
                # hide the behaviour being measured and make the gate
                # impossible to evaluate against. The implied width is logged
                # per detection instead, so the threshold can be re-checked
                # offline against any future campaign.
                if target and not SURVEY:
                    dpos, dquat = node.pose_xyz(), node.orientation()
                    kept = [c for c in search_candidates
                            if size_plausible(c, dpos, dquat, target)]
                    if len(kept) != len(search_candidates):
                        size_rejected_total += len(search_candidates) - len(kept)
                        # Rate-limited rather than once-only: how often this
                        # fires is the measurement, and a single line at the
                        # start of a flight cannot show it.
                        if time.time() > size_log_after:
                            log_event(f"Boyut elemesi: {len(search_candidates)} adaydan "
                                      f"{len(kept)} tanesi '{target}' boyutunda olabilir "
                                      f"(toplam {size_rejected_total} eleme).")
                            size_log_after = time.time() + SIZE_LOG_INTERVAL
                    search_candidates = kept

                cv2.imshow("DRONIONS AI",
                           draw_overlay(frame, search_candidates, None, phase=current_phase))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

                current_time = time.time()
                if search_candidates and current_time - last_vlm_check_time > VLM_CHECK_INTERVAL:
                    ref_path = ref_path_for(target)

                    # Hold station for the duration of the call. The setpoint
                    # thread keeps flying whatever was last commanded, so the
                    # sweep otherwise carries on through the seconds the model
                    # takes to answer -- and the object it picked, typically
                    # caught at the edge of the frame as the drone swept past
                    # (measured centre x=0.02), has left the view by the time
                    # the verdict lands. The hand-off then finds nothing near
                    # it and correctly refuses, so the target is confirmed and
                    # immediately thrown away.
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))

                    # Ask which of YOLO's detections is the target, rather than
                    # whether the target is somewhere in the view. Asking for a
                    # location produced a left-edge coordinate on all six calls
                    # measured (x=0.01-0.03) no matter where the object was, so
                    # the answer could not be matched to a detection at all.
                    if FAKE_VLM:
                        result = {"found": True, "index": 0,
                                  "message": "[SAHTE-VLM] boyutu uygun ilk aday kabul edildi"}
                    elif SURVEY:
                        # Record what the detector ranked, and carry on flying.
                        #
                        # Auto-accepting YOLO's best candidate instead was the
                        # first attempt and measured almost nothing: the wall
                        # fills 68% of the frame from the takeoff spot, so every
                        # run ended one second into the search having locked
                        # onto it, and repeating that produced copies of a
                        # single viewpoint rather than a distribution over the
                        # sweep. Never handing off means one run yields a sample
                        # at every check across the whole area, which is the
                        # question worth asking: how often is the detector alone
                        # right, from the places the search actually looks?
                        for rank, c in enumerate(search_candidates[:3]):
                            w = locate_target(c, node.pose_xyz(),
                                              node.orientation(), target=target)
                            iw = implied_width(c, node.pose_xyz(), node.orientation())
                            dx, dy, _ = node.pose_xyz()
                            log_event(
                                f"ANKET r={rank} conf={c.confidence:.3f} "
                                f"alan={c.relative_area:.4f} "
                                f"genislik={f'{iw:.2f}' if iw else 'yok'} "
                                f"dunya={f'{w[0]:.2f},{w[1]:.2f}' if w else 'yok'} "
                                f"drone={dx:.2f},{dy:.2f}")
                        last_vlm_check_time = current_time
                        continue
                    else:
                        result = select_candidate(frame, search_candidates, target,
                                                  reference_img_path=ref_path)
                    msg = result['message']
                    log_event(f"Gemini Cevabı: {msg}")

                    # An API failure is not a verdict. Counted as "not this one"
                    # the drone keeps sweeping ground it cannot actually see and
                    # ends up reporting the target missing -- measured across a
                    # whole run where every call returned 404 because the model
                    # had been retired, and the user was told nothing.
                    api_error = "API Hatası" in msg
                    rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg

                    if rate_limited:
                        print(f"\n[!] Gemini kotası doldu, {RATE_LIMIT_BACKOFF:.0f}s bekleniyor...")
                        last_vlm_check_time = current_time + RATE_LIMIT_BACKOFF - VLM_CHECK_INTERVAL
                    else:
                        last_vlm_check_time = current_time

                    if api_error:
                        vlm_errors += 1
                        # One failure is a hiccup and is worth retrying: the
                        # successful run's first call died on a DNS error and
                        # its second found the box.
                        if vlm_errors >= VLM_ERROR_LIMIT:
                            dialogue.abort(vlm_failure_message(msg))
                            target = None
                            vlm_errors = 0
                            node.set_target_altitude(HOVER_ALTITUDE)
                            node.set_desired_twist(
                                hold_altitude_twist(node.current_altitude()))
                            continue
                    else:
                        # A [NONE] is a real answer and needs no announcement;
                        # the search narration already tells the user what is
                        # happening. Speaking the model's own words here read
                        # its English justification, and its errors, aloud.
                        vlm_errors = 0

                    if result['found']:
                        chosen = search_candidates[result['index']]
                        print("\n[!] Gemini hedefi doğruladı. YOLO takibine geçiliyor...")
                        speak("Hedef doğrulandı. Takibe geçiliyor.")
                        tracker = Tracker()
                        frames_lost = 0
                        last_alt_update = time.time()
                        announced_arrival = False
                        arrived_frames = 0
                        # Hand the chosen detection's own centre to the tracker
                        # hand-off. It comes from YOLO's box rather than from
                        # the model's spatial guess, so the disagreement guard
                        # downstream is checking continuity of a real detection
                        # instead of trusting a described position.
                        gemini_point = chosen.normalized_center
                        # Size of what was approved, kept for the hand-off. The
                        # centre alone cannot tell the box from the wall behind
                        # it, since a detection covering most of the frame is
                        # near every point in it.
                        gemini_area = chosen.relative_area
                        log_event(
                            f"Gemini kirpma #{result['index'] + 1} sec ti: "
                            f"merkez={gemini_point[0]:.2f},{gemini_point[1]:.2f} "
                            f"conf={chosen.confidence:.3f} alan={chosen.relative_area:.4f}")
                        center_point = gemini_point
                        center_deadline = time.time() + CENTERING_MAX_SECONDS
                        # Gemini's sentence is the one thing geometry cannot
                        # supply -- what the object looks like and what it sits
                        # next to. Kept to append to the spoken location.
                        # From the model's own split of its reply, not from
                        # slicing the raw text: it now returns a machine token,
                        # an English justification and the Turkish sentence, and
                        # only the last of those is spoken.
                        found_context = result.get('context') or ''
                        current_phase = PHASE_CENTER
                    elif not rate_limited and not result['found']:
                        print(f"\n[?] '{target}' bulunamadı. Aramaya devam ediliyor...")

            elif current_phase == PHASE_CENTER:
                # Face the confirmed target before chasing it. Yaw and altitude
                # only, never forward: the approach is what pushed a corner-of-
                # frame target out of view before tracking could establish.
                ret, frame = node.cap_read()
                if not ret:
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    time.sleep(0.03)
                    continue

                cands = filter_candidates(detector.detect(frame))
                detected_count = len(cands)
                # Same size gate as the hand-off, and it matters more here:
                # `nearest` is what the spoken answer's position is computed
                # from, so letting the wall win means reporting the wall's
                # distance as the object's, which the user has no way to doubt.
                cands = [c for c in cands
                         if gemini_area <= 0
                         or (1.0 / GEMINI_AREA_RATIO_MAX
                             <= c.relative_area / gemini_area
                             <= GEMINI_AREA_RATIO_MAX)]
                size_rejected = detected_count - len(cands)
                cx_t, cy_t = center_point
                nearest = min(cands,
                              key=lambda c: (c.normalized_center[0] - cx_t) ** 2
                                            + (c.normalized_center[1] - cy_t) ** 2,
                              default=None)
                gap = None
                if nearest is not None:
                    gap = math.hypot(nearest.normalized_center[0] - cx_t,
                                     nearest.normalized_center[1] - cy_t)

                cv2.imshow("DRONIONS AI",
                           draw_overlay(frame, cands, None, phase=current_phase))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

                if nearest is None or gap > GEMINI_POINT_MAX_DIST:
                    # Nothing to centre on this frame. Turn towards where it was
                    # rather than hovering and waiting for it to come back.
                    #
                    # Measured: the model confirms targets caught at the very
                    # edge of the sweep -- three confirmations in one flight at
                    # x=0.01, 0.02 and 0.54 -- and an object clipped by the
                    # frame border is exactly the one the detector then fails to
                    # find again. Holding station left it there until the
                    # deadline expired, throwing away a confirmation that had
                    # already cost an API call. Yawing the way it was last seen
                    # brings it inward.
                    twist = Twist()
                    twist.linear.z = compute_altitude_vz(node.current_altitude())
                    if center_point[0] < 0.5:
                        twist.angular.z = SEARCH_RECOVER_ANGULAR
                    else:
                        twist.angular.z = -SEARCH_RECOVER_ANGULAR
                    node.set_desired_twist(twist)
                    if time.time() > center_deadline:
                        # Three different failures used to share one message,
                        # which made it impossible to tell a target that drifted
                        # out of view from one the size gate was rejecting --
                        # the second means the gate is doing its job.
                        if detected_count == 0:
                            why = "hicbir sey algilanmadi"
                        elif size_rejected and not cands:
                            why = (f"{size_rejected} aday boyut disi "
                                   f"(onaylanan alan={gemini_area:.4f})")
                        elif gap is not None:
                            why = f"en yakin aday {gap:.2f} uzakta"
                        else:
                            why = "uygun aday yok"
                        msg = f"Ortalama basarisiz -- {why}, aramaya donuluyor."
                        log_event(msg)
                        print(f"\n[?] {msg}")
                        node.set_target_altitude(HOVER_ALTITUDE)
                        current_phase = PHASE_SEARCH
                    time.sleep(0.03)
                    continue

                center_point = nearest.normalized_center
                err_x = center_point[0] - 0.5
                err_y = center_point[1] - 0.5

                now = time.time()
                dt = min(0.2, now - last_alt_update)
                last_alt_update = now
                if abs(err_y) > TRACK_VERTICAL_DEADBAND:
                    node.set_target_altitude(
                        node.target_altitude() - (err_y / 0.5) * TRACK_ALT_RATE * dt)

                twist = Twist()
                twist.angular.z = -MAX_ANGULAR * max(-1.0, min(1.0, err_x / 0.5))
                node.set_desired_twist(twist)

                if abs(err_x) < CENTER_OK and abs(err_y) < CENTER_OK:
                    # Check the size before saying anything. Measured: a run
                    # centred on the wall, spoke "your box is 1.5 metres to
                    # your left", and only then did the tracking re-check drop
                    # it for being 4.2 m wide. The order was backwards -- a
                    # blind user has no way to notice that the thing just
                    # described is a wall, so the retraction comes too late to
                    # help. Nothing is spoken until the object could be the
                    # target at all.
                    if target and not size_plausible(nearest, node.pose_xyz(),
                                                     node.orientation(), target):
                        iw = implied_width(nearest, node.pose_xyz(),
                                           node.orientation())
                        msg = (f"Ortalanan nesne reddedildi: {iw:.1f} m genisliginde, "
                               f"'{target}' bu boyutta olamaz.")
                        log_event(msg)
                        print(f"\n[?] {msg}")
                        node.set_target_altitude(HOVER_ALTITUDE)
                        node.set_desired_twist(
                            hold_altitude_twist(node.current_altitude()))
                        current_phase = PHASE_SEARCH
                        continue

                    log_event(f"Hedef ortalandi (x={center_point[0]:.2f} "
                              f"y={center_point[1]:.2f}) -- takibe geciliyor.")

                    # Answer the user *now*, before flying over there. Being
                    # told where a thing is, is the assistive act; the approach
                    # afterwards is optional. The position comes from geometry
                    # -- drone pose plus camera ray onto the ground plane --
                    # rather than from asking the model where it thinks the
                    # object is, which it answers badly (measured: a left-edge
                    # coordinate on every call regardless of the truth).
                    # `target` is passed so the support surface can be chosen
                    # rather than assumed: indoors most things worth finding
                    # are on a table, and projecting those onto the floor put
                    # them over a metre too far away.
                    target_xyz = locate_target(nearest, node.pose_xyz(),
                                               node.orientation(), target=target)
                    if target_xyz:
                        # The estimate in world coordinates, logged so a run can
                        # be scored against the scenario's known object
                        # positions afterwards. Without it the log records what
                        # the drone said but not whether it locked the right
                        # object -- and the box, the blue distractor and the
                        # wall are all things it has confused before.
                        dx, dy, _ = node.pose_xyz()
                        log_event(f"Hedef konumu (dunya) x={target_xyz[0]:.2f} "
                                  f"y={target_xyz[1]:.2f} | drone x={dx:.2f} y={dy:.2f}")
                        last_report = describe_target(
                            target_xyz, USER_POSITION, USER_YAW,
                            label=f"{target.capitalize()}", context=found_context)
                        # Delivered through the dialogue so it lands in the
                        # conversation history: the next thing the user says is
                        # most likely a question about this answer.
                        target_last = target
                        last_answer_frame = frame.copy()
                        dialogue.record_answer(last_report)
                    else:
                        log_event("Konum hesaplanamadi (isin yere ulasmiyor).")
                    tracker = Tracker()
                    frames_lost = 0
                    arrived_frames = 0
                    announced_arrival = False
                    gemini_point = center_point
                    current_phase = PHASE_TRACK
                elif now > center_deadline:
                    msg = (f"Ortalama zaman asimi (x={err_x:+.2f} y={err_y:+.2f}) "
                           f"-- aramaya donuluyor.")
                    log_event(msg)
                    print(f"\n[?] {msg}")
                    node.set_target_altitude(HOVER_ALTITUDE)
                    current_phase = PHASE_SEARCH

            elif current_phase == PHASE_TRACK:
                ret, frame = node.cap_read()
                if not ret:
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    time.sleep(0.03)
                    continue

                candidates = detector.detect(frame)
                filtered_candidates = filter_candidates(candidates)
                tracked_candidates = tracker.track_objects(filtered_candidates)

                # Bound the pursuit. Same reasoning as the search area: a
                # chase with no limit is one lock-on away from leaving the
                # world entirely.
                tx, ty, _ = node.horizontal_pose()
                if (tx < SEARCH_AREA_X[0] - TRACK_LEASH_MARGIN
                        or tx > SEARCH_AREA_X[1] + TRACK_LEASH_MARGIN
                        or ty < SEARCH_AREA_Y[0] - TRACK_LEASH_MARGIN
                        or ty > SEARCH_AREA_Y[1] + TRACK_LEASH_MARGIN):
                    msg = (f"Takip alan disina cikti ({tx:.1f}, {ty:.1f}) -- "
                           f"birakiliyor, aramaya donuluyor.")
                    log_event(msg)
                    print(f"\n[!] {msg}")
                    speak("Hedef takibi iptal edildi.")
                    node.set_target_altitude(HOVER_ALTITUDE)
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    locked_track_id = None
                    gemini_point = None
                    current_phase = PHASE_SEARCH
                    continue

                nav_decision = None
                if tracked_candidates:
                    frames_lost = 0
                    if gemini_point:
                        # First tracked frame after confirmation: follow the
                        # object Gemini actually pointed at, then hold onto its
                        # track id so later frames keep following the same
                        # thing.
                        gx, gy = gemini_point
                        # Only detections of roughly the approved size are
                        # eligible. Proximity alone picked a detection covering
                        # 66% of the frame to stand in for a crop covering 0.4%
                        # -- 165 times larger, the wall rather than the box --
                        # and the drone then chased it 4 m out of the search
                        # area. Nothing about a centre distance can catch that,
                        # because a detection that large is near every point.
                        plausible = [
                            c for c in tracked_candidates
                            if gemini_area <= 0
                            or (1.0 / GEMINI_AREA_RATIO_MAX
                                <= c.relative_area / gemini_area
                                <= GEMINI_AREA_RATIO_MAX)]
                        if not plausible:
                            msg = (f"Takip iptal: onaylanan nesne alan={gemini_area:.4f}, "
                                   f"benzer boyutta aday yok "
                                   f"(en yakin {min((abs(c.relative_area - gemini_area), c.relative_area) for c in tracked_candidates)[1]:.4f}).")
                            log_event(msg)
                            print(f"\n[?] {msg}")
                            gemini_point = None
                            current_phase = PHASE_SEARCH
                            node.set_target_altitude(HOVER_ALTITUDE)
                            node.set_desired_twist(
                                hold_altitude_twist(node.current_altitude()))
                            continue
                        best_candidate = min(
                            plausible,
                            key=lambda c: (c.normalized_center[0] - gx) ** 2
                                          + (c.normalized_center[1] - gy) ** 2)
                        gap = math.hypot(best_candidate.normalized_center[0] - gx,
                                         best_candidate.normalized_center[1] - gy)
                        if gap > GEMINI_POINT_MAX_DIST:
                            # The same guard as at confirmation, repeated here
                            # because selection happens a frame later and the
                            # detections can have changed in between: that gap
                            # is how a confirmation at x=0.01 still ended up
                            # following a detection at x=0.78. Disagreement is
                            # not a target -- drop back to searching.
                            msg = (f"Takip iptal: Gemini x={gx:.2f} y={gy:.2f}, "
                                   f"en yakin YOLO adayi {gap:.2f} uzakta.")
                            log_event(msg)
                            print(f"\n[?] {msg}")
                            gemini_point = None
                            current_phase = PHASE_SEARCH
                            node.set_target_altitude(HOVER_ALTITUDE)
                            node.set_desired_twist(
                                hold_altitude_twist(node.current_altitude()))
                            continue
                        log_event(
                            f"Aday secildi (Gemini konumuna en yakin, "
                            f"fark {gap:.2f}): "
                            f"merkez={best_candidate.normalized_center[0]:.2f},"
                            f"{best_candidate.normalized_center[1]:.2f} "
                            f"conf={best_candidate.confidence:.3f} "
                            f"alan={best_candidate.relative_area:.4f}")
                        gemini_point = None
                        locked_track_id = best_candidate.track_id
                        locked_point = best_candidate.normalized_center
                    else:
                        # Stay on the object that was confirmed. Falling back to
                        # tracked_candidates[0] here silently re-targets to
                        # whatever ranks highest on the current frame, which is
                        # how a run that correctly locked onto the real box
                        # (area 0.0036) ended up declaring arrival on the wall
                        # (area ~0.12) a few frames later.
                        same_id = [c for c in tracked_candidates
                                   if locked_track_id is not None
                                   and c.track_id == locked_track_id]
                        if same_id:
                            best_candidate = same_id[0]
                        else:
                            # Track id dropped (occlusion, re-detection). Accept
                            # the nearest detection to where it last was, but
                            # only if it is close enough to plausibly be it.
                            cand = min(tracked_candidates,
                                       key=lambda c: math.hypot(
                                           c.normalized_center[0] - locked_point[0],
                                           c.normalized_center[1] - locked_point[1]))
                            jump = math.hypot(
                                cand.normalized_center[0] - locked_point[0],
                                cand.normalized_center[1] - locked_point[1])
                            if jump > GEMINI_POINT_MAX_DIST:
                                frames_lost += 1
                                nav_decision = None
                                node.set_desired_twist(
                                    navdecision_to_twist(None, node.current_altitude()))
                                if frames_lost > MAX_FRAMES_LOST:
                                    print("\n[!] Hedef kaybedildi. VLM aramasına (Mod 1) geri dönülüyor...")
                                    speak("Hedef kaybedildi. Ortam tekrar taranıyor.")
                                    node.set_target_altitude(HOVER_ALTITUDE)
                                    locked_track_id = None
                                    current_phase = PHASE_SEARCH
                                continue
                            best_candidate = cand
                            locked_track_id = cand.track_id
                        locked_point = best_candidate.normalized_center

                    # Keep checking that what is being chased is still the
                    # right size. The lock survives occlusion by design, and a
                    # measured run shows what that costs: the approach drifted
                    # onto the wall, which then filled the frame and read as
                    # arrival 2 s after the drone had been told the target was
                    # 3.7 m away.
                    if target and not size_plausible(best_candidate, node.pose_xyz(),
                                                     node.orientation(), target):
                        iw = implied_width(best_candidate, node.pose_xyz(),
                                           node.orientation())
                        msg = (f"Takip birakildi: izlenen nesne {iw:.1f} m genisliginde, "
                               f"'{target}' bu boyutta olamaz.")
                        log_event(msg)
                        print(f"\n[!] {msg}")
                        node.set_target_altitude(HOVER_ALTITUDE)
                        node.set_desired_twist(
                            hold_altitude_twist(node.current_altitude()))
                        locked_track_id = None
                        current_phase = PHASE_SEARCH
                        continue

                    # Something solid between here and the target. TRACK used to
                    # ignore the lidar entirely, which was defensible only while
                    # nothing had been seen to happen: in flight the drone flew
                    # into the wall on its way to a target 3.7 m beyond it. The
                    # approach is abandoned rather than routed around -- the
                    # answer has already been spoken, and telling the user the
                    # way is blocked is more use than silently colliding.
                    tgt_world = locate_target(best_candidate, node.pose_xyz(),
                                              node.orientation(), target=target)
                    tgt_dist = (math.dist(tgt_world, node.pose_xyz())
                                if tgt_world else None)
                    if (node.obstacle_ahead() and tgt_dist is not None
                            and tgt_dist > node.min_range() + OBSTACLE_SAFE_DISTANCE):
                        msg = (f"Yaklasilamiyor: {node.min_range():.1f} m onde engel var, "
                               f"hedef {tgt_dist:.1f} m otede.")
                        log_event(msg)
                        print(f"\n[!] {msg}")
                        dialogue.say("Hedefe yaklaşamıyorum, önümde bir engel var. "
                                     "Konumu size söyledim.")
                        node.set_target_altitude(HOVER_ALTITUDE)
                        node.set_desired_twist(
                            hold_altitude_twist(node.current_altitude()))
                        locked_track_id = None
                        current_phase = PHASE_SEARCH
                        continue

                    # Record how close the approach actually came to something
                    # solid. The wall strike that motivated the check above left
                    # no trace in the log at all -- it was only known because it
                    # was watched happening -- so a failure that appears on some
                    # runs and not others could not be counted.
                    if node.min_range() < CONTACT_RANGE and time.time() > contact_log_after:
                        log_event(f"TEMAS RISKI: takip sirasinda {node.min_range():.2f} m "
                                  f"onde engel (govde yari genisligi "
                                  f"{AIRFRAME_HALF_WIDTH:.2f} m).")
                        contact_log_after = time.time() + CONTACT_LOG_INTERVAL

                    nav_decision = get_navigation_decision(best_candidate)

                    # Match the target's height instead of holding a fixed
                    # hover: keep it vertically centred in frame by descending
                    # when it drifts low. Steering/throttle stay entirely with
                    # the navigator -- this only moves the altitude the
                    # setpoint thread holds.
                    now = time.time()
                    dt = min(0.2, now - last_alt_update)
                    last_alt_update = now
                    # ARRIVED used to be believed on a single frame, which is
                    # how one detection of a wall filling the frame produced an
                    # instant "arrived" 3.2 m from the actual box. Require it to
                    # hold across consecutive frames instead.
                    # Arrival is judged from how much of the frame the object
                    # fills, which is only a proxy for being close to it, and a
                    # measured run showed the proxy failing outright: arrival
                    # fired 2 s after the drone said the target was 3.7 m away,
                    # because the wall had grown into the frame. Geometry knows
                    # better -- cross-check it. No opinion when the projection
                    # fails, rather than blocking arrival forever.
                    if nav_decision.get("action") == "ARRIVED":
                        if tgt_dist is not None and tgt_dist > ARRIVAL_MAX_DISTANCE:
                            if arrived_frames:
                                log_event(f"VARIS reddedildi: kare doluyor ama hedef "
                                          f"geometrik olarak {tgt_dist:.1f} m otede.")
                            arrived_frames = 0
                        else:
                            arrived_frames += 1
                    else:
                        arrived_frames = 0
                    arrived = arrived_frames >= ARRIVAL_CONFIRM_FRAMES

                    if not arrived:
                        # Only servo while still closing in. Once arrived the
                        # target is nearer than the tilted camera's centre
                        # ray reaches, so it always reads "below centre" and
                        # the servo would keep sinking to the floor clamp for
                        # no benefit.
                        _, cy = best_candidate.normalized_center
                        error_y = cy - 0.5
                        if abs(error_y) > TRACK_VERTICAL_DEADBAND:
                            # +error_y = target below centre -> descend toward it.
                            delta = -(error_y / 0.5) * TRACK_ALT_RATE * dt
                            node.set_target_altitude(node.target_altitude() + delta)

                    if arrived and not announced_arrival:
                        announced_arrival = True
                        log_event(f"HEDEFE VARILDI: {target} "
                                  f"({node.current_altitude():.1f}m)")
                        print(f"\n[✓] Hedefe varıldı: {target}")
                        speak(f"{target} hedefine varıldı.")
                    # Deliberately not cleared when `arrived` goes false again:
                    # relative_area sits right on ARRIVAL_RELATIVE_AREA and
                    # dithers across it, which announced arrival over and over.
                    # It resets when the target does (loss or a new command).
                else:
                    frames_lost += 1
                    last_alt_update = time.time()
                node.set_desired_twist(navdecision_to_twist(nav_decision, node.current_altitude()))

                if frames_lost > MAX_FRAMES_LOST:
                    print("\n[!] Hedef kaybedildi. VLM aramasına (Mod 1) geri dönülüyor...")
                    speak("Hedef kaybedildi. Ortam tekrar taranıyor.")
                    # Back up to search altitude -- a wide view is what finds
                    # the target again; the tracking altitude was for closing in.
                    node.set_target_altitude(HOVER_ALTITUDE)
                    locked_track_id = None
                    current_phase = PHASE_SEARCH
                    continue

                out_frame = draw_overlay(frame, tracked_candidates, nav_decision, phase=current_phase)
                cv2.imshow("DRONIONS AI", out_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    node.set_desired_twist(hold_altitude_twist(node.current_altitude()))
                    node.set_target_altitude(HOVER_ALTITUDE)
                    current_phase = PHASE_SEARCH
                    target = None

    except KeyboardInterrupt:
        print("\nSistem kapatılıyor...")

    stop_heartbeat.set()
    # Stop the setpoint thread first, otherwise it just keeps republishing
    # altitude-hold at 30 Hz and the zero Twist below never actually sticks.
    node.stop_setpoint_stream()
    # Deliberately a bare (zero) Twist here, not altitude-hold: on shutdown
    # we stop commanding entirely and let PX4's own failsafe (COM_OF_LOSS_T)
    # bring it back via RTL -- confirmed working in testing. This node does
    # not implement its own landing sequence.
    # rclpy's own SIGINT/SIGTERM handler may already have invalidated the
    # context by the time we get here -- guard against publishing/shutting
    # down twice (same issue as the ground-rig node's shutdown).
    if rclpy.ok():
        node.publish_twist(Twist())
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    log_event("System shutdown.")


if __name__ == '__main__':
    main()
