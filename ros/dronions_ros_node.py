#!/usr/bin/env python3
"""
ROS 2 wrapper for the DRONIONS hybrid VLM+YOLO pipeline.

Replaces main.py's cv2.VideoCapture camera input and geometry_msgs-less
navigation output with a Gazebo-simulated camera (subscribed over ROS 2) and
a geometry_msgs/Twist publisher, without touching the Gemini/YOLO/navigation/
voice logic itself. main.py and config.py are untouched, so the Windows/IP
camera path keeps working exactly as before.

Run (from a shell with ROS 2 sourced and the DRONIONS venv activated):
    source /opt/ros/jazzy/setup.bash
    source venv/bin/activate
    python3 ros/dronions_ros_node.py

Requires ros/launch/dronions_sim.launch.py already running in another
terminal (provides /camera/image_raw and /cmd_vel).
"""
import os
import sys
import threading
import queue
import time
import math

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
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

from config import VOICE_ENABLED
from assistant.command_parser import parse_command
from assistant.agent import capture_and_analyze
from assistant.speech import speak
from assistant.listen import get_voice_input
from utils.logger import log_event

from perception.detector import YOLOWorldDetector
from perception.filters import filter_candidates
from perception.tracker import Tracker
from navigation.navigator import get_navigation_decision
from ui.overlay import draw_overlay

PHASE_SEARCH = "VLM SEARCHING"
PHASE_TRACK = "YOLO TRACKING"

# Kept under the sim world's DiffDrive plugin limits (0.5 m/s / 1.0 rad/s,
# see ros/worlds/dronions_diff_drive_camera.sdf) as a second safety layer.
MAX_LINEAR = 0.3
MAX_ANGULAR = 0.6

# While searching, wander forward on a slow curve (drive + gentle turn at the
# same time -> traces a wide arc) instead of only rotating in place, so the
# vehicle actually changes position and can find sightlines a pure in-place
# scan never would (e.g. around an obstacle). Turn radius = linear/angular,
# so this is deliberately a *wide* curve (~4m radius) -- tight enough to
# eventually cover every direction, wide enough to actually cover ground
# instead of just circling near the spawn point. Sped up (both scaled
# together, same radius) since finding the target faster means fewer
# VLM_CHECK_INTERVAL cycles spent, and fewer Gemini calls against the
# free-tier quota.
EXPLORE_LINEAR = 0.25
SEARCH_ANGULAR = 0.0625

# Obstacle avoidance (forward lidar, see ros/worlds/*.sdf): if something is
# closer than this, stop driving into it and rotate in place (faster than
# the gentle explore turn) until the way ahead is clear again.
OBSTACLE_SAFE_DISTANCE = 1.0  # meters
AVOID_ANGULAR = 0.35


def navdecision_to_twist(nav_decision):
    """Maps navigation.navigator's action dict to a Twist command.

    Kept as its own function -- this is the only piece that needs to change
    when the ground rig is later swapped for PX4 SITL + MAVROS.
    """
    twist = Twist()
    if not nav_decision or nav_decision.get("action") in ("SEARCHING", "ARRIVED"):
        return twist  # zero Twist: hold position (still searching, or reached the target)

    # turn: -1 (full left) .. +1 (full right); throttle: 0..1. Both proportional,
    # so the vehicle steers and moves forward at the same time instead of
    # stopping dead to turn, and eases off as it nears the target.
    turn = nav_decision.get("turn", 0.0)
    throttle = nav_decision.get("throttle", 0.0)
    twist.linear.x = MAX_LINEAR * throttle
    twist.angular.z = -MAX_ANGULAR * turn
    return twist


def compute_search_twist(obstacle_ahead: bool) -> Twist:
    """Wander-and-avoid Twist used while actively searching for a target.

    Clear ahead: drive forward on a slow curve so the vehicle actually
    explores (position changes, not just orientation). Blocked: stop and
    rotate in place (always the same direction, for a predictable, non-
    oscillating turn-away) until the path clears, then resume exploring.
    """
    twist = Twist()
    if obstacle_ahead:
        twist.angular.z = AVOID_ANGULAR
    else:
        twist.linear.x = EXPLORE_LINEAR
        twist.angular.z = SEARCH_ANGULAR
    return twist


def get_console_input(q: queue.Queue):
    while True:
        cmd = input("\n[DRONIONS] Lütfen bir komut girin ('çıkış' veya 'q' ile kapatın): ")
        q.put(cmd)
        if cmd.lower() in ['q', 'çıkış', 'quit', 'exit']:
            break


class DronionsRosNode(Node):
    """Subscribes to the simulated camera and publishes Twist commands.

    cap_read() mirrors cv2.VideoCapture.read()'s (ret, frame) contract so
    main.py's phase loop can be reused almost line-for-line.
    """

    def __init__(self):
        super().__init__('dronions_ros_node')
        self._frame = None
        self._lock = threading.Lock()
        self._min_range = float('inf')
        self.create_subscription(Image, '/camera/image_raw', self._on_image, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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
        finite = [r for r in msg.ranges if math.isfinite(r)]
        self._min_range = min(finite) if finite else float('inf')

    def obstacle_ahead(self) -> bool:
        return self._min_range < OBSTACLE_SAFE_DISTANCE

    def publish_twist(self, nav_decision):
        self.cmd_pub.publish(navdecision_to_twist(nav_decision))


def main():
    rclpy.init()
    node = DronionsRosNode()

    log_event("Initializing DRONIONS ROS node (Hybrid VLM + YOLO)...")
    speak("System activated. I am listening for your commands.")

    detector = YOLOWorldDetector()
    tracker = Tracker()

    current_phase = PHASE_SEARCH
    target = None
    frames_lost = 0
    MAX_FRAMES_LOST = 60  # ~2s at 30fps before falling back to VLM search
    last_vlm_check_time = 0
    # Free-tier Gemini quota is ~5 requests/minute; 5s would demand 12/min and
    # get rate-limited almost immediately. 15s keeps it under that budget.
    VLM_CHECK_INTERVAL = 15.0
    RATE_LIMIT_BACKOFF = 30.0  # extra wait added on top of VLM_CHECK_INTERVAL after a 429

    cmd_queue = queue.Queue()
    threading.Thread(target=get_console_input, args=(cmd_queue,), daemon=True).start()
    if VOICE_ENABLED:
        threading.Thread(target=get_voice_input, args=(cmd_queue,), daemon=True).start()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            if current_phase == PHASE_SEARCH:
                ret, frame = node.cap_read()
                if not ret:
                    time.sleep(0.03)  # no frame from Gazebo yet
                    continue

                cv2.imshow("DRONIONS AI", draw_overlay(frame, [], None, phase=current_phase))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                try:
                    user_input = cmd_queue.get_nowait()
                    if user_input.lower() in ['q', 'çıkış', 'quit', 'exit']:
                        break
                    if user_input.strip():
                        parsed_cmd = parse_command(user_input)
                        target = parsed_cmd.get("target") or user_input.strip()
                        log_event(f"Hedef belirlendi: {target}")
                        speak(f"{target} aranıyor...")
                        last_vlm_check_time = 0
                except queue.Empty:
                    pass

                if not target:
                    continue

                # Actively explore for the target instead of waiting for it
                # to already be in view: wander forward on a slow curve,
                # turning away in place if the lidar sees something blocking
                # the way ahead. The Twist persists in Gazebo even while this
                # loop blocks on the (network) Gemini call below, so movement
                # continues smoothly through that pause too.
                node.cmd_pub.publish(compute_search_twist(node.obstacle_ahead()))

                current_time = time.time()
                if current_time - last_vlm_check_time > VLM_CHECK_INTERVAL:
                    target_clean = target.lower().strip()
                    target_underscore = target_clean.replace(" ", "_")
                    ref_path = None
                    for ext in [".jpg", ".png", ".jpeg"]:
                        for name in [target_clean, target_underscore]:
                            path = os.path.join("memory", name + ext)
                            if os.path.exists(path):
                                ref_path = path
                                break
                        if ref_path:
                            break

                    result = capture_and_analyze(frame, target, reference_img_path=ref_path)
                    log_event(f"Gemini Cevabı: {result['message']}")

                    rate_limited = "429" in result['message'] or "RESOURCE_EXHAUSTED" in result['message']
                    if rate_limited:
                        print(f"\n[!] Gemini kotası doldu, {RATE_LIMIT_BACKOFF:.0f}s bekleniyor...")
                        # Push the next attempt out further instead of retrying every
                        # VLM_CHECK_INTERVAL and digging the rate limit deeper.
                        last_vlm_check_time = current_time + RATE_LIMIT_BACKOFF - VLM_CHECK_INTERVAL
                    else:
                        speak(result['message'])
                        last_vlm_check_time = current_time

                    if result['found']:
                        print("\n[!] Gemini hedefi doğruladı. YOLO takibine geçiliyor...")
                        speak("Hedef doğrulandı. Takibe geçiliyor.")
                        node.cmd_pub.publish(Twist())  # stop scanning before handing off to TRACK
                        detector.set_target(target)
                        tracker = Tracker()
                        frames_lost = 0
                        current_phase = PHASE_TRACK
                    elif not rate_limited:
                        print(f"\n[?] '{target}' bulunamadı. Aramaya devam ediliyor, lütfen kamerayı etrafta gezdirin...")

            elif current_phase == PHASE_TRACK:
                ret, frame = node.cap_read()
                if not ret:
                    time.sleep(0.03)
                    continue

                candidates = detector.detect(frame)
                filtered_candidates = filter_candidates(candidates)
                tracked_candidates = tracker.track_objects(filtered_candidates)

                nav_decision = None
                if tracked_candidates:
                    frames_lost = 0
                    best_candidate = tracked_candidates[0]
                    nav_decision = get_navigation_decision(best_candidate)
                else:
                    frames_lost += 1
                node.publish_twist(nav_decision)  # None -> zero Twist == stop

                if frames_lost > MAX_FRAMES_LOST:
                    print("\n[!] Hedef kaybedildi. VLM aramasına (Mod 1) geri dönülüyor...")
                    speak("Hedef kaybedildi. Ortam tekrar taranıyor.")
                    current_phase = PHASE_SEARCH
                    continue

                out_frame = draw_overlay(frame, tracked_candidates, nav_decision, phase=current_phase)
                cv2.imshow("DRONIONS AI", out_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    node.publish_twist(None)
                    current_phase = PHASE_SEARCH
                    target = None

    except KeyboardInterrupt:
        print("\nSistem kapatılıyor...")

    # rclpy installs its own SIGINT/SIGTERM handler that may already have
    # invalidated the context by the time we get here (e.g. Ctrl+C or an
    # external `timeout`/kill) -- guard against publishing/shutting down twice.
    if rclpy.ok():
        node.publish_twist(None)
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    log_event("System shutdown.")


if __name__ == '__main__':
    main()
