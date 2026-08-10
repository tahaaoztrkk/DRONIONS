# AI HANDOVER: DRONIONS ROS 2 / GAZEBO / PX4

> **To the next assistant:** read this before touching anything. Most of what is
> below was found empirically and cost real debugging time. These failure modes
> are not guessable from the code.

## 1. What this is

DRONIONS is an autonomous target-finding drone. A user names an object in plain
language; the drone searches for it, identifies it, approaches it and announces
arrival. Identification is hybrid:

- **YOLO-World** (`ultralytics`, zero-shot, GPU) screens **every** camera frame.
  Cheap, high recall, no idea *which* box is yours.
- **Gemini** (`google-genai`) decides. It receives crops of YOLO's detections
  plus the reference photo from the **Visual Memory Bank**
  (`memory/<target>.jpg`) and answers which crop is the target. Precise,
  expensive, rate-limited.
- **ByteTrack** (`trackers` package) keeps the confirmed object across frames.

Voice in/out (`assistant/listen.py`, `assistant/speech.py`) run on their own
threads and are unaffected by any of the below.

Note the ordering: **YOLO runs first and gates Gemini**, not the reverse. Gemini
is called only when YOLO already has something worth asking about.

## 2. Two simulations, both working

| | Ground rig | PX4 quadrotor |
|---|---|---|
| Node | `ros/dronions_ros_node.py` | `ros/dronions_ros_node_px4.py` |
| World | `ros/worlds/dronions_diff_drive_camera.sdf` | `PX4-Autopilot/Tools/simulation/gz/worlds/dronions_scenario.sdf` |
| Actuation | `/cmd_vel` via `ros_gz_bridge` | MAVROS → PX4 OFFBOARD |
| Launch | `ros/launch/dronions_sim.launch.py` | 4 terminals, see below |

The ground rig is the simpler one and still works; keep it. The PX4 node is a
**separate file** on purpose — `navigation/navigator.py` is shared and was not
modified by the port.

## 3. Running the PX4 simulation (4 terminals)

```bash
# 1 — PX4 SITL + Gazebo
cd ~/DRONIONS && ./scripts/run_px4_sim.sh

# 2 — MAVROS (must be PX4's "Onboard" instance, port 14580)
source /opt/ros/jazzy/setup.bash
ros2 launch mavros px4.launch fcu_url:="udp://:0@127.0.0.1:14580"

# 3 — camera + lidar bridge (sensors only; MAVLink owns actuation)
cd ~/DRONIONS && source /opt/ros/jazzy/setup.bash
ros2 launch ros/launch/dronions_px4_bridge.launch.py

# 4 — the node
cd ~/DRONIONS && source /opt/ros/jazzy/setup.bash && source venv/bin/activate
python3 ros/dronions_ros_node_px4.py
```

Type a target (e.g. `box`) at the prompt once it is airborne.

PX4 has its **own** venv (`~/PX4-Autopilot/px4venv`) for its build tooling. Do
not use the DRONIONS venv for it — numpy versions conflict.

## 4. Flight phase machine

`SEARCH` → `CENTER` → `TRACK`, falling back to `SEARCH` at every failure.

- **SEARCH** — bounded lawnmower sweep (`SearchPattern`) over `SEARCH_AREA_X/Y`,
  rows along x. YOLO screens every frame; Gemini is called only when YOLO has a
  candidate **and** `VLM_CHECK_INTERVAL` has elapsed. Lidar contact commits a
  randomized turn; a waypoint that stays blocked is abandoned.
- **CENTER** — yaw and altitude-servo onto the confirmed object until it is
  within `CENTER_OK` of frame centre. No forward motion. This phase exists
  because the sweep almost always confirms the target at the frame edge, and
  driving forward from there loses it.
- **TRACK** — approach via `navigator.get_navigation_decision()`, following the
  **locked `track_id`**, altitude servoing on the target's vertical position in
  frame. `ARRIVED` must hold for `ARRIVAL_CONFIRM_FRAMES`.

## 5. Things that will waste your time if you don't know them

### PX4 / MAVROS

- **MAVROS rejects `param/set` until it has downloaded PX4's full parameter
  list** (~15-30 s after link-up). Called earlier, every set fails *instantly*
  and silently. The node now retries until they stick and **refuses to fly** if
  they don't. Symptom when broken: takeoff, then Hold → RTL → land, every single
  time, because `GF_ACTION` was left at its default of 2 (Hold mode).
- **`COM_RC_IN_MODE` must be 4** ("ignore all sources"), not 1 ("MAVLink only").
  Under 1 PX4 still expects MANUAL_CONTROL messages and raises
  `manual_control_signal_lost` the moment you take off.
- **PX4's arming check wants a GCS on the "Normal" MAVLink instance (18570)**,
  not the Onboard one MAVROS uses. The node sends synthetic heartbeats there
  itself. Without it, arming is refused with no explanation.
- **The setpoint stream must never stop.** It lives on its own thread
  (`_setpoint_loop`); the main loop only *sets* a twist. Anything blocking —
  Gemini, the YOLO model load — would otherwise drop OFFBOARD.
- **A busy loop starves the GCS heartbeat thread.** An idle path without a
  `sleep` saturated the CPU, delayed the pure-python heartbeat past
  `COM_DL_LOSS_T`, and PX4 RTL'd on "Connection to ground station lost".
- Diagnose failsafes with
  `build/px4_sitl_default/bin/px4-listener failsafe_flags` — it names the exact
  flag that flipped. The console usually does not.
- `gz sim` is started with `&` inside PX4's own script, so **Ctrl+C does not
  kill it**. Stale instances get silently reused with corrupted physics state.
  Verify with `ps aux` before every relaunch.

### Perception

- **Prompt expansion is not optional.** With the bare prompt `"box"`, YOLO-World
  detected *only the 3 m wall* and missed the actual cardboard box entirely.
  `utils/prompts.py` `PROMPT_DATABASE` fixes this; a target with no entry
  degrades to a single weak prompt.
- **Negative prompts are fed as ordinary classes** and dropped in
  `detector._parse_result`. A negative matching a large part of the scene
  distorts the rest — adding `"floor"`/`"ground"` measurably flipped the ranking
  the wrong way. Keep negatives to things actually confused for the target.
- **Do not rank candidates by size or centredness.** `filter_candidates` used to
  reward both, which put the wall above the real box even when the box scored
  higher on confidence. Ranking is confidence only.
- **Do not ask Gemini where something is.** Across six calls it answered with a
  left-edge coordinate (`x=0.01-0.03`) regardless of the true position. Ask
  *which crop* instead (`agent.select_candidate`) — comparison it does well.
- **Hold station during the Gemini call.** It takes seconds, the drone keeps
  flying, and the object it picked has left the frame by the time the answer
  lands.
- **Keep the locked `track_id`.** Falling back to "highest confidence on this
  frame" silently re-targets: a run that correctly locked the real box declared
  arrival on the wall a few frames later.

### Gemini quota

Free tier is **20 requests/day per Google Cloud project per model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Issuing a new API key in
the same project does **not** reset it — switching model does, since each has
its own allowance. `GEMINI_MODEL` in `config.py` (or `.env`) exists for exactly
this. A single test run at `VLM_CHECK_INTERVAL = 15.0` can exhaust one model's
daily allowance.

### Hardware quirks on this machine

Hybrid AMD iGPU + NVIDIA RTX. Gazebo does not pick the NVIDIA card by itself and
falls back to a renderer slow enough that, with the GUI up, the sim stalls PX4
through lockstep and MAVROS drops the link. `scripts/run_px4_sim.sh` sets the
offload variables. YOLO is unaffected — it runs on the GPU via `DEVICE = "cuda"`.

## 6. Verified

One full run on the PX4 quad with the wall and both distractors present: Gemini
rejected the green sphere and the wall panel by name, matched the real box on its
taped seam against the reference photo, the CENTER phase rescued a detection at
(0.98, 0.91) — the very corner of the frame — the hand-off matched to 0.01, and
the drone arrived at the correct box.

Verified separately: lidar avoidance against the wall (the drone routes around
it), zero failsafes across multi-minute flights, bridge rates ~27 Hz camera /
~50 Hz lidar.

## 7. Not verified / open

- **Repeatability.** The end-to-end success above is a single run.
- **The tracking leash** (`TRACK_LEASH_MARGIN`) has never actually fired. It was
  added after a pursuit flew 131 m out of the scenario.
- **The drone never considers flying over the wall.** The wall is 3 m, cruise is
  `HOVER_ALTITUDE = 2.0` m, and the search has no vertical dimension at all — so
  going over is not merely unchosen, it is unrepresented. Adding it would make
  the quad's 3D advantage matter and would find the target faster.
- Camera is 1280×960 while `IMAGE_SIZE = 640`, so frames are downscaled before
  inference. The resolution costs image-plumbing CPU, not inference.
- `obstacle_ahead()` is used in SEARCH only; TRACK ignores the lidar entirely.
  Deliberate, not an oversight.
