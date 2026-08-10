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

## 3. Running the PX4 simulation

One command. It brings up all four processes, waits for each to be ready, and
leaves the node in the foreground on your keyboard:

```bash
cd ~/DRONIONS && INTERACTIVE=1 HEADLESS=0 scripts/run_sim_chain.sh
```

Type a target (`kutuyu bul`, `box`) at the prompt once it is airborne, then
**Enter** to confirm. `h` cancels, `q` quits. Logs go to `/tmp/dronions-sim/`
per process, and cumulatively to `logs/dronions_run.log`.

Without `INTERACTIVE=1` the script pipes one target in and runs unattended,
which is what the measurement scripts use. Do not use that mode to test
anything involving the dialogue: a conversation cannot be scripted ahead of
time, and the confirmation step will sit waiting for an answer a closed pipe
can never give.

The four processes, if you need them separately:

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

PX4 has its **own** venv (`~/PX4-Autopilot/px4venv`) for its build tooling. Do
not use the DRONIONS venv for it — numpy versions conflict.

## 4. Flight phase machine

`SEARCH` → `CENTER` → `TRACK`, falling back to `SEARCH` at every failure.

Command intake runs **above** this dispatch, in every phase — see §5
Interaction for why that matters.

- **SEARCH** — bounded lawnmower sweep (`SearchPattern`) over `SEARCH_AREA_X/Y`,
  rows along x. YOLO screens every frame; Gemini is called only when YOLO has a
  candidate **and** `VLM_CHECK_INTERVAL` has elapsed. Lidar contact commits a
  randomized turn; a waypoint that stays blocked is climbed over
  (`SEARCH_CLIMB_STEP` up to `MAX_SEARCH_ALTITUDE`) and abandoned only if that
  ceiling is reached. Straying outside the area overrides the waypoint with a
  direct return; position is logged every `POSE_LOG_INTERVAL`.
- **CENTER** — yaw and altitude-servo onto the confirmed object until it is
  within `CENTER_OK` of frame centre. No forward motion. This phase exists
  because the sweep almost always confirms the target at the frame edge, and
  driving forward from there loses it. Candidates are gated by size against the
  crop Gemini approved (`GEMINI_AREA_RATIO_MAX`); the user's spoken answer is
  computed here, so a wrong association here is a wrong answer.
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
  degrades to a single weak prompt. The lookup is by exact string, so near
  misses cost the same as unknown words — a flight searched for `"key"` on one
  prompt while the `"keys"` entry sat unused. Singular/plural variants are now
  tried, and the node logs the expansion at the start of every search so the
  degradation is visible rather than silent.
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

- **Match the hand-off on size, not just position.** A detection covering 66% of
  the frame stood in for a crop covering 0.4% — the wall rather than the box —
  with a reported centre gap of `0.00`, because a detection that large is near
  every point in the frame. `GEMINI_AREA_RATIO_MAX` gates it. The gate belongs
  in the CENTER phase too, since that is where the spoken answer's position is
  computed: without it the user is told the wall's distance and has no way to
  doubt it.

### Interaction

- **Read the command queue in every phase.** Intake once sat inside the SEARCH
  branch, so from target confirmation until tracking ended nothing was read.
  Typed commands piled up and the prompt just re-appeared, which is
  indistinguishable from being ignored — and it left the user unable to call
  the drone off while it flew at the wrong thing. Any `continue` upstream of
  the queue read causes the same symptom; this has now been the bug twice.
- **A new target must clear the tracking lock**, or the drone keeps chasing the
  previous object under a new name.
- **Derive the give-up deadline from the search pattern.** A picked 180 s was
  shorter than one lap (198 s), so the drone reported ground empty that it had
  never flown over.
- **An API failure is not a verdict.** Counted as "not this one", the drone
  sweeps ground it cannot see and eventually reports the object missing. Three
  consecutive failures stop the search and say why.
- **Never speak raw API text.** `result['message']` went straight to TTS, so a
  404 JSON body was read aloud as the drone's reply. The same applies to the
  model's own justification: `select_candidate` is asked for a Turkish sentence
  addressed to the user precisely because that sentence is spoken.

### Gemini quota

Free tier is **20 requests/day per Google Cloud project per model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Issuing a new API key in
the same project does **not** reset it — switching model does, since each has
its own allowance. `GEMINI_MODEL` in `config.py` (or `.env`) exists for exactly
this. A single test run at `VLM_CHECK_INTERVAL = 15.0` can exhaust one model's
daily allowance.

A 429 is **not always the daily cap** — there is a per-minute limit too, and it
recovers. One run took two 429s mid-search and then got a clean `[MATCH]` thirty
seconds later. Only consecutive failures are treated as fatal, for this reason.

Models also retire. `gemini-2.5-flash` now answers a **newly issued** key with
404 `no longer available to new users` while existing keys still reach it, so
renewing a key can blind the system in a way that looks like nothing at all.
The default is `gemini-3.5-flash`; pin the old one in `.env` to reproduce the
measurements in `docs/`.

### Hardware quirks on this machine

Hybrid AMD iGPU + NVIDIA RTX. Gazebo does not pick the NVIDIA card by itself and
falls back to a renderer slow enough that, with the GUI up, the sim stalls PX4
through lockstep and MAVROS drops the link. `scripts/run_px4_sim.sh` sets the
offload variables. YOLO is unaffected — it runs on the GPU via `DEVICE = "cuda"`.

`px4-rc.gzsim` decides on the GUI with `[ -z "$HEADLESS" ]` — **emptiness, not
value**. `HEADLESS=0` suppresses the window exactly like `HEADLESS=1` does, so
the variable has to be *unset* for a visible Gazebo. `run_sim_chain.sh` removes
it from the child environment rather than passing `0`.

## 6. Verified in flight

Perception and flight, across many runs: Gemini rejects the green sphere and the
wall panel by name and matches the real box on its taped seam against the
reference photo; the CENTER phase rescues detections confirmed at the frame edge
(as far out as (0.98, 0.91)); lidar avoidance routes around the wall and the
sweep climbs over it when it cannot; zero failsafes across multi-minute flights;
bridge rates ~27 Hz camera / ~50 Hz lidar.

Every path through the interaction layer has now fired in flight, not only in
unit tests:

| Path | Evidence |
|---|---|
| Confirm before acting | `Onaylayın: box aramamı istiyorsunuz` → Enter → search |
| Correction attached to rejection | `no,phone` → confirms `phone`, tracking lock cleared |
| Progress narration, rate-limited | direction changes spoken, repeats held to 30 s |
| Location answer | `Box hafif solunuzda, yaklaşık 4.2 metre (6 adım) ötede, yerde.` |
| Follow-up from the kept frame | *"üzerinde yazı var mı?"* → *"Hayır… ortasından geçen şeffaf bir bant şeridi var."* — correct, no re-flight |
| Bounded give-up | fired at **317 s**, after a full lap (195 s) plus half of a second |
| Vision-service abort | three consecutive 429s → *"günlük kullanım hakkı doldu… yarın tekrar deneyebiliriz"* |

The sweep's own traversal estimate (`SearchPattern.sweep_seconds()`) predicted
198 s for one lap; the logged position trace completed one in 195 s. That is
what the give-up deadline is derived from, rather than a picked constant.

Localization measured against Gazebo ground truth, n=21: **2.8° median bearing
error, 0.49 m median distance error**.

## 7. Not verified / open

- **Repeatability of the whole chain** is better than it was — several
  end-to-end successes, fastest 43 s — but success still depends on Gemini
  answering. Two runs failed purely on the API (one on a retired-model 404, one
  on daily quota) and neither was a fault of the search.
- **Why the drone once flew to (18.9, −12.5)**, nineteen metres outside a 6.5 m
  area, is still unexplained. The sweep now logs its position every 10 s and
  turns back at the boundary, so a recurrence will leave a trace; the original
  had none.
- **The `TRACK_LEASH_MARGIN` leash has now fired** (it caught that excursion),
  but only as a symptom of the above, never as designed.
- Camera is 1280×960 while `IMAGE_SIZE = 640`, so frames are downscaled before
  inference. The resolution costs image-plumbing CPU, not inference.
- `obstacle_ahead()` is used in SEARCH only; TRACK ignores the lidar entirely.
  Deliberate, not an oversight.
- **Indoor scenarios are still blocked** by `OBSTACLE_SAFE_DISTANCE = 1.0` m
  against ~0.8 m doorways.
