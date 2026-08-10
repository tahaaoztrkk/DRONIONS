# Dronions: Miniature Personal Drone Assistants for Accessible Everyday Interaction

This project focuses on the design and development of "Dronions", miniature
personal drones that operate from a wearable docking station located on the
user's wrist.

You name an object in plain language — *"kutuyu bul"*, *"find my charger"* — and
the drone searches the environment, works out which one is **yours**, and tells
you where it is **relative to you**: *"Box hafif solunuzda, yaklaşık 4.2 metre
(6 adım) ötede, yerde."*

That last part is the point. The drone answering from its own point of view is
answering the wrong question, because the user then has to translate it. See
[docs/PLAN_EVERYDAY_SCENARIOS.md](docs/PLAN_EVERYDAY_SCENARIOS.md) §5b for the
experiment that measures how much this matters.

## How it finds things

Two perception layers with different strengths, in a cascade:

| Layer | Runs | Role |
|---|---|---|
| **YOLO-World** (zero-shot, GPU) | every frame | Screening. Finds *box-like things* cheaply. Fully offline. |
| **Gemini** (VLM) | gated | Deciding. Given crops of YOLO's detections plus a reference photo, says which one is yours. The only online part. |
| **ByteTrack** | every frame | Continuity. Keeps the confirmed object locked during the approach. |

The **Visual Memory Bank** is what makes it *your* object rather than *an*
object: drop a reference photo at `memory/<target>.jpg` and Gemini matches
against it. In testing this is what separated the real cardboard box from a
same-shaped blue box beside it — it recognised the taped seam.

Gemini is only asked when YOLO already has a candidate and a minimum interval has
elapsed, which keeps the expensive layer rare without blinding the cheap one.
When the model's verdict is handed to the tracker, candidates are matched on
**size as well as position**: a detection covering most of the frame is near
every point in it, so proximity alone once let the wall stand in for the box.

## Talking to it

The interaction layer follows the four findings in the CHI '26 reference work
(Wei et al.), all of them verified in flight here:

- **Confirms before acting.** A misheard target that flies is worse than one
  that asks. A rejection and its correction arrive together — *"hayır, telefon"*
  — and are handled as one utterance.
- **Narrates where it is looking**, rate-limited. Hearing is a safety channel
  for a blind user, so a repeated update waits far longer than a new one.
- **Expects follow-ups.** *"Üzerinde yazı var mı?"* is answered from the frame
  kept at the moment of the result, without flying again — 35% of user queries
  in the reference study were follow-ups.
- **Says less.** Replies are capped at two short sentences.

Two failure modes are reported as *different answers*, because a user who cannot
look has no other way to tell them apart:

| Situation | What it says |
|---|---|
| Area swept, nothing found | *"…bulunamadı. Aradığım alanı tamamen taradım."* |
| Vision service unreachable | *"…göremiyorum. Aramayı durduruyorum."* + why |

The give-up deadline is derived from the sweep's own traversal time rather than
picked: a fixed 180 s was shorter than one lap of the search area, so the drone
reported ground empty that it had never flown over.

## Simulation

Two environments, both working. Full instructions and the hard-won gotchas live
in **[AI_HANDOVER_ROS.md](AI_HANDOVER_ROS.md)** — read that before running
anything.

### PX4 quadrotor (the real thing)

One command brings up PX4 SITL, MAVROS, the sensor bridge and the node, and
leaves the node on your keyboard:

```bash
INTERACTIVE=1 HEADLESS=0 scripts/run_sim_chain.sh
```

`HEADLESS` must be **unset**, not `0`, for the Gazebo window — PX4 tests
`[ -z "$HEADLESS" ]`, so any value at all suppresses the GUI. The script handles
this; the note is for when you run PX4 by hand.

Logs land in `/tmp/dronions-sim/` (one per process) and, cumulatively across
runs, in `logs/dronions_run.log`.

The drone takes off, sweeps the area in a bounded lawnmower pattern, climbs over
obstacles it cannot get around, centres on the target once confirmed, reports
where it is, and then approaches it.

### Ground rig (simpler, no flight stack)

A differential-drive rig carrying the same camera and lidar.

```bash
source /opt/ros/jazzy/setup.bash && source venv/bin/activate
ros2 launch ros/launch/dronions_sim.launch.py    # terminal 1
python3 ros/dronions_ros_node.py                 # terminal 2
```

## Layout

```
assistant/     Gemini agent, dialogue state machine, command parsing, speech
perception/    YOLO-World detector, candidate filtering, tracker
navigation/    steering from a bounding box + geometric localization
ui/            on-screen overlay
ros/           ROS 2 nodes, launch files, ground-rig world
scripts/       run helpers and measurement experiments
docs/          scenario plan, measured results
memory/        Visual Memory Bank reference photos
```

`navigation/navigator.py` is deliberately actuation-agnostic — it emits
`turn`/`throttle`, and each node maps that onto its own vehicle. That is what let
the PX4 port reuse it unchanged. `navigation/spatial.py` holds the localization:
a camera ray through the **bottom edge** of the bounding box, projected onto the
ground plane, then expressed relative to the user. Measured against Gazebo
ground truth, n=21: **2.8° median bearing error, 0.49 m median distance error**.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
```

Gemini's free tier allows **20 requests/day per project per model**. A new API
key in the same project does *not* reset it; switching `GEMINI_MODEL` does, since
each model has its own allowance. Note that a newly issued key cannot reach
models that have entered retirement — `gemini-2.5-flash` answers new keys with
404, which cost a whole test flight before the drone learned to say so.
