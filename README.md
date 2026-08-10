# Dronions: Miniature Personal Drone Assistants for Accessible Everyday Interaction

This project focuses on the design and development of "Dronions", miniature
personal drones that operate from a wearable docking station located on the
user's wrist.

You name an object in plain language — *"box"*, *"charger"* — and the drone
searches the environment for it, identifies **your** specific one, flies to it
and tells you it has arrived.

## How it finds things

Two perception layers with different strengths, in a cascade:

| Layer | Runs | Role |
|---|---|---|
| **YOLO-World** (zero-shot, GPU) | every frame | Screening. Finds *box-like things* cheaply. |
| **Gemini** (VLM) | gated | Deciding. Given crops of YOLO's detections plus a reference photo, says which one is yours. |
| **ByteTrack** | every frame | Continuity. Keeps the confirmed object locked during the approach. |

The **Visual Memory Bank** is what makes it *your* object rather than *an*
object: drop a reference photo at `memory/<target>.jpg` and Gemini matches
against it. In testing this is what separated the real cardboard box from a
same-shaped blue box beside it — it recognised the taped seam.

Gemini is only asked when YOLO already has a candidate and a minimum interval has
elapsed, which keeps the expensive layer rare without blinding the cheap one.

## Simulation

Two environments, both working. Full instructions and the hard-won gotchas live
in **[AI_HANDOVER_ROS.md](AI_HANDOVER_ROS.md)** — read that before running
anything.

### Ground rig (simpler, start here)

A differential-drive rig carrying the same camera and lidar. No flight stack.

```bash
source /opt/ros/jazzy/setup.bash && source venv/bin/activate
ros2 launch ros/launch/dronions_sim.launch.py    # terminal 1
python3 ros/dronions_ros_node.py                 # terminal 2
```

### PX4 quadrotor (full flight stack)

PX4 SITL + MAVROS + Gazebo, four terminals — see the handover document. Terminal
one is wrapped in a helper because Gazebo needs to be forced onto the discrete
GPU on this machine:

```bash
./scripts/run_px4_sim.sh
```

The drone takes off, sweeps the area in a bounded lawnmower pattern, avoids
obstacles by lidar, centres on the target once confirmed, and approaches it.

## Layout

```
assistant/     Gemini agent, command parsing, speech in/out
perception/    YOLO-World detector, candidate filtering, tracker
navigation/    steering/throttle from a bounding box (shared, actuation-agnostic)
ui/            on-screen overlay
ros/           ROS 2 nodes, launch files, ground-rig world
scripts/       run helpers
memory/        Visual Memory Bank reference photos
```

`navigation/navigator.py` is deliberately actuation-agnostic — it emits
`turn`/`throttle`, and each node maps that onto its own vehicle. That is what let
the PX4 port reuse it unchanged.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
```

Gemini's free tier allows **20 requests/day per project per model**. A new API
key in the same project does *not* reset it; switching `GEMINI_MODEL` does, since
each model has its own allowance.
