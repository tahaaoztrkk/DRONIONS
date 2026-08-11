# Plan: assistive scenarios, grounded in the CHI '26 reference work

Scope for now: **one drone, no wrist dock**. The wearable docking station stays
in the concept, not in the prototype.

---

## 1. What the reference paper actually built

*Towards LLM-powered Assistive Drone for Blind and Low Vision Users* — Wei et
al., CHI '26 (NUS / Augmented Human Lab). Formative study N=9 → participatory
iteration with 3 BLV users + 5 experts → evaluation N=6.
Code: `github.com/iamyize/chi26-llm-drone`.

**Pipeline:** Flic button → Google STT → GPT-4o generates Python from a
**high-level function library** → `exec()` on a Tello EDU → vision tasks handled
by GPT-4o multimodal → Google TTS back to the user.

**The decision that matters most for us:** for safety (DG3) the drone
**only moves between pre-mapped waypoints** — `origin_to_table()`,
`origin_to_shelf()`, `table_to_shelf()`. "Searching" means visiting known
waypoints, capturing an image at each, and asking the VLM which location most
likely holds the object. There is no autonomous exploration, deliberately: it
prevents the LLM from generating unsafe flight.

**Their function library** (Appendix B) is worth copying as a design, not
reinventing:

```
take_off, land, report_status
origin_to_table / table_to_origin / origin_to_shelf / shelf_to_origin / table_to_shelf
detect_objects(command)   find_item(item)   read()
describe_color(item)      count(item)       ask_follow_up(command)
where_am_i()              where_is_exit()
```

**Their three evaluation tasks**, and the scores:

| Task | Score /5 |
|---|---|
| Object localization — "find my cup in this room" | 4.0 |
| Object recognition — "identify the important items on the table" | 3.8 |
| **Spatial orientation — "find the exit door"** | **3.2** |

SUS 73.3 (SD 14.9 — high variance; one participant rated it poor).

**Error budget:** STT 5.63%, code generation 7.04%, recognition 6.35%,
partially-correct answers 6.35%. Latency: 1.46 s code gen, 3.18 s vision.

**Design findings we should treat as settled, not re-derive:**

- Wake words were rejected; a **tactile button** is preferred (deliberate, no
  false triggers). P6 wanted it especially outdoors.
- A **confirmation step** before execution ("To confirm, you would like me to
  find your bottle") was added after transcription errors caused task failures.
- **Progressive narration** is valued: P6 — *"while the drone is locating, it
  tells me where it is looking. So at least I have an orientation, which I find
  very, very useful."*
- **Follow-up questions were 35% of all queries.** Multi-turn with memory is not
  a nice-to-have.
- **Less audio, not more.** Hearing is a safety channel for BLV people; v3
  limited response length deliberately.
- Multi-threading so API calls overlap drone movement.

---

## 2. Where we can contribute something they could not

Their own limitations section names the gaps, and our setup happens to address
several of them.

**We have ground truth; they had a lab and a Likert scale.** Their spatial
orientation task scored 3.2 and they could only measure that subjectively. In
Gazebo the true pose of the drone, the user and every object is known, so
"your keys are two metres ahead, slightly right" can be scored **numerically**
against truth — distance error in metres, bearing error in degrees. This is the
single clearest thing we can add.

**Their weakest result has a nameable cause we can remove.** They attribute the
3.2 to LLMs being poor at spatial reasoning, *"further compounded in our context,
where the drone may be oriented differently from the user (e.g., the drone's
camera is opposite to the user)"*. Our system already knows the drone's pose from
`/mavros/local_position/pose`. The relation between target, drone and user is
**geometry, not language** — compute it, and leave the LLM only the job of
phrasing it. That is a concrete, testable improvement over the published system.

**We can afford autonomous search; they could not.** Waypoint-only flight was a
safety workaround for a real drone in a room with people. In simulation, crashing
costs nothing. The lawnmower search, lidar avoidance and climb-over already built
here explore the axis they deliberately closed off — worth keeping, but as *our*
contribution, not as a reimplementation of theirs.

**We can break their controlled setup on purpose.** They note objects were
"primarily placed by the researchers" and that real scenes have occlusion and
moving objects. Occluders, clutter and a moving person are cheap to add in
simulation and directly target their stated limitation.

---

## 3. Architecture changes

### 3.1 Adopt the function-library + code-generation pattern

Replace today's single-target flow (`target string → YOLO prompts → Gemini
yes/no`) with a small function library the LLM composes against. Ours differs in
one respect: because we can search autonomously, `find_item()` need not be
limited to pre-mapped waypoints.

Keep the library small and rigid. Free-form LLM output driving a flying vehicle
is exactly what their DG3 was protecting against, and their measured 7.04% code
generation error rate is the reason.

### 3.2 Ground the spatial description in geometry, not the LLM

This is the core technical idea. Three frames must be reconciled:

- object pose in the world (from detection + drone pose + camera geometry)
- drone pose (known)
- **user pose and facing** (must exist in the scene)

From these, compute distance and bearing **relative to the user's facing**, then
hand the LLM only: the target, the computed relation, and the visual context it
alone can supply ("next to a blue mug"). The LLM writes the sentence; it does not
do the geometry.

Expected output: *"Your keys are about two metres ahead and slightly to your
right, on the table, next to a blue mug."*

### 3.3 Interaction loop, following their proven design — **built and verified**

Implemented in `assistant/dialogue.py`, wired into the PX4 node. Every path
below has fired in flight, not only in unit tests:

| Their finding | What it became | Verified |
|---|---|---|
| confirm before acting | `Onaylayın: box aramamı istiyorsunuz` | yes |
| — | correction attached to the rejection (`no,phone`) | yes |
| narrate progress | rate-limited; repeats held to 30 s, new information to 8 s | yes |
| multi-turn follow-ups | answered from the frame kept at the moment of the result | yes |
| concise speech | replies capped at two short sentences | yes |
| bounded "I could not find it" | deadline derived from the sweep's own lap time | fired at 317 s |
| *(theirs did not need this)* | "I cannot see" as a **distinct** answer from "it is not there" | fired on quota exhaustion |

Two design decisions worth carrying into the write-up.

**The give-up deadline is computed, not chosen.** A fixed 180 s was shorter
than one lap of the search area, so the drone reported ground empty that it had
never flown over — an assistive system asserting a negative it had not earned.
`SearchPattern.sweep_seconds()` predicted 198 s for a lap; the logged position
trace completed one in 195 s, and the deadline is that estimate × 1.6.

**A dead sensor is not a negative result.** Their drone flew a fixed waypoint
list under supervision; ours searches autonomously, so an API failure counted as
"not this one" produces minutes of confident searching by a system that cannot
see, ending in "not found". A run was lost exactly this way to a retired-model
404. Three consecutive failures now stop the search and name the cause — quota,
unreachable model, or network — because the user has no independent way to tell
a blind drone from an empty room.

Both are consequences of autonomy that their supervised, waypoint-based design
never had to answer for. They are the clearest thing this work adds to the
interaction findings rather than merely reproducing.

---

## 4. Scenario set

Deliberately aligned with their three tasks so results are comparable, then
extended where we can go further.

| # | Scenario | Mirrors | What it adds |
|---|---|---|---|
| S1 | "Find my cup in this room" | their Object Localization (4.0) | ground-truth distance/bearing error |
| S2 | "What's on the table?" | their Object Recognition (3.8) | precision/recall against known contents |
| S3 | "Where is the exit?" | their **Spatial Orientation (3.2)** | **geometric grounding vs LLM reasoning — the headline comparison** |
| S4 | "Read the label / what number is that?" | `read()` | OCR at range; needs textured text assets |
| S5 | Occluded and cluttered variants of S1 | — | their named limitation |
| S6 | Moving obstacle (a person crossing) | — | their named limitation; also exercises avoidance |

S3 is the one to build first after the infrastructure, because it is where the
published system is weakest and where our approach has a principled advantage.

---

## 5. Measurement

What we can measure objectively, which they could not:

- **Description accuracy** — |stated distance − true distance|, |stated bearing −
  true bearing|. Report distributions, not averages alone.
- **Frame-mismatch sensitivity** — sweep the angle between drone facing and user
  facing; show error growth for LLM-only description and flatness for the
  geometric one. This is the experiment that makes the point.
- **False-confidence rate** — how often does it confidently state a wrong
  location? Weight this heavily; their error table shows recognition errors put
  "the stuffed toy at the table" when it was on the shelf.
- **Task success and time**, per scenario.
- Latency, to compare against their 1.46 s / 3.18 s.

Subjective measures (SUS, Likert) need people and belong to a later study, but
the scenarios should be built so that such a study is possible without rework.

---

## 5b. Result: the frame-mismatch experiment (pilot, n=6)

Run with `scripts/experiment_frame_mismatch.py`; raw data in
`logs/frame_mismatch.csv`. The target and user stay fixed while the drone views
the target from six bearings, so the angle between where the drone looks and
where the user faces sweeps -180 to +135 degrees. Three ways of answering
"where is it, from the user's point of view?" scored against ground truth:

| method | median distance err | median bearing err |
|---|---|---|
| **geometric** (project detection, convert to user frame) | **0.49 m** | **13.2°** |
| llm naive (image only -- what a camera-only pipeline has) | 0.53 m | 29.7° |
| llm posed (image + drone/user poses stated in words) | 0.73 m | 45.3° |

Three things this shows, in order of how much they surprised us:

1. **The LLM's bearing answers are close to uninformative.** Truth was +29.7°.
   Three of six naive answers had exactly 29.7° error -- i.e. the model replied
   "0 degrees, straight ahead". The rest clustered at ~15°. It is not tracking
   the geometry; it is emitting a plausible small number.
2. **Telling the LLM the poses made it worse** (45.3° vs 29.7°), including one
   131.9° answer. Handed the coordinates, it attempts the arithmetic and gets it
   wrong; left naive, it at least guesses harmlessly. This was meant to be the
   *strong* baseline and it came last.
3. **Distance is not the problem** -- all three methods land within about half a
   metre. The gap the reference work identified is specifically about direction,
   which is exactly what geometry fixes and language does not.

Honesty about our own arm: the geometric estimator is roughly **unbiased** (mean
signed error +2.7° bearing, -0.16 m distance) but **noisy** (signed errors span
-27° to +31°). The scatter tracks the detection box moving on the object, not a
correctable offset in the maths. At 4 m, 13° is about 0.9 m sideways -- enough to
matter for "slightly to your right" versus "to your right".

**n=6 makes this a pilot, not a result.** A publishable version needs more
viewpoints, several targets, repeated trials per viewpoint, and signed-error
distributions rather than medians.

---

## 5c. Scaling the pilot: what the measurement actually cost

Turning the n=6 pilot into a defensible number exposed four flaws in the *rig*,
not in the system. Each looked like a result until it was chased down, so they
are recorded here to stop anyone repeating them.

**The commanded pose is not the pose.** Teleporting an unarmed airframe to
altitude means it starts falling immediately; by the time a frame arrives it is
no longer where it was put. Projecting with the requested pose corrupted every
estimate, and results drifted run to run as the simulator slowed.

**MAVROS's pose is not ground truth either.** `/mavros/local_position/pose` is
PX4's EKF estimate, and an EKF cannot track an instantaneous teleport. Switching
to it more than tripled the measured bearing error (5.6° → 21.9°).

**Gazebo's own pose is ground truth, but `gz model -p` costs 5.1 s a call.** It
reloads world state each time and starved the ROS executor so badly that 65% of
trials never received a camera frame. A bridged pose topic is the right answer;
the bridged `TFMessage` never reached the subscriber here, and the topic is
`pose/info`, not the `dynamic_pose/info` first assumed.

**The simulator degrades under this abuse.** Across otherwise identical runs the
no-detection rate climbed 7% → 10% → 20%, and eventually `gz` stopped answering
service calls entirely. Hundreds of teleports are not a workload Gazebo is happy
with.

**With correct ground truth (n=21 usable):**

| | median | p90 |
|---|---|---|
| bearing error | **2.8°** | 11.7° |
| distance error | **0.49 m** | 0.89 m |
| position error | 0.59 m | 1.08 m |

Altitude drift between teleport and capture was ~0 (median 0.04 m), confirming
the reference pose was finally right. This is the number to quote, with the
caveat that n=21 is still small.

**Recommendation for the real sweep: stop teleporting.** Fly the drone normally
and record measurements during flight. PX4 then controls the vehicle, the EKF is
valid, MAVROS's pose can be used directly, and none of the four problems above
exists. Covering viewpoints takes longer in wall-clock terms but the setup is
sound, and it measures the system as it actually operates.

---

## 5d. Result: how much work the VLM stage is actually doing (n=80 viewpoints)

The system pairs a cheap detector with an expensive model, and the reason given
for that has always been "YOLO alone confuses the wall for the box". True, but
asserted rather than measured — and the two end-to-end failures on record were
both the Gemini API, so search reliability and system reliability could not be
told apart at all.

`scripts/experiment_repeatability.py` runs the node in survey mode: the VLM is
skipped, the drone flies the full sweep without ever handing off, and every
ranked detection is logged with its world position and scored against the
scenario's static object positions. Five runs, 80 viewpoints, 106 detections,
no API quota spent.

| Top-ranked detection is… | share of viewpoints |
|---|---|
| the **wall** | **71%** (57/80) |
| the real box | 12% (10/80) |
| the blue distractor | 10% (8/80) |
| the sphere | 6% (5/80) |

So a detector-only system flies at the wall in roughly seven viewpoints out of
ten. That is the headline, but the more useful number is next to it.

**The box appears anywhere in the detector's candidates in only 14% of
viewpoints (11/80)** — barely above the 12% where it is ranked first. In other
words, when YOLO sees the box it almost always ranks it top (10 of 11); the
failure is not ranking, it is recall. And with a mean of **1.3 candidates per
viewpoint**, the question actually reaching the VLM is usually not "which of
these is yours" but "is this one thing yours" — to which the answer is
overwhelmingly no.

Three consequences worth carrying into the design:

- **The VLM's job is rejection, not selection.** It cannot rescue a viewpoint
  where the detector produced no box, because it only ever sees the detector's
  crops. Its value is refusing the wall 86% of the time.
- **Per-viewpoint recall bounds the whole cascade**, and it is 14%. The system
  works because the sweep supplies ~16 viewpoints per run and needs only one
  good one — redundancy, not per-frame reliability. That is also why the
  give-up deadline has to cover a full lap.
- **Raising the number of crops sent to the VLM would buy almost nothing.**
  The gap between "box ranked first" and "box detected at all" is two
  percentage points. Effort belongs on detection recall.

### What the campaign then paid for

The obvious reading of "the wall wins 71% of viewpoints" is that the VLM is
carrying the system, and the fix is a better VLM. The data says something
cheaper. Every detection's **physical size** can be computed before any model
is called, from the same ground-plane range already used to locate the target,
and the two classes separate cleanly:

| | implied width, measured | true |
|---|---|---|
| box | 0.53 – 1.12 m (median 0.61) | 0.63 m |
| wall | 1.26 – 4.90 m (median 3.70) | 4.0 m |

An upper bound of 2.5× the target's expected width sits inside that gap: over
the 106 recorded detections it rejects **69 of 69 wall hits** while keeping
**81% of real box hits**. That is now `size_plausible()` in the search loop.

This is the filter the negative prompts were supposed to be and never were.
`"wall"` as a negative class cannot fire, because YOLO-World labels the
scenario wall *"cardboard box"* — the label is wrong, so label-based exclusion
has nothing to work with. Size does not care what the detector calls a thing,
which is also why it should generalise: in any real room something larger than
the target will be in view, and rejecting it should not depend on the detector
naming it correctly.

Two side effects worth noting. It saves API calls, since a viewpoint left with
no plausible candidate never reaches the model at all — on these numbers that
is most viewpoints. And it does not run in survey mode, so the raw detector
behaviour stays measurable and the threshold can be re-checked against future
campaigns; the implied width is logged per detection for exactly that.

Not yet flight-tested: the threshold comes from recorded data, not from a run
made with the gate active.

Secondary observations from the same runs: the box was seen at least once in
4 of 5 runs (median 50 s, range 39–86 s), and one run never saw it at all. The
wall is *more* confident than the box (median 0.233 vs 0.183) and 42× larger in
frame (0.273 vs 0.0065), which is why confidence-only ranking puts it first and
why the size gate at hand-off matters. The unexplained 19 m excursion did not
recur in any of the five runs.

---

## 6. Environment

- **Indoor room, human scale.** Ceiling 2.4 m — today's `HOVER_ALTITUDE = 2.0`
  and the 5.5 m climb-over are both illegal indoors; cruise drops to ~1.2 m and
  the climb must be ceiling-clamped.
- **Doorways ~0.8 m, and the vehicle does not fit through them in any useful
  sense.** This was previously written down as an avoidance-threshold problem
  (`OBSTACLE_SAFE_DISTANCE = 1.0 m` being wider than a doorway), which turned
  out to be wrong when it was finally measured rather than assumed. Three
  things are actually in the way:

  1. **The airframe is 0.68 m wide.** x500_base puts the rotors at ±0.174 m and
     the props are 13 inch, so the swept half-width is 0.339 m. A 0.8 m doorway
     leaves **±6 cm** of lateral tolerance, which a sweep flying at 0.25 m/s on
     yaw-tracked waypoints cannot hold.
  2. **The lidar cannot see the frame in time.** The fan is ±0.35 rad, spanning
     only ±0.33 m at 0.9 m range, so the door edges fall outside it until the
     drone is already too close to turn. Simulated off-centre approaches at
     0.5 m read as clear while the airframe would strike the frame.
  3. **Replacing `min(ranges)` with a lateral-corridor test changes nothing**
     at this fan width — every return close enough to matter is already inside
     the corridor. The test is implemented and correct, and becomes useful only
     once the fan is widened.

  The honest reading is that this is not a tuning problem. The concept is a
  *miniature* drone launched from the wrist; x500 is a 500-class quadrotor,
  roughly ten times that. Options are to model wider (1.0–1.2 m) doorways and
  document the mismatch, to build a scaled-down airframe and retune PX4 for it,
  or to restrict indoor scenarios to open-plan space. Choosing the vehicle is
  prior to any avoidance work.
- **A user model** with position and facing, plus named locations (table, shelf,
  door) so a waypoint-style mode is also possible for comparison.
- **Textured meshes only.** Measured here: keys (7 cm), phone (15 cm), mug
  (10 cm) **and a 45 cm plain-coloured box** all produced *zero* YOLO detections
  at 2.2 m at both 640 and 1280 inference resolution, while the textured
  `cardboard_box` mesh detects reliably at the same range. The frame capture
  confirms the box was clearly visible and well framed — the discriminator was
  **appearance detail, not size**. A scene of coloured primitives would fail
  perception for reasons that do not exist in the real world. Use Gazebo Fuel or
  CC0 meshes.

### Small objects

Their evaluation used a cup, table items and a door — all comfortably large.
Keys came up only as an aspiration (P6: *"locating my cane or my keys or my
mug"*). Geometry at 60° FOV:

Corrected: the camera's horizontal FOV is **1.74 rad (99.7°)**, read from
`mono_cam/model.sdf` -- an earlier draft of this plan assumed 60°, which was
optimistic. A wider lens makes every object smaller:

| Range | Keys (7 cm) @1280 | @640 |
|---|---|---|
| 1.0 m | ~38 px | ~19 px |
| 2.0 m | ~19 px | ~10 px |

`IMAGE_SIZE = 640` halves an already marginal target. Cheapest-first: raise
`IMAGE_SIZE`; cruise lower for small targets; two-stage look (sweep, then
descend and re-examine — the CENTER phase is the natural place); tiled inference;
and only as a last resort let the VLM screen instead of YOLO.

Settle it on **one textured table** before building the room. The harness exists
(spawn → teleport → capture → detect; no Gemini, no quota).

---

## 7. Order of work

1. **User in the scene + geometric spatial description.** Turns the existing
   pipeline assistive and unlocks the S3 comparison. Highest value per effort.
2. **Interaction loop**: confirm-before-execute, progress narration, multi-turn
   memory, bounded give-up.
3. **Function library + LLM planner**, replacing the single-target flow.
4. **Textured room**, retuned for ceiling, cruise height and doorways.
5. **S3 frame-mismatch experiment** — the headline measurement.
6. Small-object work (§6) if keys-class targets are in scope.
7. Occlusion / moving-obstacle variants.

---

## 8. Still open from the simulation work

- **Repeatability** — better than it was: several end-to-end successes, fastest
  43 s from command to answer. But the two failures on record were both the
  Gemini API rather than the search (a retired-model 404, and daily quota), so
  the *system's* reliability and the *search's* reliability are still
  confounded. Script N headless runs, exercising the non-Gemini parts (search
  coverage, obstacle clearing, climb-over) without limit and reserving quota
  for identification.
- **One unexplained excursion.** A run put the drone at (18.9, −12.5) —
  nineteen metres outside an area 6.5 m across — and it recovered on its own.
  There was no position trace to diagnose it from. The sweep now logs its
  position every 10 s and turns back at the boundary, so a recurrence will
  leave evidence; until it recurs, the cause is unknown. Candidates worth
  checking first: a yaw-sign disagreement between MAVROS pose and the waypoint
  controller, or EKF disturbance around the `TM: Time jump detected` warnings
  MAVROS emits under simulation-time changes.
- **Gemini quota shapes what can be tested in a day.** 20 requests/day per
  project *per model*; a new key in the same project does not reset it, but
  switching `GEMINI_MODEL` does. A 429 is not always the daily cap — there is a
  per-minute limit that recovers within seconds, which is why only *consecutive*
  failures are treated as fatal.
- **Indoor scenarios are blocked by the vehicle, not the threshold.** The
  0.8 m-doorway problem was recorded here as an avoidance-tuning issue and is
  not one: the airframe is 0.68 m wide, leaving ±6 cm of tolerance, and the
  lidar fan cannot see the frame in time regardless. See §6. Picking the
  vehicle comes before any further avoidance work.
- **Untextured primitives are invisible to YOLO-World** (§6, Small objects), so
  the realistic indoor scene needs textured assets before any "can it find keys"
  result means anything. The node now logs each target's prompt expansion and
  warns when there is none, which is the cheapest way to find out which objects
  still need entries in `PROMPT_DATABASE`.
