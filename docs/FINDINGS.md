# DRONIONS — Measured Findings

This file records results that were **measured**, not expected. Every entry
rests on a number and says where that number came from. Guesses and hypotheses
do not belong here — refuted hypotheses have their own section, because what
was tried and did not hold is also a result.

**Scope caveat, applying to every number below:** all measurements were taken
in one room, under one lighting setup, rendered by Gazebo. They say what a
given change is worth in simulation. They say nothing about a real room.

---

## 1. Detector: open vocabulary against a task-specific model

Measured at the viewpoints the search actually flies from, on the same frames,
scored by the same rule (`scripts/compare_detectors.py`,
`logs/detector_comparison.csv`). A detection counts only if its box covers
where the object truly is, at roughly the width geometry predicts.

| object | visible | YOLO-World | fine-tuned YOLOv8n |
|---|---|---|---|
| laptop | 8 | 6 | 8 |
| book | 8 | 2 | 8 |
| mug | 8 | 6 | 8 |
| phone | 8 | 1 | 8 |
| box | 4 | 4 | 4 |
| **total** | **36** | **19 (53%)** | **36 (100%)** |

**What it means:** the gates, the vision-language model and the tracker only
ever see what the detector already found. No amount of work behind a detector
that finds the phone in one viewpoint of eight can move that number. Weeks of
gate tuning were bounded by this.

**The two confidences are not comparable.** Across correct detections the open
model sits at a median of 0.64 and the trained one at 0.37. The trained model
is less certain but more often right — drawing a tight box on a small object is
a harder thing to be certain about. Merged raw, the open model would win every
ranking for a reason unconnected to being right, so `perception/hybrid.py`
divides each confidence by its own median before ranking.

**This is a trade, not a winner.** The trained model knows only the five
objects it was shown; the project's premise is that a user asks in their own
words. The hybrid runs both: the open model answers everything, the trained
model is authoritative for its own classes.

---

## 2. Camera rays must be cast from the lens, not from the airframe origin

The camera is mounted 0.12 m forward and 0.242 m above `base_link`
(`px4/models/x500_dronions/model.sdf`). Every projection in `spatial.py` was
casting its ray from `base_link` instead.

The error is a lever arm, so it grows as the object gets closer. Deviation from
the true direction, over 24 detections:

| | median | at 0.45 m |
|---|---|---|
| from the airframe origin | 15.0° | 43° |
| from the lens | **2.9°** | **7.7°** |

Localisation error over 26 detections: **0.24 m → 0.15 m** (median), with 18 of
the 26 improving.

**Side effect:** the plane projection was reading 0.61–0.81 times the true range
at every viewpoint. That was read as "the geometry breaks down on descent" and
is why `RANGE_TRUST_RATIO` exists. The geometry was sound; it was being cast
from the wrong point.

---

## 3. The vehicle's heading estimate is not its heading

MAVROS reports a yaw that differs from the true orientation while its position
is correct (`scripts/measure_heading_bias.py`). The ray leaves the right point
in the wrong direction, so the error is lateral and grows with range.

- Bias measured across five starts: **−4.16, −4.18, −4.99, −6.13, −5.98
  degrees**. Each is a fresh EKF alignment, so it cannot be hard-coded.
- Within one flight: −5.98 just before takeoff, −3.90 eight minutes later, and
  only 0.12° of drift over the two minutes after that. The shift happens around
  takeoff and then slows.
- The right moment to measure is therefore **immediately before the flight**;
  the target is localised about a minute after takeoff.

Same object, same start position, four runs:

| run | estimate | error |
|---|---|---|
| uncorrected | (3.00, −0.58) | 0.363 m |
| uncorrected | (2.89, −0.63) | 0.440 m |
| fixed −4.18° | (2.99, −0.38) | 0.171 m |
| **per-run −5.98°** | **(3.08, −0.30)** | **0.085 m** |

True book position (3.05, −0.22).

**The honest pair of numbers:** the perception pipeline's own accuracy is
**~0.09 m**; accuracy including the vehicle's heading estimate is **~0.40 m**.
On real hardware the gap between them is what magnetometer calibration closes.
The measurement rig reads the bias from ground truth, so it is an experimental
control, not part of the system.

---

## 4. Range policy: plane, apparent size, and the hybrid

35 valid detections, with the corrected measurement rig.

| policy | median | mean | worst | >0.5 m | \|dz\| |
|---|---|---|---|---|---|
| always plane | 0.106 m | 0.275 m | 3.766 m | 4 | 0.06 m |
| always size | 0.145 m | 0.167 m | 0.633 m | 1 | 0.12 m |
| **hybrid (current, threshold 2.0)** | **0.106 m** | **0.155 m** | 0.676 m | **1** | **0.06 m** |
| hybrid, threshold 1.5 | 0.123 m | 0.158 m | 0.676 m | 1 | 0.06 m |
| hybrid, threshold 3.0 | 0.106 m | 0.267 m | 3.766 m | 3 | 0.06 m |

**Conclusion: no change needed.** The current rule takes the plane's median and
its height accuracy, and the size estimate's resistance to outliers. The plane
alone fails catastrophically at times (4.5 m); size degrades more gracefully but
is twice as bad on height.

---

## 5. The colour gate works only when the reference is the object's own colour

Over 16 detections, counting the target's own object and the confusers
separately.

| target | own object | confusers | state |
|---|---|---|---|
| book | 4/4 | **2/12** | works |
| laptop | 4/4 | 12/12 | structurally dead |
| mug | 3/4 | 8/12 | weak; also rejects its own object |
| phone | 4/4 | 7/12 | weak, but rejects the book 4/4 |

**Mechanism:** the gate ORs over the references. A **single** reference that
captured the wooden table opens the whole set to everything on that table. Two
of the book's four old references were wood (hue 13 and 26); regenerated, the
set became blue cover throughout (220–223) and the gate went from 12/12 to
2/12.

**The laptop is structurally unsolvable:** its screen blue is the book cover's
blue. Colour cannot separate those two, however clean the reference.

**Tried and abandoned:** cropping the reference to the object's own pixels using
the segmentation mask. Measured, and worse. The black phone's own saturated
pixels read hue 37 against a table at 28 — from the drone the phone already
looks wood-coloured, and the gate compares two unmasked drone views. Masking
made the phone's gate stop rejecting the book (4/4 → 0/4) and dropped the
book's own gate from 2/12 to 12/12.

---

## 6. End-to-end repeatability, all runs from the origin

Four targets, two real flights each.

- **8/8** runs handed off and reached the target.
- **7/8** reached the right object. The one deviation was a phone run whose
  estimate drifted 0.73 m and landed nearer the bottle than the phone.
- Zero crashes, zero premature-arrival rejections, zero quota events.
- Time from command to arrival: laptop 23 s, mug 18–23 s, book 26 s.

**Limit of this measurement:** every run approached from the same place,
because the trained detector finds the target from the takeoff spot and the
search barely runs. These numbers say "finding and approaching work"; they do
not say "they work from every geometry". Section 11 addresses that.

---

## 7. Localisation resolution against object spacing

The phone (2.62, −0.16) and the book (3.05, −0.22) are **0.44 m** apart.
Measured localisation error is 0.11–0.23 m. The error therefore approaches half
the spacing and **can flip the label**: one run estimated (2.85, −0.18) — 0.23 m
from the phone, 0.20 m from the book. The drone did not go to the book, but the
scoring counted it as the book.

This is the numerical answer to "how confidently can we say which object it went
to", and it is this pipeline's present limit.

---

## 8. The Gemini free-tier quota corrupts measurements

The limit is **20 calls per day**, per project per model. When it is exhausted
the SDK retries a 429 internally with exponential backoff and returns only once
it finally succeeds.

**Measured:** on the 19th call of the day, a single search call took **4 minutes
46 seconds**, and the drone hung motionless for all of it — having detected the
target in the first second. Nothing in the log said so.

The same flight on a fresh quota: the call took **11.2 seconds**, the whole
flight **26 seconds**.

**Consequence:** no timing measurement taken while the quota is exhausted is
valid. Calls are now bounded to a 15 s timeout with one retry, 429s are logged,
and every call's duration is recorded.

---

## 9. What the open vocabulary retains: furniture, not small objects

YOLO-World's recall on eight objects that appear **nowhere** in our training
data (`scripts/measure_open_vocab.py`, `logs/open_vocab.csv`). 94 of 96 samples
usable.

| unseen | | | unseen | |
|---|---|---|---|---|
| chair | 5/5 | | bowl | **0/8** |
| table | 8/8 | | headphones | **0/5** |
| bookshelf | 8/8 | | bottle | **0/4** |
| sofa | 4/5 | | | |
| cabinet | 5/8 | | **total** | **30/51 (59%)** |

The split is sharp: **it finds furniture almost perfectly and small objects not
at all.**

**Why this matters for the fine-tuning decision:** what the open vocabulary
retains here is furniture, and furniture is what the surface scan depends on —
the ability to work out whether the target is on a table or on a sofa comes from
there. If fine-tuning YOLO-World costs that, it costs not only "find my charger"
but the **search's surface logic**. Any acceptance criterion has to cover it.

**Measurement note:** the same script scores 25% (8/32) on the four trained
classes, whereas `compare_detectors` scores 53% (19/36). They do not measure the
same thing: the flying system uses a curated prompt list plus negative prompts,
while this script uses bare class names — because there is no curated list for
the eight unseen objects and both groups must be scored by one rule. This
number is not the flying system's performance; it is a consistent baseline for
the before/after comparison.

---

## 10. Fine-tuning YOLO-World destroys the open vocabulary completely

Supervisor's proposal: instead of a separate closed-set model, fine-tune
YOLO-World itself on our room data. All four suggested mitigations were applied
— `freeze=10` (backbone frozen), `lr0=0.001`, a 40-epoch cap, and `patience=10`
early stopping. Training stopped at 22 epochs: 20 minutes, peak GPU 3.39 GiB,
960 px.

| | before fine-tuning | after fine-tuning |
|---|---|---|
| **eight unseen objects** | 30/51 (59%) | **0/42 (0%)** |
| four trained objects | 8/32 (25%) | 28/28 (100%) |

The open vocabulary did not weaken, it **vanished**. Not one detection across 42
samples. `table` at 8/8 in the baseline, `bookshelf` at 8/8, `chair` at 5/5 —
all zero.

**The measurement is sound:** the same `set_classes` path scores 28/28 on the
four trained classes, so the mechanism works; the model simply no longer
responds to words it was not shown.

**Why the mitigations did not help:** YOLO-World's open vocabulary does not live
in the backbone. The alignment between the text embeddings (`txt_feats`,
1×80×512) and the image happens in modules 21 (`C2fAttn`) and 22
(`WorldDetect`) — exactly the two modules `freeze=10` leaves trainable. Freezing
the backbone protects general visual features; it does not protect the
vocabulary.

**The gain side adds nothing:** the fine-tuned model scores 100% on its own four
objects, but the separately trained YOLOv8n already scores 36/36 (see section
1). The only thing the trade buys is carrying one model instead of two; what it
costs is the property the project is built on, and the search's surface logic.

**Decision: the hybrid stays.** The upstream record points the same way
(ultralytics#10038): there too a fine-tuned YOLO-World both lost its zero-shot
ability and performed worse on its own classes than a fine-tuned plain YOLOv8.

---

## 11. Repeatability from varied geometries

Three targets, six real flights each, every flight with a different warm-up —
identification is held back so that the first sighting happens wherever the
sweep has carried the drone (`DRONIONS_SEARCH_WARMUP`).

| target | arrived | right object | median error | worst | diversity |
|---|---|---|---|---|---|
| phone | 6/6 | 6/6 | 0.10 m | 0.12 m | good (2 of 6 at one point) |
| laptop | 5/6 | 5/5 | 0.16 m | 0.21 m | weak (3 of 5 at one point) |
| book | 4/6 | 4/4 | 0.13 m | 0.27 m | weak (3 of 4 at one point) |
| **total** | **15/18** | **15/15** | **0.13 m** | | |

**Not one run went to the wrong object.** None of the three failures was the
system's: two were Gemini answering `504 DEADLINE_EXCEEDED` (service
congestion), one was the run on which the daily quota ran out.

Across the phone's six runs the first-sighting points spread around the room:
bearings from −69° to +69°, ranges 0.60–2.61 m. Again no relationship between
error and viewing angle — the third independent measurement of that result.

**Whether the warm-up produces diversity depends on the object.** It worked for
the phone and not for the laptop or the book: in both of those the first
sighting happened at 2.9–3.4 m, meaning the object is already visible from a
distance the moment the warm-up ends, and the drone hands off before it has
moved. The mechanism works but is not sufficient on its own.

**The mug could not be measured, in two attempts.** The first time all six runs
fell to the daily quota running out (12 × 429). Repeated the next day on a fresh
quota, it was again 0/6, this time mostly on the service side: **16 × 503
UNAVAILABLE** ("high demand"), 6 × 504, 14 × 429.

Neither attempt says anything about the system — the measurement simply never
happened. It is recorded here because it explains why the four-object table
stands at three, and because a measurable cost of depending on an external
service is a finding in itself.

---

## 12. What the quota gate bought

Asking the vision-language model only when the trained detector has found the
target class in the frame, measured:

| | before | after |
|---|---|---|
| calls per run | 4.5 | **1.7** |
| a six-run campaign | 27 calls (limit 20 — does not fit) | **10 calls** |

Campaigns now fit inside one day. One run skipped 10 frames, so the gate is
genuinely firing, and detection still records honestly what was visible.

---

## 13. The open vocabulary works in flight, on an object never trained

Demonstrated end to end with a target the trained detector has never seen and
that has no entry in the prompt database: a chair.

```
11:57:41  "A chair is not an object I know. I will try to find it,
           but I have no reference to compare against."
11:57:44  Model reply: [NONE]  x4          (still searching)
11:58:53  Target confirmed. Now tracking.
11:58:55  Target position (world) x=1.55 y=0.71
          ^ source=world label=chair confidence=0.888  293x303 px
11:58:56  ARRIVED: chair at altitude 1.58 m
```

True chair position (1.50, 0.90) — **0.20 m error, 75 seconds** from command to
arrival.

The decisive field is `source=world`: the candidate came from the **open**
model, not the trained one. Section 10 measured that a fine-tuned YOLO-World
scores 0/42 on unseen objects, so the same flight with that model would have
found nothing. This single run is the live counterpart of that measurement.

**The target was chosen from section 9, not by guesswork.** The open model
finds furniture almost perfectly (chair 5/5, table 8/8, bookshelf 8/8) and small
unseen objects not at all (bowl 0/8, bottle 0/4). Asking for a bottle would have
failed, and failed for a measured reason rather than bad luck.

**Also confirmed in this flight:** the spoken output and the console rendering
are in English while the log file stays Turkish, and the honest "I do not know
this object" message is what a user hears before the system tries anyway.

---

## 14. All four targets in one interactive session

The demo path, run from the keyboard with speech on, asking for each object in
turn without restarting.

| target | estimate | error | distance at arrival |
|---|---|---|---|
| laptop | (3.00, 0.23) | 0.05 m | 0.77 m |
| book | (2.97, −0.25) | 0.09 m | 0.69 m |
| phone | (2.63, −0.29) | 0.13 m | 0.76 m |
| mug | (2.61, 0.51) | 0.27 m | 0.75 m |

**4/4 found, tracked and reached.** Median error 0.13 m, matching the campaign
figure from separate flights (section 11).

**The climb gate fired twice**, moving from one object to the next — the exact
case it was added for. Asking for a target while the drone is still climbing
back to search altitude after arriving at something else had produced a correct
identification the centring could not use: at 1.4 m altitude and 0.34 m range
the phone sits at y=0.89 in the frame, and by the time the climb finished the
sweep had turned away. Both transitions here succeeded.

This is also the first end-to-end confirmation of the demo path as a whole:
English speech through Piper, English console rendering over a Turkish log
file, keyboard commands with confirmation, and consecutive targets in one
session.

**The failure path was rehearsed too**, by disconnecting the network mid-run —
which produces the same message as a 503 or 504 from the service:

```
11:29:20  [Errno -3] name resolution failure
11:31:44  "I cannot reach the vision service at the moment, so I cannot see.
           I am stopping the search; you can try again shortly."
11:32:55  ARRIVED: book at altitude 1.61 m, 0.77 m      (network restored)
```

The drone states the fault and stops, rather than hanging silently, and the
next command works once the service returns. This matters for a system whose
user cannot see the screen: an unexplained silence and a failure are
indistinguishable to them.

---

## Refuted hypotheses

What was tried and did not hold is also a result.

**"The localisation error comes from the viewing angle."** Two independent
measurements refuted it. Five runs with the real model: 3°→0.23 m, 7°→0.11 m,
55°→0.12 m, 81°→0.04 m, 89°→0.10 m. The largest error came from the run looking
almost head-on. There is no relationship between angle and error.

**"Inference at 1280 px finds small objects better."** It halved close-range
recall. YOLO-World is trained at 640 and cannot be moved off it.

**"YOLO-World's larger dataset will improve accuracy on our objects."** Not
supported: its pre-training does not help on our objects (phone 1/8, book 2/8).
The bottleneck is not data volume but that these objects are 20–40 px in this
rendering. The upstream record (ultralytics#10038) also reports a fine-tuned
YOLO-World performing **worse** than a fine-tuned plain YOLOv8 on the custom
classes, while additionally losing zero-shot ability.

---

## Errors found in the measurement rig

A pattern that recurred throughout this project: the measurement rig was
silently wrong when first built, and its output could be read as a system
failure. Recorded because the pattern is still live.

| error | how it looked | what it actually was |
|---|---|---|
| Mis-indented surface scan | drone announced "arrived" at its start position | the detector was left prompting for furniture and boxed the table |
| `place()` grounded beneath the viewpoint | outliers in localisation | the airframe was dropped into furniture and thrown sideways; the viewpoint measured was not the one requested |
| Spawn point inside a cardboard box | "flight anomaly", assumed a crash | a real contact ejection; the clearance check only considered furniture |
| PX4's local origin is the spawn point | target not found from new angles | world coordinates were shifted and the drone swept a different region |
| PX4 statustext channel dead | failsafe reason unknown | the node subscribes but logged zero messages across every run; the reason was published, nobody listened |
| Anomaly log recorded only a distance | "3.0 m / 2.97 s" | genuine flight and an estimator jump looked identical; adding the endpoints revealed z = −4.5 m |

---

## Still open

**The PX4 statustext channel is silent.** The node subscribes to
`/mavros/statustext/recv` and has logged zero messages across every run, while
PX4's own console is full of `commander`, `failsafe` and
`health_and_arming_checks` reports. Everything PX4 says about itself — preflight
failures, arming refusals, failsafe reasons — is invisible to the system.

It has already cost real time: when a failsafe landed the aircraft mid-
experiment the reason had to be guessed at from a tone-alarm line in PX4's
console, and the guess was wrong. That console only exists because the test
chain captures it; a deployment without the chain would have nothing at all.

Likely cause, untested: MAVROS is attached to PX4's *Onboard* link (14580),
while STATUSTEXT is normally streamed to the *Normal* link (18570) where a
ground station would sit.

**The mug's six-run campaign was never collected.** Two attempts failed
entirely on quota and service errors (12 × 429, then 16 × 503 and 6 × 504), so
the four-object table in section 11 stands at three. The mug itself works: it
was found and reached in the interactive session (section 14) with 0.27 m of
error. What is missing is the repeated measurement, not the capability.

**The phone's single 0.73 m deviation** remains unexplained. It was established
as not being the viewing angle (three independent measurements), it has not
recurred, and the phone measured 0.13 m in the most recent session. One
observation, still without a cause.

**Position-stream dropouts after arrival** — four in each of two runs, no
visible harm.
