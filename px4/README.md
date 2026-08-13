# PX4 simulation assets

These are the source of truth for the simulation. PX4 looks for models, worlds
and airframes inside its own tree, so that is where they were originally
written — and nothing was tracking them there. `Tools/simulation/gz` is a
submodule, so they sat as untracked content inside it: a fresh clone of PX4, or
a `git clean` in it, would have taken the whole simulation with it. Five
campaigns of measurements rested on files that existed on exactly one disk.

Install them into a PX4 checkout with:

```bash
scripts/install_px4_assets.sh              # into ~/PX4-Autopilot
scripts/install_px4_assets.sh --check      # report drift, change nothing
```

`--check` exits non-zero when the PX4 copy has drifted, so it is worth running
before a measurement campaign — a model edited in PX4 and not brought back here
is how results stop being reproducible.

| File | What it carries |
|---|---|
| `models/x500_dronions/model.sdf` | the stock x500 plus a camera pitched 0.35 rad down, a widened forward lidar fan, and a downward lidar |
| `models/x500_dronions/model.config` | model metadata |
| `worlds/dronions_scenario.sdf` | the 3 m wall, the cardboard box target, the blue box and sphere distractors |
| `airframes/4022_gz_x500_dronions` | registers the airframe with PX4 |

The installer also adds the airframe to
`ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt`. That edit is why
installing needs a PX4 rebuild afterwards:

```bash
cd ~/PX4-Autopilot && make px4_sitl gz_x500_dronions_dronions_scenario
```

## Why each deviation from stock exists

**Camera pitched down 0.35 rad.** Stock `mono_cam` looks level, which puts
ground targets outside the frame entirely from a 2 m hover, and drops them out
of the bottom of the frame on approach.

**Forward lidar widened to 20 samples over ±0.35 rad.** The stock LW20 is a
single ray, which only sees an obstacle dead ahead; `OBSTACLE_SAFE_DISTANCE`
was tuned against a fan. The stock roll of 1.57 rad is also removed — with one
ray it changes nothing, but it turns a fan on its side, and that is why every
return read `.inf` while the drone circled a 3 m wall.

**Downward lidar added.** The forward fan says nothing about what the drone is
above. After climbing to 3.5 m to see past the wall, the drone would spot the
target and descend into the top of the wall it was still standing over.
