#!/usr/bin/env python3
"""
Görevi: Gerçek konumlardan otomatik etiketli tespit veri seti üretir.

The open-vocabulary detector is the measured bottleneck. At real search
viewpoints it finds the phone in one of six, the book one of six, the mug three
of six -- and no amount of gate tuning moves that, because the gates only see
what the detector already found. The question is what a detector trained on
these objects would do instead.

Labelling by hand is the usual cost of answering that, and in simulation it is
avoidable: the object positions are known exactly, so the boxes can be computed
rather than drawn. This flies the drone over a grid of viewpoints, projects each
object's 3D extent into the frame, and writes YOLO-format labels. Nothing is
traced by hand and nothing depends on the detector under test, which is the
point -- a dataset built from detections would inherit whatever the detector is
already confused by.

Read the limits before trusting what comes out. The boxes are axis-aligned
world extents, so they are slightly loose, uniformly. Occlusion is not modelled:
an object hidden behind furniture still gets a label, which teaches the model
that a table edge is sometimes a mug. And a model trained here learns these
meshes under this lighting; the numbers it produces are about this dataset and
do not transfer to a real room.

  scripts/make_dataset.py                    # room world + bridge up
  scripts/make_dataset.py --grid 5 --out data/room
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import cv2
import rclpy

sys.path.insert(0, '/home/taha/DRONIONS')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation.spatial import CAMERA_HFOV, CAMERA_PITCH_DOWN
from gz_pose import PoseReader, find_world
from experiment_small_objects import (Grab, MODEL, CAM_OFFSET, fresh, place,
                                      settled_pose, world_control)

# class -> (centre x, y, z of the visible body, half-extents in world axes)
#
# Half-extents use the largest horizontal dimension on both axes, so the box
# encloses the object at any yaw. Slightly loose, and loose in the same way for
# every sample, which is what matters for a label.
OBJECTS = [
    ("laptop", (3.05, 0.22, 1.112), (0.19, 0.19, 0.097)),
    ("book", (3.05, -0.22, 1.067), (0.105, 0.105, 0.026)),
    ("mug", (2.62, 0.24, 1.063), (0.083, 0.083, 0.048)),
    ("phone", (2.62, -0.16, 1.023), (0.080, 0.080, 0.006)),
    ("box", (0.10, 1.10, 0.15), (0.25, 0.25, 0.15)),
]
CLASSES = [name for name, _, _ in OBJECTS]

# Room bounds to fly within, kept off the walls.
AREA_X = (-0.8, 2.6)
AREA_Y = (-1.3, 1.3)
ALTITUDES = (1.4, 1.8, 2.2)
# Below this the object is a smudge and the label teaches noise.
MIN_BOX_PX = 10


def corners(centre, half):
    cx, cy, cz = centre
    hx, hy, hz = half
    return [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def project(point, drone_xyz, yaw, img_w, img_h):
    """World point to pixel, or None if behind the camera."""
    cam = (drone_xyz[0] + CAM_OFFSET[0] * math.cos(yaw) - CAM_OFFSET[1] * math.sin(yaw),
           drone_xyz[1] + CAM_OFFSET[0] * math.sin(yaw) + CAM_OFFSET[1] * math.cos(yaw),
           drone_xyz[2] + CAM_OFFSET[2])
    dx, dy, dz = point[0] - cam[0], point[1] - cam[1], point[2] - cam[2]
    bx = dx * math.cos(yaw) + dy * math.sin(yaw)
    by = -dx * math.sin(yaw) + dy * math.cos(yaw)
    cp, sp = math.cos(CAMERA_PITCH_DOWN), math.sin(CAMERA_PITCH_DOWN)
    fx = bx * cp - dz * sp
    fz = bx * sp + dz * cp
    if fx <= 0.01:
        return None
    focal = (img_w / 2.0) / math.tan(CAMERA_HFOV / 2.0)
    return (img_w / 2.0 - by / fx * focal, img_h / 2.0 - fz / fx * focal)


def box_for(centre, half, drone_xyz, yaw, img_w, img_h):
    """Pixel box enclosing the object, or None if not usefully in frame."""
    pts = [project(c, drone_xyz, yaw, img_w, img_h) for c in corners(centre, half)]
    pts = [p for p in pts if p]
    if len(pts) < 8:                      # partly behind the camera: skip it
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0.0, min(xs)), min(float(img_w), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(img_h), max(ys))
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    # Mostly outside the frame means the visible part is not what the label
    # describes, and a box clipped to the edge teaches the wrong shape.
    full = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if full <= 0 or (x1 - x0) * (y1 - y0) / full < 0.6:
        return None
    return x0, y0, x1, y1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=5, help='her eksende kac nokta')
    ap.add_argument('--bearings', type=int, default=6)
    ap.add_argument('--out', default='data/room')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    world = find_world()
    if not world or 'room' not in world:
        sys.exit(f"Oda dunyasi gerekiyor (bulunan: {world}).")
    pose = PoseReader(world)
    pose.start()
    if pose.wait_for_pose(15) is None:
        sys.exit(f"Drone pozu gelmedi. Gorulen: {pose.seen_names()}")

    rclpy.init()
    node = rclpy.create_node('make_dataset')
    g = Grab(node)
    world_control(world, 'pause: true')
    if fresh(node, g, world) is None:
        sys.exit("Kare yok -- kopru calisiyor mu?")

    for split in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)

    xs = [AREA_X[0] + (AREA_X[1] - AREA_X[0]) * i / max(1, a.grid - 1)
          for i in range(a.grid)]
    ys = [AREA_Y[0] + (AREA_Y[1] - AREA_Y[0]) * i / max(1, a.grid - 1)
          for i in range(a.grid)]
    bearings = [2 * math.pi * i / a.bearings for i in range(a.bearings)]

    kept = skipped = 0
    counts = {c: 0 for c in CLASSES}
    n = 0
    for vx in xs:
        for vy in ys:
            for vz in ALTITUDES:
                for yaw in bearings:
                    # A little jitter, so the model does not learn the grid.
                    jx = vx + random.uniform(-0.12, 0.12)
                    jy = vy + random.uniform(-0.12, 0.12)
                    jz = vz + random.uniform(-0.10, 0.10)
                    jyaw = yaw + random.uniform(-0.10, 0.10)
                    place(world, MODEL, jx, jy, jz, jyaw)
                    img = fresh(node, g, world)
                    if img is None:
                        continue
                    drone = settled_pose(pose) or (jx, jy, jz)
                    h_img, w_img = img.shape[:2]
                    lines = []
                    for idx, (name, centre, half) in enumerate(OBJECTS):
                        box = box_for(centre, half, drone, jyaw, w_img, h_img)
                        if not box:
                            continue
                        x0, y0, x1, y1 = box
                        lines.append(
                            f"{idx} {((x0 + x1) / 2) / w_img:.6f} "
                            f"{((y0 + y1) / 2) / h_img:.6f} "
                            f"{(x1 - x0) / w_img:.6f} {(y1 - y0) / h_img:.6f}")
                        counts[name] += 1
                    n += 1
                    if not lines:
                        # Frames with nothing in them are worth keeping, but not
                        # in bulk: a set that is mostly empty teaches the model
                        # that the answer is usually nothing.
                        skipped += 1
                        if skipped % 4:
                            continue
                    split = "val" if n % 5 == 0 else "train"
                    stem = f"{n:05d}"
                    cv2.imwrite(os.path.join(a.out, "images", split, stem + ".jpg"), img)
                    with open(os.path.join(a.out, "labels", split, stem + ".txt"),
                              "w") as fh:
                        fh.write("\n".join(lines))
                    kept += 1
                    if kept % 50 == 0:
                        print(f"  {kept} kare  ({', '.join(f'{k}:{v}' for k, v in counts.items())})")

    with open(os.path.join(a.out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out)}\n"
                 f"train: images/train\nval: images/val\n"
                 f"names:\n" + "".join(f"  {i}: {c}\n"
                                       for i, c in enumerate(CLASSES)))

    print(f"\n{kept} kare yazildi ({n} bakis denendi)")
    for c in CLASSES:
        print(f"  {c:8} {counts[c]:5} etiket")
    print(f"-> {a.out}/data.yaml")

    world_control(world, 'pause: false')
    pose.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
