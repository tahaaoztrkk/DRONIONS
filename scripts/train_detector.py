#!/usr/bin/env python3
"""
Görevi: Odanın nesneleri için kapalı kümeli bir dedektör eğitir.

The open-vocabulary detector is the measured bottleneck: at real search
viewpoints it finds the phone in one of six, the book one of six, the mug three
of six. Gate tuning cannot move that, because the gates only ever see what the
detector already found. This trains a small closed-set model on the same
objects to find out how much of that limit is the approach and how much is the
objects being genuinely small.

It is not meant to replace YOLO-World. Doing that would cost the thing the
project is about -- a user saying "find my charger" and the system trying,
without anyone retraining anything. The intended shape is both: the open model
for whatever is asked for, this one for the objects it has actually been shown,
and a measured statement about what each is worth.

Read what the number will mean before trusting it. This trains on Gazebo meshes
under one lighting setup, so it learns this dataset, not this task. A recall
figure from it says what a task-specific detector buys in simulation, and says
nothing about a real room.

  scripts/train_detector.py                       # data/room, 100 epochs
  scripts/train_detector.py --epochs 40 --model yolov8s.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, '/home/taha/DRONIONS')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/room/data.yaml')
    ap.add_argument('--model', default='yolov8n.pt',
                    help='baslangic agirliklari; n en kucuk')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=960,
                    help='egitim cozunurlugu; nesneler kucuk oldugu icin '
                         'varsayilan 640 degil')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--name', default='room')
    a = ap.parse_args()

    if not os.path.exists(a.data):
        sys.exit(f"Veri seti yok: {a.data}\n"
                 "Once scripts/make_dataset.py calistirin.")

    from ultralytics import YOLO

    # Bigger than the usual 640. The objects this has to find are 20-40 px
    # across at search range, and halving the image before inference is the
    # first thing that would throw them away -- the same trade measured on the
    # open-vocabulary side, where it cost more than it bought because that
    # model is trained at 640 and cannot be moved off it. Here the training
    # resolution is ours to choose, so it matches what the camera delivers.
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                name=a.name, project='runs/detect', exist_ok=True,
                # The dataset is one room under one light, so the model will
                # happily memorise it. Augmentation is the cheap defence, and
                # flips are safe here in a way rotation is not: the camera is
                # always level and always pitched down, so an upside-down
                # training image is a view the drone can never take.
                fliplr=0.5, flipud=0.0, degrees=0.0,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
                translate=0.1, scale=0.4)

    best = os.path.join('runs/detect', a.name, 'weights', 'best.pt')
    print(f"\n-> {best}")
    print("Karsilastirmak icin: scripts/compare_detectors.py --weights " + best)


if __name__ == '__main__':
    main()
