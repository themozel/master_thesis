#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Build calibration sets (.npy, uint8 NHWC) for re-quantizing the two
models with the Hailo Dataflow Compiler. Preprocessing matches exactly what
detect_classify_pipeline_hailo.py feeds the HEFs at runtime:

  detector   : letterbox to 704x704 (aspect kept, pad 114, top-left)
  classifier : plain resize to 280x280 with INTER_CUBIC

--color picks the channel order written to the .npy (OpenCV reads BGR):
  detector   : bgr. YOLOX trains on BGR, and the rebuilt detector alls has
               no input_conversion, so the model sees exactly this.
  classifier : its alls keeps input_conversion(bgr_to_rgb) (trained on RGB).
               Whether the DFC wants calibration data before (bgr) or after
               (rgb) that conversion is decided in README step 6 with
               eval_emulated_classifier.py -- build both, they're cheap.

Examples:
  python prepare_calib.py detector --images-dir .../zeus-cropped/images/train \
      --n 1024 --color bgr --out calib/detector_bgr.npy
  python prepare_calib.py classifier --crops-dir .../zeus-cropped-classification/train \
      --n 2048 --color rgb --out calib/classifier_rgb.npy
"""

import argparse
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(img, size):
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXT)


def pick_detector_images(images_dir, n, rng):
    files = list_images(images_dir)
    rng.shuffle(files)
    return files[:n]


def pick_classifier_crops(crops_dir, n, rng):
    """Class-stratified: every class gets up to ceil(n / num_classes) crops
    (all of them if it has fewer), the remainder is filled at random, so rare
    classes are represented in the quantization ranges."""
    per_class = {d.name: list_images(d) for d in sorted(Path(crops_dir).iterdir()) if d.is_dir()}
    quota = math.ceil(n / max(1, len(per_class)))
    picked, rest = [], []
    for files in per_class.values():
        rng.shuffle(files)
        picked += files[:quota]
        rest += files[quota:]
    rng.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    rng.shuffle(picked)
    print("crops per class: " + ", ".join(
        f"{k}={min(len(v), quota)}" for k, v in per_class.items()))
    return picked[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["detector", "classifier"])
    ap.add_argument("--images-dir", help="detector: full frames (e.g. zeus-cropped/images/train)")
    ap.add_argument("--crops-dir", help="classifier: ImageFolder root (e.g. .../classification/train)")
    ap.add_argument("--n", type=int, default=None, help="default 1024 detector / 2048 classifier")
    ap.add_argument("--size", type=int, default=None, help="default 704 detector / 280 classifier")
    ap.add_argument("--color", choices=["bgr", "rgb"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .npy path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.model == "detector":
        n, size = args.n or 1024, args.size or 704
        files = pick_detector_images(args.images_dir, n, rng)
        prep = lambda img: letterbox(img, size)
    else:
        n, size = args.n or 2048, args.size or 280
        files = pick_classifier_crops(args.crops_dir, n, rng)
        prep = lambda img: cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8, shape=(len(files), size, size, 3))
    for i, f in enumerate(files):
        img = prep(cv2.imread(str(f)))
        data[i] = img[..., ::-1] if args.color == "rgb" else img
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)}")
    data.flush()
    print(f"Wrote {args.out}: {data.shape} uint8 ({args.color})")


if __name__ == "__main__":
    main()
