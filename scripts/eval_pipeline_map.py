#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Evaluate the yolox-m -> efficientnet-lite3 detect+classify pipeline
against the zeus-cropped test-set ground truth, reporting COCO AP in the
same format/metric as
inference_results/yolox-vs-damoyolo-zeus-cropped/comparison.json, so the two
are directly comparable.

Methodology mirrors yolox/evaluators/coco_evaluator.py exactly (same
COCOeval_opt-or-fallback, same stats[0] = AP@[.5:.95]) on the identical
2603-image zeus-cropped test split. Two AP numbers are produced from the
SAME detections in one pass:
  - ap_detector_only : category assigned by the detector's own 33-class head
    (no classifier involved) -- a same-run sanity baseline that should land
    close to comparison.json's yolox-m entry (ap=0.727).
  - ap_pipeline       : category assigned by the efficientnet-lite3 classifier
    run on the cropped detection -- this is the number comparable to the
    single-stage detectors in comparison.json.

evaluate() is the reusable entry point: it takes already-constructed
detector/classifier objects (see detect_classify_pipeline.py, which imports
and calls this as a helper for its own --eval-map flag) so a model already
loaded elsewhere isn't loaded onto the GPU a second time.

Run standalone (from the master_thesis conda env):
    conda activate master_thesis
    python eval_pipeline_map.py \
        --output-dir /home/amo/zeus-training/master_thesis/inference_results/pipeline_yolox_m_efficientnet_lite3_zeus_cropped
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import time

import cv2
import torch
from pycocotools.coco import COCO

from yolox.evaluators.coco_evaluator import per_class_AP_table

# Kept in sync with detect_classify_pipeline.py's own defaults of the same
# name. Not imported from there to avoid a module-level circular import
# (that script imports this one as a helper).
DEFAULT_YOLOX_CKPT = (
    "/home/amo/zeus-training/master_thesis/data/zeus-cropped/"
    "YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth"
)
DEFAULT_CLASSIFIER_CKPT = "/home/amo/zeus-training/output/best_model.pt"

try:
    from yolox.layers import COCOeval_opt as COCOeval
except ImportError:
    from pycocotools.cocoeval import COCOeval
    print("Using standard COCOeval (COCOeval_opt not available).")


def coco_evaluate(coco_gt, dt_list, detector_classes, label):
    if not dt_list:
        print(f"[warn] no detections for {label}, skipping")
        return None, None, None

    _, tmp = tempfile.mkstemp(suffix=".json")
    with open(tmp, "w") as f:
        json.dump(dt_list, f)
    coco_dt = coco_gt.loadRes(tmp)
    os.remove(tmp)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    redirected = io.StringIO()
    with contextlib.redirect_stdout(redirected):
        ev.summarize()
    print(f"\n=== {label} ===")
    print(redirected.getvalue())
    table = per_class_AP_table(ev, class_names=detector_classes)
    print("per class AP:\n" + table)

    per_class = {}
    precisions = ev.eval["precision"]
    for idx, name in enumerate(detector_classes):
        p = precisions[:, :, idx, 0, -1]
        p = p[p > -1]
        per_class[name] = float(p.mean()) if p.size else None

    return float(ev.stats[0]), float(ev.stats[1]), per_class


def evaluate(detector, classifier, detector_classes, data_dir, output_dir,
             yolox_ckpt_path, classifier_ckpt_path, output_filename="map_results.json"):
    """Run `detector` (a YoloxDetector, built with class_agnostic=False for
    protocol parity) + `classifier` (an EfficientnetClassifier) over the
    zeus-cropped test split and score against ground truth. Both must already
    be constructed and on the target device. Returns the results dict; also
    writes it to <output_dir>/<output_filename> if output_dir is given."""
    ann_file = os.path.join(data_dir, "annotations", "instances_test2017.json")
    coco_gt = COCO(ann_file)
    name_to_catid = {c["name"]: c["id"] for c in coco_gt.dataset["categories"]}
    img_ids = coco_gt.getImgIds()

    dt_detector_only = []
    dt_pipeline = []

    t0 = time.time()
    for i, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(data_dir, "images", info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [warn] could not read {img_path}, skipping")
            continue

        dets = detector.detect(img)
        if not dets:
            continue

        crops = []
        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
            crops.append(img[y1:y2, x1:x2])
        cls_results = classifier.classify_batch(crops)

        for d, (cls_name, cls_conf) in zip(dets, cls_results):
            x1, y1, x2, y2 = d["bbox"]
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            dt_detector_only.append({
                "image_id": img_id,
                "category_id": name_to_catid[d["class_name"]],
                "bbox": bbox_xywh,
                "score": d["class_conf"],
            })
            dt_pipeline.append({
                "image_id": img_id,
                "category_id": name_to_catid[cls_name],
                "bbox": bbox_xywh,
                "score": d["class_conf"] * cls_conf,
            })

        if (i + 1) % 200 == 0 or (i + 1) == len(img_ids):
            print(f"  {i + 1}/{len(img_ids)} images processed ({time.time() - t0:.1f}s)")

    ap_det, ap50_det, per_class_det = coco_evaluate(
        coco_gt, dt_detector_only, detector_classes,
        "detector-only (yolox-m's own 33-class head, same-run baseline)",
    )
    ap_pipe, ap50_pipe, per_class_pipe = coco_evaluate(
        coco_gt, dt_pipeline, detector_classes, "yolox-m + efficientnet-lite3 pipeline",
    )

    yolox_size_mb = os.path.getsize(yolox_ckpt_path) / 1e6
    cls_size_mb = os.path.getsize(classifier_ckpt_path) / 1e6
    tsize = detector.exp.test_size[0]

    out = {
        "meta": {
            "dataset": "zeus-cropped",
            "split": "test",
            "notes": (
                f"Two-stage pipeline eval (detect_classify_pipeline.py): yolox_m_leaky_zeus "
                f"detects at {tsize}x{tsize} (conf={detector.exp.test_conf}, "
                f"nms={detector.exp.nmsthre}, class-specific NMS to match "
                "yolox/evaluators/coco_evaluator.py exactly); every detection is cropped and "
                f"re-classified by tf_efficientnet_lite3 ({os.path.basename(classifier_ckpt_path)}, "
                "280x280 input) into the 33 zeus-cropped classes. Evaluated with pycocotools "
                "COCOeval, same metric (stats[0] = AP@[.5:.95]) and identical 2603-image test "
                "split as inference_results/yolox-vs-damoyolo-zeus-cropped/comparison.json, so "
                "'ap' below is directly comparable to that file's per-model 'ap' values. "
                "'ap_detector_only_baseline' reruns the identical boxes/scores but categorized "
                "by the detector's own head (classifier not involved) as a same-run sanity "
                "check against that file's yolox-m entry (ap=0.727)."
            ),
        },
        "overall": {
            "yolox-m+efficientnet-lite3": {
                "ap": ap_pipe,
                "ap50": ap50_pipe,
                "checkpoint_size_mb": round(yolox_size_mb + cls_size_mb, 1),
                "detector_checkpoint_size_mb": round(yolox_size_mb, 1),
                "classifier_checkpoint_size_mb": round(cls_size_mb, 1),
                "detector_eval_resolution": f"{tsize}x{tsize}",
                "classifier_eval_resolution": "280x280",
                "per_class_ap": per_class_pipe,
            },
            "yolox-m_detector_only_baseline": {
                "ap": ap_det,
                "ap50": ap50_det,
                "checkpoint_size_mb": round(yolox_size_mb, 1),
                "detector_eval_resolution": f"{tsize}x{tsize}",
                "per_class_ap": per_class_det,
            },
        },
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {out_path}")

    print(f"ap (pipeline)        = {ap_pipe}")
    print(f"ap (detector-only)   = {ap_det}  (comparison.json yolox-m = 0.727)")
    return out


def run(args):
    """Standalone CLI entry point: builds its own detector/classifier."""
    from detect_classify_pipeline import DETECTOR_CLASSES, EfficientnetClassifier, YoloxDetector

    print(f"Loading YOLOX detector from {args.yolox_ckpt} ({args.device}) ...")
    detector = YoloxDetector(args.yolox_ckpt, args.device, test_size=args.tsize,
                              conf=args.conf, nms=args.nms, fuse=args.fuse,
                              class_agnostic=False)
    print(f"Loading EfficientNet-Lite3 classifier from {args.classifier_ckpt} ...")
    classifier = EfficientnetClassifier(args.classifier_ckpt, args.device)

    return evaluate(detector, classifier, DETECTOR_CLASSES, args.data_dir, args.output_dir,
                     args.yolox_ckpt, args.classifier_ckpt, output_filename="results.json")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/amo/zeus-training/master_thesis/data/zeus-cropped")
    ap.add_argument("--yolox-ckpt", default=DEFAULT_YOLOX_CKPT)
    ap.add_argument("--classifier-ckpt", default=DEFAULT_CLASSIFIER_CKPT)
    ap.add_argument("--output-dir",
                     default="/home/amo/zeus-training/master_thesis/inference_results/"
                             "pipeline_yolox_m_efficientnet_lite3_zeus_cropped")
    ap.add_argument("--conf", type=float, default=0.001,
                     help="Detection confidence threshold (0.001, matching the tools/eval.py "
                          "settings comparison.json's yolox-s/m/l numbers were produced with)")
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--tsize", type=int, default=704)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fuse", action="store_true")
    return ap


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
