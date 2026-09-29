#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Two-stage detection -> classification pipeline for the Zeus signal
dataset, running on Hailo-8 hardware via compiled .hef files instead of the
PyTorch checkpoints (see detect_classify_pipeline.py for that version; the
two share group/class definitions and the visualization renderer via
pipeline_common.py, and produce the same results.json schema).

--------------------------------------------------------------------------
IMPORTANT - the HailoRT calls in this script are UNTESTED
--------------------------------------------------------------------------
It was written on a dev machine with no HailoRT runtime (`hailo_platform`)
installed and no Hailo-8 device attached, so while both .hef files below
were compiled for real (parse/optimize/compile all ran successfully, using
the `hailo_dfc` conda env's hailomz/Dataflow Compiler -- pure software, no
device needed), none of the actual HailoRT inference calls in this script
could be exercised end-to-end. The API calls mirror the pattern this
repo already uses in
hailo_model_zoo/hailo_model_zoo/core/infer/hw_infer_utils.py (VDevice / HEF
/ ConfigureParams / InputVStreamParams / OutputVStreamParams / InferVStreams),
and the on-device NMS output parsing follows HailoRT's documented
HAILO_FORMAT_ORDER_HAILO_NMS convention (a list of `num_classes` arrays,
each row [y_min, x_min, y_max, x_max, score] normalized to the network's
padded input size) -- but this must be verified against the real output on
your target device before trusting it. If the shapes don't match, the
detector will raise with a message pointing at _parse_nms_output().

The classifier .hef (output/hailo/efficientnet_lite3_zeus_cropped.hef) was
compiled from output/efficientnet_lite3_custom.onnx via:
    conda activate hailo_dfc
    hailomz parse --yaml <yaml> --ckpt output/efficientnet_lite3_custom.onnx
    hailomz optimize --yaml <yaml> --har <parsed.har> --calib-path <~100 sample crops>
    hailomz compile --yaml <yaml> --har <optimized.har>
using a yaml/alls adapted from the generic efficientnet_lite3 template (the
alls must live under hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/ --
that's where hailomz's path_resolver looks, regardless of what --yaml
points at). The compiled alls bakes BOTH `input_conversion(bgr_to_rgb)` and
`logits_layer(efficientnet_lite3_zeus_cropped/fc1, softmax, -1, cpu)`
on-device, matching this script's defaults (--no-classifier-bgr-to-rgb
implied by bgr_to_rgb_on_host=False below, --classifier-raw-logits off).
If you recompile with a different alls, double check those two choices
still match before trusting results -- get them wrong and the classifier
will silently run on flipped color channels or double-softmaxed scores
without erroring.

Run (on the Hailo-8 host, with hailo_platform installed):
    python detect_classify_pipeline_hailo.py \
        --source /path/to/image_or_dir \
        --output-dir ./pipeline_output_hailo \
        --save-vis
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from pipeline_common import DETECTOR_CLASSES, GROUP_MAP, draw_annotated_image, get_image_list

try:
    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
except ImportError:
    sys.exit(
        "hailo_platform (HailoRT's Python API) is required and was not found.\n"
        "This script must run on the Hailo-8 host with HailoRT installed, not "
        "on a plain dev machine -- see detect_classify_pipeline.py for the "
        "PyTorch/GPU version."
    )

DEFAULT_DETECTOR_HEF = (
    "/home/amo/zeus-training/master_thesis/data/zeus-cropped/"
    "YOLOX_outputs/yolox_m_leaky_zeus/hailo/yolox_m_leaky_zeus_zeuscropped.hef"
)
DEFAULT_DETECTOR_INPUT_SIZE = 704  # matches yolox_m_leaky_zeus_zeuscropped.yaml's input_shape
DEFAULT_CLASSIFIER_HEF = "/home/amo/zeus-training/output/hailo/efficientnet_lite3_zeus_cropped.hef"
DEFAULT_CLASSIFIER_INPUT_SIZE = 280  # matches train_efficientnet_lite3.py's --img-size


# --------------------------------------------------------------------------------
# Thin wrapper around a single-network HEF (mirrors
# hailo_model_zoo's own HefWrapper pattern in core/infer/hw_infer_utils.py)
# --------------------------------------------------------------------------------

class HailoModel:
    def __init__(self, hef_path, input_format_type=FormatType.UINT8,
                 output_format_type=FormatType.FLOAT32):
        self.hef = HEF(hef_path)
        self.device = VDevice()
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.device.configure(self.hef, configure_params)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        self.input_name = self.input_vstream_info.name
        _, in_h, in_w, _ = self.input_vstream_info.shape if len(self.input_vstream_info.shape) == 4 \
            else (1, *self.input_vstream_info.shape)
        self.input_h, self.input_w = in_h, in_w

        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=input_format_type
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=output_format_type
        )

    def infer(self, input_uint8_hwc):
        """input_uint8_hwc: single HxWx3 uint8 array (already resized to the
        network's input shape). Returns {output_name: np.ndarray} for batch=1."""
        batch = np.expand_dims(input_uint8_hwc, axis=0)
        with InferVStreams(self.network_group, self.input_vstreams_params,
                            self.output_vstreams_params) as pipeline:
            with self.network_group.activate(self.network_group_params):
                results = pipeline.infer({self.input_name: batch})
        return results


# --------------------------------------------------------------------------------
# Stage 1: YOLOX detector (NMS baked into the HEF, see nms_config_yolox_m_leaky_zeus_zeuscropped.json)
# --------------------------------------------------------------------------------

def _letterbox(img, size):
    """Same convention as yolox's own preproc: resize preserving aspect ratio,
    pad to (size, size) with 114 gray, anchored top-left. Returns (padded_bgr,
    ratio) so boxes decoded in the padded/network space can be mapped back to
    the original image with `box / ratio`."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, ratio


def _class_agnostic_nms(dets, iou_th):
    """Greedy IoU suppression across ALL classes combined (not just within
    each class). Needed because the on-chip nms_postprocess baked into the
    detector HEF only suppresses duplicates within each of the 33 raw
    detector classes (Hailo's NMS postprocess has no class-agnostic option --
    see nms_config_yolox_*.json's per-class max_proposals_per_class), so two
    different raw classes firing on the same physical object (e.g. the
    confusable sig_switch_left_free / sig_switch_left_locked pair) both
    survive untouched however much they overlap. This mirrors class_agnostic
    =True in detect_classify_pipeline.py's YoloxDetector (torchvision.ops.nms
    over all boxes), reimplemented without torch since this script must run
    on the lean edge device."""
    if not dets:
        return dets
    boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
    scores = np.array([d["class_conf"] for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_th]
    return [dets[i] for i in keep]


class HailoYoloxDetector:
    def __init__(self, hef_path, input_size=DEFAULT_DETECTOR_INPUT_SIZE, conf=0.3, nms_iou_th=0.5):
        self.model = HailoModel(hef_path)
        self.input_size = input_size
        self.conf = conf
        # Cross-class suppression threshold (see _class_agnostic_nms). Distinct
        # from the per-class nms_iou_th (0.65) baked into the HEF itself --
        # 0.5 matches the overlap the rosbag regression review flagged as
        # "should have been merged."
        self.nms_iou_th = nms_iou_th
        # yolox_m_leaky_zeus_zeuscropped.alls does input_conversion(bgr_to_rgb)
        # on-device, so the host feeds BGR (OpenCV's native order) unchanged.

    def detect(self, img):
        """img: BGR np.ndarray. Returns list of dicts with bbox (x1,y1,x2,y2 in
        original image coords), class_name, class_conf -- same schema as
        detect_classify_pipeline.py's YoloxDetector.detect()."""
        height, width = img.shape[:2]
        padded, ratio = _letterbox(img, self.input_size)

        results = self.model.infer(padded)
        raw_output = results[self.model.output_vstream_infos[0].name]
        per_class_dets = _parse_nms_output(raw_output, num_classes=len(DETECTOR_CLASSES))

        dets = []
        for cls_idx, class_dets in enumerate(per_class_dets):
            for det in class_dets:
                y1n, x1n, y2n, x2n, score = det
                if score < self.conf:
                    continue
                # normalized [0,1] over the padded network input -> padded-pixel
                # coords -> original image coords (same `/ ratio` as the
                # PyTorch path's YoloxDetector.detect()).
                x1 = x1n * self.input_size / ratio
                y1 = y1n * self.input_size / ratio
                x2 = x2n * self.input_size / ratio
                y2 = y2n * self.input_size / ratio
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": DETECTOR_CLASSES[cls_idx],
                    "class_conf": float(score),
                })
        return _class_agnostic_nms(dets, self.nms_iou_th)


def _parse_nms_output(raw_output, num_classes):
    """HailoRT's documented format for an on-device NMS output
    (FORMAT_ORDER_HAILO_NMS, which is what `device_pre_post_layers.nms: true`
    / hpp compiles to): for batch=1, a list of length `num_classes`, each
    entry a (N_i, 5) array of [y_min, x_min, y_max, x_max, score] normalized
    to [0, 1]. HailoRT wraps this per-image, so unwrap the batch dim first.

    UNVERIFIED on real hardware (see module docstring) -- if this raises or
    the shapes look wrong, print(raw_output) structure/shape here and adjust
    to match what your HailoRT version actually returns."""
    try:
        per_image = raw_output[0] if isinstance(raw_output, (list, tuple, np.ndarray)) else raw_output
        if len(per_image) != num_classes:
            raise ValueError(f"expected {num_classes} per-class entries, got {len(per_image)}")
        return [np.asarray(class_dets).reshape(-1, 5) for class_dets in per_image]
    except Exception as e:
        raise RuntimeError(
            "Could not parse the detector HEF's NMS output in the expected "
            "HailoRT HAILO_NMS layout (list of per-class (N,5) [y1,x1,y2,x2,score] "
            f"arrays). Raw output type={type(raw_output)}. Inspect its actual "
            "structure/shape on your device and fix _parse_nms_output()."
        ) from e


# --------------------------------------------------------------------------------
# Stage 2: EfficientNet-Lite3 classifier
# --------------------------------------------------------------------------------

class HailoEfficientnetClassifier:
    def __init__(self, hef_path, classes_json_path, input_size=DEFAULT_CLASSIFIER_INPUT_SIZE,
                 bgr_to_rgb_on_host=True, apply_softmax=False):
        self.model = HailoModel(hef_path)
        self.input_size = input_size
        self.bgr_to_rgb_on_host = bgr_to_rgb_on_host
        # efficientnet_lite3.alls' postprocessing bakes softmax on-device
        # (`device_pre_post_layers.softmax: true`); only turn this on if your
        # compiled alls skipped that and the HEF outputs raw logits instead.
        self.apply_softmax = apply_softmax
        with open(classes_json_path) as f:
            self.classes = json.load(f)

    def _preprocess(self, crop_bgr):
        img = crop_bgr
        if self.bgr_to_rgb_on_host:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        return img.astype(np.uint8)

    def classify(self, crop_bgr):
        results = self.model.infer(self._preprocess(crop_bgr))
        probs = np.asarray(results[self.model.output_vstream_infos[0].name]).reshape(-1)
        if self.apply_softmax:
            probs = np.exp(probs - probs.max())
            probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def classify_batch(self, crops_bgr):
        # HailoModel.infer() above is single-image; batching would need a
        # multi-frame input array through the same InferVStreams pipeline,
        # skipped here for simplicity since per-image crop counts in this
        # pipeline are small. Swap in real batching if throughput matters.
        return [self.classify(c) for c in crops_bgr]


# --------------------------------------------------------------------------------
# Pipeline glue (mirrors detect_classify_pipeline.py's run_pipeline exactly,
# using the Hailo classes above in place of the PyTorch ones)
# --------------------------------------------------------------------------------

def run_pipeline(args):
    print(f"Loading YOLOX detector HEF from {args.detector_hef} ...")
    detector = HailoYoloxDetector(args.detector_hef, input_size=args.tsize, conf=args.conf,
                                   nms_iou_th=args.nms)

    print(f"Loading EfficientNet-Lite3 classifier HEF from {args.classifier_hef} ...")
    classifier = HailoEfficientnetClassifier(
        args.classifier_hef, args.classifier_classes, input_size=args.classifier_size,
        bgr_to_rgb_on_host=args.classifier_bgr_to_rgb, apply_softmax=args.classifier_raw_logits,
    )

    images = get_image_list(args.source)
    if not images:
        sys.exit(f"No images found at {args.source}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)

    all_results = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] could not read {img_path}, skipping")
            continue

        raw_dets = detector.detect(img)

        kept = []
        for d in raw_dets:
            group = GROUP_MAP.get(d["class_name"])
            if group is None and not args.keep_overexposed:
                continue
            d["detector_group"] = group if group is not None else d["class_name"]
            kept.append(d)

        crops = []
        for d in kept:
            x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
            crops.append(img[y1:y2, x1:x2])

        cls_results = classifier.classify_batch(crops)
        for d, (cls_name, cls_conf) in zip(kept, cls_results):
            d["classifier_class"] = cls_name
            d["classifier_conf"] = cls_conf

        all_results.append({"image": str(img_path), "detections": kept})
        print(f"{img_path}: {len(kept)} detection(s) kept (of {len(raw_dets)} raw)")

        if args.save_vis and kept:
            vis_img = draw_annotated_image(img, kept)
            out_path = os.path.join(args.output_dir, "vis", img_path.name)
            cv2.imwrite(out_path, vis_img)

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} image result(s) to {out_json}")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Image file or directory of images")
    ap.add_argument("--detector-hef", default=DEFAULT_DETECTOR_HEF)
    ap.add_argument("--classifier-hef", default=DEFAULT_CLASSIFIER_HEF)
    ap.add_argument("--classifier-classes", default="/home/amo/zeus-training/output/classes.json",
                     help="classes.json written by train_efficientnet_lite3.py, giving the "
                          "output index -> class name mapping")
    ap.add_argument("--output-dir", default="./pipeline_output_hailo")
    ap.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    ap.add_argument("--nms", type=float, default=0.5,
                     help="Cross-class NMS IoU threshold applied on the host after the HEF's "
                          "on-chip per-class NMS, to suppress duplicate boxes of different raw "
                          "classes covering the same object (see _class_agnostic_nms)")
    ap.add_argument("--tsize", type=int, default=DEFAULT_DETECTOR_INPUT_SIZE,
                     help="Detector network input size (square), must match how the HEF was compiled")
    ap.add_argument("--classifier-size", type=int, default=DEFAULT_CLASSIFIER_INPUT_SIZE,
                     help="Classifier network input size (square), must match how the HEF was compiled")
    ap.add_argument("--classifier-bgr-to-rgb", action=argparse.BooleanOptionalAction, default=False,
                     help="Convert crops BGR->RGB on the host before feeding the classifier HEF. "
                          "Default off, since the compiled efficientnet_lite3_zeus_cropped.hef "
                          "already does this on-device (input_conversion(bgr_to_rgb) in its "
                          "alls) -- only turn this on if you recompile without that line")
    ap.add_argument("--classifier-raw-logits", action="store_true",
                     help="Apply softmax on the host to the classifier HEF's output. Only needed "
                          "if your compiled alls did NOT bake softmax on-device (the generic "
                          "efficientnet_lite3.alls template does by default)")
    ap.add_argument("--save-vis", action="store_true", help="Save annotated images")
    ap.add_argument("--keep-overexposed", action="store_true",
                     help="Keep 'overexposed' detections (default: dropped, since it "
                          "isn't one of the 3 merged groups) tagged with their raw class")
    return ap


if __name__ == "__main__":
    args = make_parser().parse_args()
    run_pipeline(args)
