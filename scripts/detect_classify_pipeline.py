#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Two-stage detection -> classification pipeline for the Zeus signal dataset.

Stage 1 (detection): the 33-class YOLOX-m model trained on zeus-cropped
(data/zeus-cropped/YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth) detects
objects. Its raw 33-class prediction is collapsed into one of three merged
groups (signals / switches / auxilary) via GROUP_MAP below. Detections that
fall outside these three groups (the "overexposed" QC class) are dropped by
default.

Stage 2 (classification): every kept detection is cropped out of the source
image and re-classified by the tf_efficientnet_lite3 model trained on
zeus-cropped-classification (output/best_model.pt) to assign the actual
fine-grained class (e.g. "sig_switch_left_locked").

Run (from the master_thesis conda env, which has yolox/torch/timm):
    conda activate master_thesis
    python detect_classify_pipeline.py \
        --source /path/to/image_or_dir \
        --output-dir ./pipeline_output \
        --save-vis
"""

import argparse
import json
import os
import sys

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

try:
    import timm
except ImportError:
    sys.exit("timm is required: pip install timm")

from yolox.data.data_augment import ValTransform
from yolox.exp import Exp as YOLOXBaseExp
from yolox.utils import fuse_model, postprocess

import eval_pipeline_map
from pipeline_common import DETECTOR_CLASSES, GROUP_MAP, draw_annotated_image, get_image_list

DEFAULT_YOLOX_CKPT = (
    "/home/amo/zeus-training/master_thesis/data/zeus-cropped/"
    "YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth"
)
DEFAULT_CLASSIFIER_CKPT = "/home/amo/zeus-training/output/best_model.pt"


# --------------------------------------------------------------------------------
# Stage 1: YOLOX detector
# --------------------------------------------------------------------------------

class ZeusCroppedExp(YOLOXBaseExp):
    """Matches the exp yolox_m_leaky_zeus was actually trained with
    (depth/width/act/num_classes/input_size, see train_log.txt) — built
    in-script rather than imported, since the on-disk exp file it was
    trained from has since been repurposed for a different (GERALD-cropped,
    31-class) run."""

    def __init__(self, num_classes=33, input_size=(704, 704), test_conf=0.3, nmsthre=0.65):
        super().__init__()
        self.depth = 0.67
        self.width = 0.75
        self.act = "lrelu"
        self.num_classes = num_classes
        self.input_size = input_size
        self.test_size = input_size
        self.test_conf = test_conf
        self.nmsthre = nmsthre


class YoloxDetector:
    def __init__(self, ckpt_path, device, test_size=704, conf=0.3, nms=0.65, fuse=False,
                 class_agnostic=True):
        self.exp = ZeusCroppedExp(input_size=(test_size, test_size), test_conf=conf, nmsthre=nms)
        self.device = device
        self.preproc = ValTransform(legacy=False)
        # True (yolox demo.py's default) suppresses overlapping boxes across raw
        # classes too, which suits real pipeline use (one physical object shouldn't
        # yield multiple surviving boxes). False matches yolox's official
        # eval/coco_evaluator.py protocol exactly, for AP numbers to be comparable.
        self.class_agnostic = class_agnostic

        model = self.exp.get_model()
        model.eval()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if fuse:
            model = fuse_model(model)
        self.model = model.to(device)

    @torch.no_grad()
    def detect(self, img):
        """img: BGR np.ndarray. Returns list of dicts with bbox (x1,y1,x2,y2 in
        original image coords), class_name, class_conf (obj_conf * cls_conf)."""
        height, width = img.shape[:2]
        ratio = min(self.exp.test_size[0] / height, self.exp.test_size[1] / width)

        inp, _ = self.preproc(img, None, self.exp.test_size)
        inp = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)

        outputs = self.model(inp)
        outputs = postprocess(
            outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre,
            class_agnostic=self.class_agnostic,
        )
        output = outputs[0]
        if output is None:
            return []

        output = output.cpu()
        bboxes = output[:, 0:4] / ratio
        obj_conf = output[:, 4]
        cls_conf = output[:, 5]
        cls_idx = output[:, 6].long()

        dets = []
        for box, oc, cc, ci in zip(bboxes, obj_conf, cls_conf, cls_idx):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": DETECTOR_CLASSES[ci.item()],
                "class_conf": float(oc.item() * cc.item()),
            })
        return dets


# --------------------------------------------------------------------------------
# Stage 2: EfficientNet-Lite3 classifier
# --------------------------------------------------------------------------------

class EfficientnetClassifier:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.device = device

        model = timm.create_model("tf_efficientnet_lite3", pretrained=False,
                                   num_classes=len(self.classes))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.model = model.to(device)

        cfg = getattr(model, "pretrained_cfg", None) or model.default_cfg
        mean, std = cfg["mean"], cfg["std"]
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @torch.no_grad()
    def classify(self, crop_bgr):
        """crop_bgr: BGR np.ndarray. Returns (class_name, confidence)."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        return self.classes[idx.item()], float(conf.item())

    @torch.no_grad()
    def classify_batch(self, crops_bgr):
        if not crops_bgr:
            return []
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        x = torch.stack(tensors).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        confs, idxs = probs.max(1)
        return [(self.classes[i.item()], float(c.item())) for i, c in zip(idxs, confs)]


# --------------------------------------------------------------------------------
# Pipeline glue
# --------------------------------------------------------------------------------

def run_pipeline(args):
    device = args.device
    print(f"Loading YOLOX detector from {args.yolox_ckpt} ({device}) ...")
    detector = YoloxDetector(args.yolox_ckpt, device, test_size=args.tsize,
                              conf=args.conf, nms=args.nms, fuse=args.fuse)

    print(f"Loading EfficientNet-Lite3 classifier from {args.classifier_ckpt} ...")
    classifier = EfficientnetClassifier(args.classifier_ckpt, device)

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
        print(f"{img_path}: {len(kept)} detection(s) kept "
              f"(of {len(raw_dets)} raw)")

        if args.save_vis and kept:
            vis_img = draw_annotated_image(img, kept)
            out_path = os.path.join(args.output_dir, "vis", img_path.name)
            cv2.imwrite(out_path, vis_img)

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} image result(s) to {out_json}")

    if args.eval_map:
        print("\nEvaluating COCO mAP against zeus-cropped test-set ground truth "
              "(eval_pipeline_map.py) ...")
        # Ground-truth mAP needs the official protocol's own settings (very low
        # conf threshold for a full PR curve, class-specific NMS), which differ
        # from the --conf/class-agnostic settings tuned for interactive use
        # above, so a second detector instance is built for this pass; the
        # classifier's behavior doesn't depend on those settings and is reused.
        map_detector = YoloxDetector(args.yolox_ckpt, device, test_size=args.tsize,
                                      conf=args.map_conf, nms=args.nms, fuse=args.fuse,
                                      class_agnostic=False)
        eval_pipeline_map.evaluate(
            map_detector, classifier, DETECTOR_CLASSES,
            data_dir=args.gt_data_dir, output_dir=args.output_dir,
            yolox_ckpt_path=args.yolox_ckpt, classifier_ckpt_path=args.classifier_ckpt,
        )


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Image file or directory of images")
    ap.add_argument("--yolox-ckpt", default=DEFAULT_YOLOX_CKPT)
    ap.add_argument("--classifier-ckpt", default=DEFAULT_CLASSIFIER_CKPT)
    ap.add_argument("--output-dir", default="./pipeline_output")
    ap.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    ap.add_argument("--nms", type=float, default=0.65, help="Detection NMS threshold")
    ap.add_argument("--tsize", type=int, default=704, help="YOLOX inference size (square)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fuse", action="store_true", help="Fuse conv+bn in the detector")
    ap.add_argument("--save-vis", action="store_true", help="Save annotated images")
    ap.add_argument("--keep-overexposed", action="store_true",
                     help="Keep 'overexposed' detections (default: dropped, since it "
                          "isn't one of the 3 merged groups) tagged with their raw class")
    ap.add_argument("--eval-map", action="store_true",
                     help="Also score this pipeline's COCO mAP against the zeus-cropped "
                          "test-set ground truth (via eval_pipeline_map.py) and save it "
                          "to <output-dir>/map_results.json. Independent of --source: it "
                          "always evaluates the full test split named by --gt-data-dir, "
                          "since mAP requires ground-truth annotations.")
    ap.add_argument("--gt-data-dir", default="/home/amo/zeus-training/master_thesis/data/zeus-cropped",
                     help="Dataset root containing annotations/instances_test2017.json, "
                          "used only for --eval-map")
    ap.add_argument("--map-conf", type=float, default=0.001,
                     help="Detection confidence threshold used only for --eval-map's "
                          "COCOeval pass (needs a low threshold for a full PR curve, "
                          "independent of --conf which is tuned for interactive use)")
    return ap


if __name__ == "__main__":
    args = make_parser().parse_args()
    run_pipeline(args)
