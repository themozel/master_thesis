#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Top-1 accuracy of the classifier HAR in the DFC emulator, on the
classification test crops (ImageFolder layout), before touching hardware.

Contexts:
  fp_optimized : float model incl. the alls pre/post layers (normalization,
                 input_conversion, softmax). Should match PyTorch (~97%).
                 If it doesn't, the INPUT FORMAT or alls normalization is
                 wrong -- fix that before any quantization tuning.
  quantized    : emulated int8 model == what the HEF will do (within noise).
                 The gap to fp_optimized is the quantization loss.

Try --color bgr and --color rgb on the fp_optimized context; the one that
gives ~97% is the input format the DFC expects -> use it for the
calibration sets of BOTH models (prepare_calib.py --color).

Examples:
  python eval_emulated_classifier.py --har classifier_q.har --test-dir .../classification/test \
      --classes classes.json --context fp_optimized --color bgr
  python eval_emulated_classifier.py --har classifier_q.har --test-dir ... --context quantized --color rgb
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

CONTEXTS = {"fp_optimized": InferenceContext.SDK_FP_OPTIMIZED,
            "quantized": InferenceContext.SDK_QUANTIZED}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", required=True, help="optimized HAR (fp_optimized needs optimize() done)")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--classes", required=True, help="classes.json (sorted class-folder names)")
    ap.add_argument("--context", choices=list(CONTEXTS), action="append", required=True)
    ap.add_argument("--color", choices=["bgr", "rgb"], required=True)
    ap.add_argument("--size", type=int, default=280)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    classes = json.load(open(args.classes))
    files = sorted(p for p in Path(args.test_dir).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    labels = [p.parent.name for p in files]
    data = np.stack([cv2.resize(cv2.imread(str(p)), (args.size, args.size),
                                interpolation=cv2.INTER_CUBIC) for p in files])
    if args.color == "rgb":
        data = data[..., ::-1]
    data = data.astype(np.float32)

    runner = ClientRunner(har=args.har)
    for ctx_name in args.context:
        with runner.infer_context(CONTEXTS[ctx_name]) as ctx:
            outs = [np.asarray(runner.infer(ctx, data[i:i + args.batch])).reshape(
                        min(args.batch, len(data) - i), -1)
                    for i in range(0, len(data), args.batch)]
        preds = [classes[i] for i in np.concatenate(outs).argmax(1)]
        per = defaultdict(lambda: [0, 0])
        for p, g in zip(preds, labels):
            per[g][0] += p == g
            per[g][1] += 1
        acc = np.mean([p == g for p, g in zip(preds, labels)])
        worst = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]
        print(f"{ctx_name:13s} color={args.color}  top-1 {acc:.4f}  (n={len(files)})")
        print("   worst: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in worst))


if __name__ == "__main__":
    main()
