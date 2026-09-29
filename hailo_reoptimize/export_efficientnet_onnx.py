#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Step 1b: export the trained EfficientNet-Lite3 classifier to ONNX, and
measure the FLOAT accuracy of the ONNX on the test crops with a given
preprocessing. That pins down the exact normalization and color order that
the alls must reproduce on-device.

- The loader accepts a full pickled model, a plain state_dict, or a dict
  holding one ("model", "state_dict", "model_state_dict"); a "module."
  prefix (DataParallel) is stripped.
- Exports NCHW 1x3x<size>x<size>, float input already normalized, so the
  alls must add normalization(mean*255, std*255).
- --test-dir: accuracy of the ONNX with --mean/--std/--color/--interp. It
  should reproduce the PyTorch accuracy (~97%). If it doesn't, the
  preprocessing guess is wrong: check train_efficientnet_lite3.py's
  transforms and change the flags until it does.

Needs: torch, timm, onnx, onnxsim, onnxruntime, opencv.

Example:
  python export_efficientnet_onnx.py --ckpt output/best_model.pt --classes classes.json \
      --out onnx/efficientnet_lite3_zeus_cropped.onnx --test-dir $CLS_DS/test
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INTERP = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def load_model(ckpt_path, arch, num_classes):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj
    import timm
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    sd = obj
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if isinstance(obj, dict) and key in obj:
            sd = obj[key]
            break
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", required=True, help="classes.json (sorted class names)")
    ap.add_argument("--arch", default="tf_efficientnet_lite3")
    ap.add_argument("--size", type=int, default=280)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--test-dir", help="ImageFolder test crops, for ONNX float accuracy")
    ap.add_argument("--mean", type=float, nargs=3, default=None, help="0-1 scale; default: timm cfg")
    ap.add_argument("--std", type=float, nargs=3, default=None, help="0-1 scale; default: timm cfg")
    ap.add_argument("--color", choices=["rgb", "bgr"], default="rgb", help="order the model trained on")
    ap.add_argument("--interp", choices=list(INTERP), default="cubic")
    args = ap.parse_args()

    classes = json.load(open(args.classes))
    model = load_model(args.ckpt, args.arch, len(classes)).eval()
    cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = args.mean or list(cfg.get("mean", (0.5, 0.5, 0.5)))
    std = args.std or list(cfg.get("std", (0.5, 0.5, 0.5)))
    print(f"timm cfg mean={cfg.get('mean')} std={cfg.get('std')}  -> using mean={mean} std={std}")

    dummy = torch.randn(1, 3, args.size, args.size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["logits"],
                      opset_version=args.opset)
    from onnxsim import simplify
    m, ok = simplify(onnx.load(args.out))
    assert ok, "onnxsim failed"
    onnx.save(m, args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out)
    with torch.no_grad():
        ref = model(dummy).numpy()
    print(f"Wrote {args.out}; ONNX vs PyTorch max abs diff: "
          f"{np.abs(ref - sess.run(None, {'input': dummy.numpy()})[0]).max():.2e}")

    m255, s255 = [round(v * 255, 3) for v in mean], [round(v * 255, 3) for v in std]
    print("\nalls normalization line for this preprocessing (0-255 input):\n"
          f"  normalization1 = normalization({m255}, {s255})")

    if not args.test_dir:
        return
    files = sorted(p for p in Path(args.test_dir).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    mean_a, std_a = np.array(mean, np.float32), np.array(std, np.float32)
    correct = 0
    for f in files:
        img = cv2.imread(str(f))
        if args.color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.size, args.size), interpolation=INTERP[args.interp])
        x = ((img.astype(np.float32) / 255 - mean_a) / std_a).transpose(2, 0, 1)[None]
        pred = classes[int(sess.run(None, {"input": x})[0].argmax())]
        correct += pred == f.parent.name
    print(f"\nONNX float top-1 on {args.test_dir}: {correct / len(files):.4f} ({correct}/{len(files)}) "
          f"[color={args.color} interp={args.interp}]")


if __name__ == "__main__":
    main()
