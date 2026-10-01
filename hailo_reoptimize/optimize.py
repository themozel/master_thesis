#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Re-quantize a parsed HAR with stronger settings (Hailo DFC Python API).

Takes the alls you used for the original build (it holds the
normalization / input_conversion / nms_postprocess / logits_layer lines the
model needs), strips its optimization-related lines, appends the improved
ones below, runs runner.optimize() on the calibration .npy, and saves the
quantized HAR plus the exact alls used (next to it, for reproducibility).

Appended lines:
  model_optimization_flavor(optimization_level=<L>, compression_level=0)
  model_optimization_config(calibration, batch_size=<B>, calibset_size=<N>)
  post_quantization_optimization(finetune, policy=enabled, ...)   # --finetune
  quantization_param(<layer>, precision_mode=a16_w16)             # per --a16-layer
  <anything from --extra-line>

Needs an NVIDIA GPU for optimization_level >= 2 / finetune (the DFC falls
back or errors without one).

Examples:
    python optimize.py --har parsed.har --list-layers
    python optimize.py --har classifier_parsed.har --base-alls efficientnet_lite3_zeus_cropped.alls \
        --calib calib/classifier_rgb.npy --opt-level 2 --finetune \
        --a16-layer efficientnet_lite3_zeus_cropped/fc1 --out classifier_q.har
      
    Yolox:
    python optimize.py \
        --har yolox_m.har \
        --base-alls hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/yolox_m_leaky_zeus_zeuscropped.alls \
        --drop-line bgr_to_rgb --drop-line nms_postprocess \
        --calib det_bgr.npy \
        --opt-level 2 --finetune --ft-epochs 4 \
        --a16-layer yolox_m_zeus_cropped/conv74 --a16-layer yolox_m_zeus_cropped/conv75 --a16-layer yolox_m_zeus_cropped/conv76 \
        --a16-layer yolox_m_zeus_cropped/conv90 --a16-layer yolox_m_zeus_cropped/conv91 --a16-layer yolox_m_zeus_cropped/conv92 \
        --a16-layer yolox_m_zeus_cropped/conv105 --a16-layer yolox_m_zeus_cropped/conv106 --a16-layer yolox_m_zeus_cropped/conv107 \
        --extra-line 'nms_postprocess("master_thesis/hailo_reoptimize/yolox/nms_config_yolox_m_zeus_cropped.generated.json", yolox, engine=cpu)' \
        --extra-line "performance_param(compiler_optimization_level=max)" \
        --out master_thesis/hailo_reoptimize/yolox/har/yolox_m_optimized.har \
        --gpu



"""

import argparse
import os
import re
import sys

# The DFC picks its GPU on import and only takes one that is <5 % used, otherwise
# it silently falls back to CPU. --gpu <idx> forces it; this must run before the import.
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import numpy as np
from hailo_sdk_client import ClientRunner

STRIP = re.compile(r"^\s*(model_optimization_flavor|model_optimization_config\s*\(\s*calibration"
                   r"|post_quantization_optimization\s*\(\s*finetune)")


def build_alls(args, calib_size):
    base = open(args.base_alls).read().splitlines() if args.base_alls else []
    drop = [re.compile(p) for p in args.drop_line]
    kept = []
    for l in base:
        if STRIP.match(l):
            print(f"  [alls] replacing: {l.strip()}")
        elif any(p.search(l) for p in drop):
            print(f"  [alls] dropping:  {l.strip()}")
        else:
            kept.append(l)
    extra = [
        f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0)",
        f"model_optimization_config(calibration, batch_size={args.batch_size}, calibset_size={calib_size})",
    ]
    if args.finetune:
        extra.append(f"post_quantization_optimization(finetune, policy=enabled, "
                     f"dataset_size={calib_size}, epochs={args.ft_epochs}, "
                     f"learning_rate={args.ft_lr})")
    extra += [f"quantization_param({l}, precision_mode=a16_w16)" for l in args.a16_layer]
    extra += args.extra_line
    return "\n".join(kept + extra) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", required=True, help="parsed (not yet optimized) HAR")
    ap.add_argument("--list-layers", action="store_true",
                     help="Print layer names/types (to pick --a16-layer) and exit")
    ap.add_argument("--base-alls", help="alls used for the original build")
    ap.add_argument("--calib", help="calibration .npy from prepare_calib.py")
    ap.add_argument("--calib-size", type=int, default=None, help="use first N (default: all)")
    ap.add_argument("--opt-level", type=int, default=2, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--finetune", action="store_true", help="QFT (quantization-aware finetuning)")
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--a16-layer", action="append", default=[], help="Run this layer in 16 bit")
    ap.add_argument("--extra-line", action="append", default=[], help="Raw alls line to append")
    ap.add_argument("--drop-line", action="append", default=[],
                     help="Regex: drop matching lines of --base-alls (e.g. 'bgr_to_rgb' for yolox)")
    ap.add_argument("--gpu", help="GPU index to force (sets CUDA_VISIBLE_DEVICES), e.g. 0. "
                                  "Default: DFC auto-select, CPU fallback if no idle GPU")
    ap.add_argument("--out", help="output quantized HAR")
    args = ap.parse_args()

    import tensorflow as tf
    print(f"GPUs visible to TensorFlow: {tf.config.list_physical_devices('GPU') or 'none (running on CPU)'}")
    runner = ClientRunner(har=args.har)
    if args.list_layers:
        layers = runner.get_hn_dict()["layers"]
        for name, l in layers.items():
            print(f"{l.get('type', '?'):28s} {name}")
        return

    calib = np.load(args.calib, mmap_mode="r")
    if args.calib_size:
        calib = calib[: args.calib_size]
    calib = np.asarray(calib, dtype=np.float32)
    print(f"calibration set: {calib.shape}")

    alls = build_alls(args, len(calib))
    alls_out = args.out.rsplit(".", 1)[0] + ".alls"
    with open(alls_out, "w") as f:
        f.write(alls)
    print(f"model script ({alls_out}):\n{alls}")

    runner.load_model_script(alls)
    runner.optimize(calib)
    runner.save_har(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
