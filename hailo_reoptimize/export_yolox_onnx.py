#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Step 1a: export the trained YOLOX checkpoint to ONNX for the Hailo parser.

- Exported with decode_in_inference=False (raw head outputs). Box decoding
  and NMS are done by the HEF's nms_postprocess instead.
- The input is raw 0-255 pixels, no mean/std (YOLOX >= 0.2 preprocessing).
  Channel order is whatever YOLOX trained on: BGR, since it reads with cv2.
- Also writes <out>.end_nodes.json: the 9 head outputs the Hailo parser
  must stop at, per stride 8/16/32 in the order [reg Conv, obj Sigmoid,
  cls Sigmoid]. They're found as the inputs of the per-level Concat that
  YOLOX's head builds (cat([reg, obj.sigmoid(), cls.sigmoid()])).
- Checks ONNX Runtime against PyTorch on a random input.

Needs: the YOLOX repo installed (pip install -e YOLOX), onnx, onnxsim,
onnxruntime, and your exp file (the one training used, with act="lrelu").

Example:
  python export_yolox_onnx.py -f exps/yolox_m_leaky_zeus.py \
      -c YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth --out onnx/yolox_m_leaky_zeus.onnx
"""

import argparse
import json
import os

import numpy as np
import onnx
import torch
from torch import nn


def find_head_end_nodes(model_proto):
    """Returns [reg, obj, cls] producer-node names for each output level."""
    producer = {o: n for n in model_proto.graph.node for o in n.output}
    levels = []
    for n in model_proto.graph.node:
        if n.op_type != "Concat" or len(n.input) != 3:
            continue
        prods = [producer.get(i) for i in n.input]
        if all(prods) and [p.op_type for p in prods] == ["Conv", "Sigmoid", "Sigmoid"]:
            levels.append([p.name for p in prods])
    return levels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    exp = get_exp(args.exp_file, None)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    model = replace_module(model, nn.SiLU, SiLU)  # no-op for the leaky variant, export-safe otherwise
    model.eval()
    model.head.decode_in_inference = False

    h, w = exp.test_size
    dummy = torch.rand(1, 3, h, w) * 255
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(model, dummy, args.out, input_names=["images"], output_names=["output"],
                      opset_version=args.opset)
    m = onnx.load(args.out)
    if not args.no_simplify:
        from onnxsim import simplify
        m, ok = simplify(m)
        assert ok, "onnxsim failed"
        onnx.save(m, args.out)
    print(f"Wrote {args.out} (input images 1x3x{h}x{w}, {exp.num_classes} classes)")

    import onnxruntime as ort
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = ort.InferenceSession(args.out).run(None, {"images": dummy.numpy()})[0]
    print(f"ONNX vs PyTorch max abs diff: {np.abs(ref - got).max():.2e}")

    levels = find_head_end_nodes(m)
    if len(levels) != 3:
        print(f"[warn] expected 3 head levels, found {len(levels)}: {levels}\n"
              "       set --end-nodes manually in parse_onnx.py (open the ONNX in netron)")
    nodes_path = args.out.rsplit(".", 1)[0] + ".end_nodes.json"
    with open(nodes_path, "w") as f:
        json.dump({"start_nodes": ["images"], "end_nodes": [n for lvl in levels for n in lvl],
                   "input_shape": [1, 3, h, w], "num_classes": exp.num_classes}, f, indent=2)
    print(f"Wrote {nodes_path}:")
    for stride, lvl in zip((8, 16, 32), levels):
        print(f"  stride {stride}: reg={lvl[0]}  obj={lvl[1]}  cls={lvl[2]}")


if __name__ == "__main__":
    main()
