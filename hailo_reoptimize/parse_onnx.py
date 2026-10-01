#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Step 2: ONNX -> parsed HAR (Hailo DFC, runner.translate_onnx_model).

Keep --net-name identical to the original build
(yolox_m_leaky_zeus_zeuscropped / efficientnet_lite3_zeus_cropped). Layer
names in the alls (e.g. efficientnet_lite3_zeus_cropped/fc1) and the vstream
names the runtime sees are derived from it.

The detector must stop at the raw head outputs, so pass the
<onnx>.end_nodes.json written by export_yolox_onnx.py. The classifier needs
no start/end nodes.

--write-nms-config (detector only): writes the nms_postprocess config json
that the alls points at. It maps each ONNX end node to the HAR layer it
became (the HN layer's "original_names") to fill in bbox_decoders. Prefer
your ORIGINAL nms_config_yolox_m_leaky_zeus_zeuscropped.json if you still
have it; use this only if it's lost, and compare the layer names against it.

Examples:
  python parse_onnx.py --onnx onnx/yolox_m_leaky_zeus.onnx \
      --net-name yolox_m_leaky_zeus_zeuscropped --nodes onnx/yolox_m_leaky_zeus.end_nodes.json \
      --out har/yolox_m_parsed.har
      
python master_thesis/hailo_reoptimize/parse_onnx.py \
  --onnx master_thesis/hailo_reoptimize/yolox/onnx/yolox_m_zeus_cropped.onnx \
  --net-name yolox_m_zeus_cropped \
  --nodes master_thesis/hailo_reoptimize/yolox/onnx/yolox_m_zeus_cropped.end_nodes.json \
  --write-nms-config master_thesis/hailo_reoptimize/yolox/nms_config_yolox_m_zeus_cropped.generated.json \
  --out master_thesis/hailo_reoptimize/yolox/har/yolox_m_parsed.har     
      
      
      
      
  python parse_onnx.py --onnx onnx/efficientnet_lite3_zeus_cropped.onnx \
      --net-name efficientnet_lite3_zeus_cropped --out har/efficientnet_parsed.har
"""

import argparse
import json
import os

from hailo_sdk_client import ClientRunner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--net-name", required=True)
    ap.add_argument("--nodes", help="end_nodes.json from export_yolox_onnx.py")
    ap.add_argument("--end-node", action="append", default=[], help="manual end node (repeatable)")
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--write-nms-config", help="detector: write yolox NMS config json here")
    ap.add_argument("--nms-score-th", type=float, default=0.2)
    ap.add_argument("--nms-iou-th", type=float, default=0.65)
    ap.add_argument("--max-proposals", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kwargs = {}
    if args.nodes:
        nodes = json.load(open(args.nodes))
        kwargs = {"start_node_names": nodes["start_nodes"], "end_node_names": nodes["end_nodes"],
                  "net_input_shapes": {nodes["start_nodes"][0]: nodes["input_shape"]}}
    if args.end_node:
        kwargs["end_node_names"] = args.end_node

    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(args.onnx, args.net_name, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    runner.save_har(args.out)
    print(f"Wrote {args.out}")
    outs = [n for n, l in runner.get_hn_dict()["layers"].items() if l.get("type") == "output_layer"]
    print(f"output layers ({len(outs)}): {outs}")

    if args.write_nms_config:
        write_nms_config(runner, nodes, args)


def write_nms_config(runner, nodes, args):
    layers = runner.get_hn_dict()["layers"]
    onnx_to_hn = {}
    for name, l in layers.items():
        for orig in l.get("original_names", []):
            onnx_to_hn[orig] = name.split("/", 1)[-1]  # model-zoo configs use names without net prefix
    missing = [n for n in nodes["end_nodes"] if n not in onnx_to_hn]
    if missing:
        raise SystemExit(f"no HAR layer found for ONNX nodes {missing}; write the json by hand")
    h, w = nodes["input_shape"][2:]
    decoders = []
    for i, stride in enumerate((8, 16, 32)):
        reg, obj, cls = (onnx_to_hn[n] for n in nodes["end_nodes"][3 * i: 3 * i + 3])
        decoders.append({"name": f"bbox_decoder{i}", "stride": stride,
                         "reg_layer": reg, "objectness_layer": obj, "cls_layer": cls})
    cfg = {"nms_scores_th": args.nms_score_th, "nms_iou_th": args.nms_iou_th,
           "image_dims": [h, w], "max_proposals_per_class": args.max_proposals,
           "classes": nodes["num_classes"], "background_removal": False,
           "background_removal_index": 0, "bbox_decoders": decoders}
    with open(args.write_nms_config, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Wrote {args.write_nms_config}:\n{json.dumps(decoders, indent=2)}")


if __name__ == "__main__":
    main()
