# import os

# os.system("""
# python yolov7/train.py \
# --img 640 \
# --batch 16 \
# --epochs 50 \
# --data data/PERCEPT/data.yaml \
# --cfg yolov7/cfg/training/yolov7.yaml \
# --weights yolov7.pt \
# --device 0
# """)

import os
import runpy
import sys
import torch

# Add yolov7 directory to path so its internal imports resolve
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov7"),
)

# PyTorch >= 2.6 changed torch.load default to weights_only=True, which breaks
# YOLOv7 checkpoints that contain custom classes. Patch it to default to False.
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    # Legacy-format .pt files default to encoding='bytes' which causes
    # "STACK_GLOBAL requires str"; latin1 preserves byte values as str.
    kwargs.setdefault("encoding", "latin1")
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

sys.argv = [
    "yolov7/train.py",
    "--img",
    "640",
    "--batch",
    "16",
    "--epochs",
    "300",
    "--data",
    "data/GERALD/data.yaml",
    # "--cfg",
    # "yolov7/cfg/training/yolov7.yaml",
    "--weights",
    "yolov7.pt",
    "--device",
    "0",
]

runpy.run_path("yolov7/train.py", run_name="__main__")
