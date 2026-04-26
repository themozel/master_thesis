import os

os.system(
    """
python yolov7/train.py \
--img 640 \
--batch 16 \
--epochs 50 \
--data data.yaml \
--cfg yolov7/cfg/training/yolov7.yaml \
--weights yolov7.pt \
--device 0
"""
)
