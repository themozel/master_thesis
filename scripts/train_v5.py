import os

os.system(
    """
python yolov5/train.py \
--img 640 \
--batch 16 \
--epochs 50 \
--data data.yaml \
--weights yolov5s.pt \
--device 0
"""
)
