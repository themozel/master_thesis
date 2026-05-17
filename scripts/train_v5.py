import os

os.system("""
python yolov5/train.py \
--img 640 \
--batch 32 \
--epochs 50 \
--data data/PERCEPT/data.yaml \
--weights yolov5s.pt \
--device 0
""")
