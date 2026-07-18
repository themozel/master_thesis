import os

os.system("""
python yolov5/train.py \
--img 640 \
--batch 16 \
--epochs 50 \
--data data/PERCEPT/images_signal_only/data.yaml \
--weights yolov5s6.pt \
--device 0
""")
