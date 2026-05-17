import os

os.system("""
python yolov5/detect.py \
--weights runs/train/exp/weights/best.pt \
--source data/images/test/sample.jpg \
--conf 0.25 \
--img 640 \
--device 0
""")
