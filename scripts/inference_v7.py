import os

IMAGE_PATH = "data/images/test/sample.jpg"
WEIGHTS = "runs/train/exp/weights/best.pt"

os.system(
    f"""
python yolov7/detect.py \
--weights {WEIGHTS} \
--source {IMAGE_PATH} \
--conf 0.25 \
--img-size 640 \
--device 0
"""
)
