import torch

model = torch.hub.load(
    "ultralytics/yolov5", "custom", "yolov5/runs/train/exp6/weights/best.pt"
)

results = model("percept_test.png")
results.show()
results.print()
results.save()
