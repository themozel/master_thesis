import torch

model = torch.hub.load("ultralytics/yolov5", "custom", "best.pt")

results = model("test.jpg")
results.show()
