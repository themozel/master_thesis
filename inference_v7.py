import torch
import cv2

model = torch.hub.load("WongKinYiu/yolov7", "custom", "best.pt")

img = cv2.imread("test.jpg")
results = model(img)

results.print()
results.show()
