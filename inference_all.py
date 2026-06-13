import os
import time
from pathlib import Path

import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# CONFIGURATION
# ============================================================


TEST_IMAGES_DIR = "data/PERCEPT/images/test"
TEST_LABELS_DIR = "data/PERCEPT/labels/test"

CONF_THRESHOLD = 0.25
IMG_SIZE = 640
DEVICE = "cuda:0"  # use "cpu" if needed

MODEL = "yolov5"
WEIGHTS_PATH = "yolov5/runs/train/exp8/weights/best.pt"
# WEIGHTS_PATH = f"runs/train/exp/weights/best.pt"
OUTPUT_DIR = f"inference_results/{MODEL}s6"

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

# PyTorch 2.6+ changed weights_only default to True, which breaks YOLOv7/YOLOv5
# checkpoints that contain numpy arrays. Patch torch.load temporarily.
import functools
_orig_torch_load = torch.load
torch.load = functools.partial(_orig_torch_load, weights_only=False)

try:
    model = torch.hub.load(f"{MODEL}", "custom", WEIGHTS_PATH, source="local")
finally:
    torch.load = _orig_torch_load

model.conf = CONF_THRESHOLD
model.imgsz = IMG_SIZE
model.to(DEVICE)
print(model.names)

print("Model loaded successfully.")


# ============================================================
# GET TEST IMAGES
# ============================================================

image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

image_paths = []

for ext in image_extensions:
    image_paths.extend(Path(TEST_IMAGES_DIR).glob(f"*{ext}"))

image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} test images.")

# ============================================================
# METRICS STORAGE
# ============================================================

latencies = []
results_summary = []

all_gt = []
all_pred = []

# ============================================================
# INFERENCE LOOP
# ============================================================

for idx, image_path in enumerate(image_paths):

    print(f"[{idx+1}/{len(image_paths)}] Processing: {image_path.name}")

    # --------------------------------------------------------
    # INFERENCE TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    results = model(str(image_path))

    #### Overwriting preds with specific classes only for evaluation purposes ####
    TARGET_CLASSES = list(range(8, 27))
    pred = results.pred[0]
    mask = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
    for cls_id in TARGET_CLASSES:
        mask |= pred[:, 5].int() == cls_id
    filtered_pred = pred[mask]
    results.pred[0] = filtered_pred
    results.xyxy[0] = filtered_pred
    #######

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0

    latencies.append(latency_ms)

    # --------------------------------------------------------
    # SAVE PREDICTION IMAGE
    # --------------------------------------------------------

    # OpenCV 4.11+ requires writable arrays; YOLOv7 may return read-only views
    results.imgs = [img.copy() for img in results.imgs]
    rendered_image = results.render()[0]

    output_image_path = os.path.join(OUTPUT_DIR, "predictions", image_path.name)

    cv2.imwrite(output_image_path, rendered_image)

    # --------------------------------------------------------
    # EXTRACT PREDICTIONS
    # --------------------------------------------------------

    # TARGET_CLASSES = list(range(8, 27))

    detections = results.xyxy[0]

    filtered_detections = detections[
        torch.isin(
            detections[:, 5].int(), torch.tensor(TARGET_CLASSES).to(detections.device)
        )
    ]

    # results.xyxy[0] = filtered_detections

    # predicted_classes = detections[:, 5].int().tolist()

    # --------------------------------------------------------
    # LOAD GROUND TRUTH LABELS
    # --------------------------------------------------------

    label_path = Path(TEST_LABELS_DIR) / f"{image_path.stem}.txt"

    gt_classes = []

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f.readlines():

                cls_id = int(line.strip().split()[0])

                gt_classes.append(cls_id)

    # --------------------------------------------------------
    # SIMPLE CLASSIFICATION MATCHING
    # --------------------------------------------------------

    # NOTE:
    # This is simplified accuracy evaluation.
    # For full object detection mAP evaluation,
    # use YOLOv5 val.py

    gt_present = 1 if len(gt_classes) > 0 else 0
    # pred_present = 1 if len(predicted_classes) > 0 else 0
    filtered_detections = 1 if len(filtered_detections) > 0 else 0

    all_gt.append(gt_present)
    # all_pred.append(pred_present)
    all_pred.append(filtered_detections)

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results_summary.append(
        {
            "image": image_path.name,
            "latency_ms": latency_ms,
            # "num_predictions": len(predicted_classes),
            "num_predictions": filtered_detections,
            "ground_truth_objects": len(gt_classes),
        }
    )

# ============================================================
# CALCULATE METRICS
# ============================================================

average_latency = sum(latencies) / len(latencies)

precision = precision_score(all_gt, all_pred, zero_division=0)
recall = recall_score(all_gt, all_pred, zero_division=0)
f1 = f1_score(all_gt, all_pred, zero_division=0)

accuracy = sum([1 for gt, pred in zip(all_gt, all_pred) if gt == pred]) / len(all_gt)

# ============================================================
# SAVE SUMMARY TXT
# ============================================================

summary_txt = f"""
=============================
{MODEL} INFERENCE SUMMARY
=============================

Images processed: {len(image_paths)}

Average latency (ms): {average_latency:.2f}

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}
"""

with open(f"{OUTPUT_DIR}/summary.txt", "w") as f:
    f.write(summary_txt)

print(summary_txt)

# ============================================================
# SAVE CSV RESULTS
# ============================================================

df = pd.DataFrame(results_summary)

csv_path = f"{OUTPUT_DIR}/detailed_results.csv"

df.to_csv(csv_path, index=False)

print(f"Saved CSV results to: {csv_path}")

# ============================================================
# LATENCY PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(latencies)

plt.xlabel("Image Index")
plt.ylabel("Latency (ms)")
plt.title("Inference Latency per Image")

latency_plot_path = f"{OUTPUT_DIR}/plots/latency_plot.png"

plt.savefig(latency_plot_path)

print(f"Saved latency plot: {latency_plot_path}")

# ============================================================
# LATENCY HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(latencies, bins=20)

plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.title("Latency Distribution")

hist_plot_path = f"{OUTPUT_DIR}/plots/latency_histogram.png"

plt.savefig(hist_plot_path)

print(f"Saved histogram plot: {hist_plot_path}")

# ============================================================
# METRICS BAR PLOT
# ============================================================

metrics_names = ["Accuracy", "Precision", "Recall", "F1"]

metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))

plt.bar(metrics_names, metrics_values)

plt.ylim([0, 1])

plt.title("Detection Metrics")

metrics_plot_path = f"{OUTPUT_DIR}/plots/metrics_plot.png"

plt.savefig(metrics_plot_path)

print(f"Saved metrics plot: {metrics_plot_path}")

# ============================================================
# FINAL PRINT
# ============================================================

print("Inference completed successfully.")
