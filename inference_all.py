"""
Run batch inference on a labeled test split and export latency + classification-style metrics.

This script is framework-agnostic for local Torch Hub repos and supports both YOLOv5 and YOLOv7
through command-line arguments.

What it does:
1. Loads a custom YOLO model checkpoint from local weights.
2. Runs inference over all test images in TEST_IMAGES_DIR.
3. Optionally filters predictions to TARGET_CLASSES for evaluation.
4. Computes image-level metrics (accuracy/precision/recall/F1) based on object presence.
5. Saves summary text, per-image CSV, rendered predictions, and plots.

Quick usage:
1. Choose model + weights + dataset paths.
2. Run with arguments from project root.

Run command examples from project root:
- YOLOv5:
        python inference_all.py \
            --model yolov5 \
            --weights yolov5/runs/train/yolov5s6_20260614_signals_only/weights/best.pt \
            --test-images data/PERCEPT/images_signal_only/images/test \
            --test-labels data/PERCEPT/images_signal_only/labels/test \
            --output-dir inference_results/yolov5s6_20260614_signals_only

- YOLOv7:
        python inference_all.py \
            --model yolov7 \
            --weights yolov7/runs/train/exp/weights/best.pt \
            --test-images data/PERCEPT/images_signal_only/images/test \
            --test-labels data/PERCEPT/images_signal_only/labels/test \
            --output-dir inference_results/yolov7_exp

YOLOv5 setup example:
- MODEL = "yolov5"
- WEIGHTS_PATH = "yolov5/runs/train/<experiment>/weights/best.pt"
- OUTPUT_DIR = "inference_results/<run_name>"

YOLOv7 setup example:
- MODEL = "yolov7"
- WEIGHTS_PATH = "yolov7/runs/train/<experiment>/weights/best.pt"
    (or another valid YOLOv7 best.pt path)
- OUTPUT_DIR = "inference_results/<run_name>"

Notes:
- Default target classes are 0 1 2 3. Override with --target-classes.
- Metrics here are image-level presence metrics, not mAP. Use val.py/test.py for full detection metrics.
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def yolo_xywhn_to_xyxy_abs(x_center, y_center, width, height, img_w, img_h):
    """Convert normalized YOLO box format to absolute XYXY coordinates."""
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    return [x1, y1, x2, y2]


def box_iou_xyxy(box1, box2):
    """Compute IoU for two XYXY boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def compute_ap_from_precision_recall(recalls, precisions):
    """Compute AP using COCO-style 101-point interpolation."""
    if len(recalls) == 0:
        return 0.0

    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_levels:
        mask = recalls >= r
        p = np.max(precisions[mask]) if np.any(mask) else 0.0
        ap += p

    return ap / len(recall_levels)


def evaluate_map(predictions, ground_truths, class_ids, iou_thresholds):
    """Evaluate AP@0.50 and AP@[0.50:0.95] on collected predictions/labels."""
    gt_by_class_image = {cls_id: {} for cls_id in class_ids}
    pred_by_class = {cls_id: [] for cls_id in class_ids}

    for gt in ground_truths:
        cls_id = gt["class_id"]
        if cls_id not in gt_by_class_image:
            continue
        gt_by_class_image[cls_id].setdefault(gt["image_id"], []).append(gt["bbox"])

    for pred in predictions:
        cls_id = pred["class_id"]
        if cls_id not in pred_by_class:
            continue
        pred_by_class[cls_id].append(pred)

    per_class_ap50 = {}
    per_class_ap50_95 = {}
    ap50_values = []
    ap50_95_values = []

    for cls_id in class_ids:
        class_preds = sorted(pred_by_class[cls_id], key=lambda x: x["score"], reverse=True)
        class_gts = gt_by_class_image[cls_id]
        npos = sum(len(v) for v in class_gts.values())

        if npos == 0:
            per_class_ap50[cls_id] = None
            per_class_ap50_95[cls_id] = None
            continue

        class_ap_per_iou = []

        for thr in iou_thresholds:
            matched = {
                img_id: np.zeros(len(boxes), dtype=bool)
                for img_id, boxes in class_gts.items()
            }

            tp = np.zeros(len(class_preds), dtype=float)
            fp = np.zeros(len(class_preds), dtype=float)

            for i, pred in enumerate(class_preds):
                img_id = pred["image_id"]
                pred_box = pred["bbox"]
                gt_boxes = class_gts.get(img_id, [])

                if not gt_boxes:
                    fp[i] = 1.0
                    continue

                ious = np.array([box_iou_xyxy(pred_box, gt_box) for gt_box in gt_boxes], dtype=float)
                best_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
                best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0

                if best_iou >= thr and not matched[img_id][best_idx]:
                    tp[i] = 1.0
                    matched[img_id][best_idx] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recalls = tp_cum / max(npos, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

            class_ap_per_iou.append(compute_ap_from_precision_recall(recalls, precisions))

        per_class_ap50[cls_id] = class_ap_per_iou[0]
        per_class_ap50_95[cls_id] = float(np.mean(class_ap_per_iou))

        ap50_values.append(class_ap_per_iou[0])
        ap50_95_values.append(float(np.mean(class_ap_per_iou)))

    summary = {
        "AP@[0.50:0.95]": float(np.mean(ap50_95_values)) if ap50_95_values else 0.0,
        "AP@0.50": float(np.mean(ap50_values)) if ap50_values else 0.0,
    }

    return summary, per_class_ap50, per_class_ap50_95

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for local YOLOv5/YOLOv7 repos.")

    parser.add_argument("--model", choices=["yolov5", "yolov7"], required=True,
                        help="Local model repo directory to load via torch.hub.")
    parser.add_argument("--weights", required=True,
                        help="Path to model weights (best.pt).")
    parser.add_argument("--test-images", required=True,
                        help="Directory containing test images.")
    parser.add_argument("--test-labels", required=True,
                        help="Directory containing YOLO-format label txt files.")
    parser.add_argument("--output-dir", default="",
                        help="Output directory for reports, plots, and predictions.")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Confidence threshold used by the model.")
    parser.add_argument("--device", default="cuda:0",
                        help="Inference device, for example cuda:0 or cpu.")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size.")
    parser.add_argument("--target-classes", type=int, nargs="+", default=None,
                        help="Optional class IDs used for evaluation filtering. Defaults to all classes.")
    parser.add_argument("--save-only-target-classes", action="store_true",
                        help="Save rendered predictions only for images with target-class detections.")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if not output_dir:
        weights_stem = Path(args.weights).stem
        output_dir = f"inference_results/{args.model}_{weights_stem}"

    # ============================================================
    # CREATE OUTPUT DIRECTORIES
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)

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
        model = torch.hub.load(args.model, "custom", args.weights, source="local")
    finally:
        torch.load = _orig_torch_load

    model.conf = args.conf_threshold
    model.imgsz = args.img_size
    model.to(args.device)
    print(model.names)

    print("Model loaded successfully.")

    # ============================================================
    # GET TEST IMAGES
    # ============================================================

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(args.test_images).glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} test images.")

    if not image_paths:
        print("No images found. Exiting without running inference.")
        return

    # ============================================================
    # METRICS STORAGE
    # ============================================================

    latencies = []
    results_summary = []

    all_gt = []
    all_pred = []

    detection_predictions = []
    detection_ground_truths = []

    if args.target_classes is None:
        if isinstance(model.names, dict):
            target_classes = sorted(int(k) for k in model.names.keys())
        else:
            target_classes = list(range(len(model.names)))
    else:
        target_classes = args.target_classes

    print(f"Evaluating target classes: {target_classes}")
    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    # ============================================================
    # INFERENCE LOOP
    # ============================================================

    for idx, image_path in enumerate(image_paths):

        print(f"[{idx+1}/{len(image_paths)}] Processing: {image_path.name}")

    # --------------------------------------------------------
    # INFERENCE TIMER
    # --------------------------------------------------------

        start_time = time.perf_counter()

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read image {image_path}, skipping.")
            continue

        img_h, img_w = image.shape[:2]

        results = model(str(image_path))

        #### Overwriting preds with specific classes only for evaluation purposes ####
        pred = results.pred[0]
        mask = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
        for cls_id in target_classes:
            mask |= pred[:, 5].int() == cls_id
        filtered_pred = pred[mask]
        results.pred[0] = filtered_pred
        results.xyxy[0] = filtered_pred
        #######

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0

        latencies.append(latency_ms)

    # --------------------------------------------------------
    # EXTRACT PREDICTIONS
    # --------------------------------------------------------

        detections = results.xyxy[0]

        filtered_detections = detections[
            torch.isin(
                detections[:, 5].int(), torch.tensor(target_classes).to(detections.device)
            )
        ]

        for det in filtered_detections.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det[:6]
            detection_predictions.append(
                {
                    "image_id": image_path.name,
                    "class_id": int(cls_id),
                    "score": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    # --------------------------------------------------------
    # SAVE PREDICTION IMAGE
    # When SAVE_ONLY_TARGET_CLASSES=True, only saves images that have at
    # least one detection in TARGET_CLASSES (classes 21-27).
    # Set SAVE_ONLY_TARGET_CLASSES=False to save every image.
    # --------------------------------------------------------

        if not args.save_only_target_classes or len(filtered_detections) > 0:
            # OpenCV 4.11+ requires writable arrays; different hubs expose ims or imgs.
            if hasattr(results, "ims") and results.ims is not None:
                results.ims = [img.copy() for img in results.ims]
            elif hasattr(results, "imgs") and results.imgs is not None:
                results.imgs = [img.copy() for img in results.imgs]
            rendered_image = results.render()[0]

            output_image_path = os.path.join(output_dir, "predictions", image_path.name)

            cv2.imwrite(output_image_path, rendered_image)

    # --------------------------------------------------------
    # LOAD GROUND TRUTH LABELS
    # --------------------------------------------------------

        label_path = Path(args.test_labels) / f"{image_path.stem}.txt"

        gt_classes = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f.readlines():

                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])

                    gt_classes.append(cls_id)

                    if cls_id in target_classes:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bbox_xyxy = yolo_xywhn_to_xyxy_abs(
                            x_center, y_center, width, height, img_w, img_h
                        )
                        detection_ground_truths.append(
                            {
                                "image_id": image_path.name,
                                "class_id": cls_id,
                                "bbox": bbox_xyxy,
                            }
                        )

    # --------------------------------------------------------
    # SIMPLE CLASSIFICATION MATCHING
    # --------------------------------------------------------

    # NOTE:
    # This is simplified accuracy evaluation.
    # For full object detection mAP evaluation,
    # use YOLOv5 val.py

        gt_present = 1 if len(gt_classes) > 0 else 0
        # pred_present = 1 if len(predicted_classes) > 0 else 0
        pred_present = 1 if len(filtered_detections) > 0 else 0

        all_gt.append(gt_present)
        all_pred.append(pred_present)

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

        results_summary.append(
            {
                "image": image_path.name,
                "latency_ms": latency_ms,
                "num_predictions": len(filtered_detections),
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

    map_summary, per_class_ap50, per_class_ap50_95 = evaluate_map(
        detection_predictions,
        detection_ground_truths,
        target_classes,
        iou_thresholds,
    )

    # ============================================================
    # SAVE SUMMARY TXT
    # ============================================================

    summary_txt = f"""
=============================
{args.model} INFERENCE SUMMARY
=============================

Images processed: {len(image_paths)}

Average latency (ms): {average_latency:.2f}

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}

Detection AP@0.50:      {map_summary['AP@0.50']:.4f}
Detection AP@0.50:0.95: {map_summary['AP@[0.50:0.95]']:.4f}
"""

    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(summary_txt)

    print(summary_txt)

    # ============================================================
    # SAVE CSV RESULTS
    # ============================================================

    df = pd.DataFrame(results_summary)

    csv_path = f"{output_dir}/detailed_results.csv"

    df.to_csv(csv_path, index=False)

    print(f"Saved CSV results to: {csv_path}")

    metrics_report = {
        "summary_metrics": map_summary,
        "per_class_ap50": {str(k): v for k, v in per_class_ap50.items()},
        "per_class_ap50_95": {str(k): v for k, v in per_class_ap50_95.items()},
        "num_images_processed": len(image_paths),
        "num_detections": len(detection_predictions),
        "target_classes": target_classes,
    }
    metrics_json_path = f"{output_dir}/metrics_report.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Saved detection metrics JSON to: {metrics_json_path}")

    # ============================================================
    # LATENCY PLOT
    # ============================================================

    plt.figure(figsize=(10, 5))

    plt.plot(latencies)

    plt.xlabel("Image Index")
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency per Image")

    latency_plot_path = f"{output_dir}/plots/latency_plot.png"

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

    hist_plot_path = f"{output_dir}/plots/latency_histogram.png"

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

    metrics_plot_path = f"{output_dir}/plots/metrics_plot.png"

    plt.savefig(metrics_plot_path)

    print(f"Saved metrics plot: {metrics_plot_path}")

    # ============================================================
    # FINAL PRINT
    # ============================================================

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
