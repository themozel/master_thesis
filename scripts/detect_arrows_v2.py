"""
Detect directional arrow plates in signal images and copy matches
to data/PERCEPT/images_signal_only/with_arrows/.

Arrow plates are small rectangular white/light-gray signs with dark arrow symbols,
typically found near/below railroad signals.

Detection strategy (fast):
1. Use YOLO label bounding boxes to locate signal regions.
2. Expand search area around each signal (arrows are mounted nearby).
3. In expanded ROI, find small bright rectangular plates via thresholding.
4. Inside each plate candidate, detect dark arrow shapes using solidity
   and convexity-defect analysis.

VERSION 2: Optimized version - no full-image grid scan, only searches near
signal bounding boxes. Returns boolean (no box coordinates).
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path


def get_signal_boxes(label_path, img_w, img_h, target_class_ids):
    """Read YOLO label file and return bounding boxes for target signal classes."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in target_class_ids:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append((x1, y1, x2, y2))
    return boxes


def has_arrow_plate(gray_roi):
    """
    Fast check: does this grayscale ROI contain a small bright plate
    with a dark arrow shape inside?
    """
    h, w = gray_roi.shape[:2]
    if h < 20 or w < 20:
        return False

    # Find bright regions (potential plate backgrounds)
    # Try multiple thresholds for robustness
    for thresh_val in (170, 190, 210):
        _, bright = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological close to merge nearby bright pixels into plates
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Plate should be small-to-medium sized (roughly 15x15 to 80x80 pixels)
            if area < 150 or area > 6000:
                continue

            # Check rectangularity
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 1.8:
                continue

            # Bounding rect of this plate candidate
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 12 or bh < 12:
                continue

            plate = gray_roi[by:by+bh, bx:bx+bw]

            # Threshold dark content inside the plate
            _, dark_mask = cv2.threshold(plate, 100, 255, cv2.THRESH_BINARY_INV)
            dark_ratio = np.count_nonzero(dark_mask) / (bw * bh)
            if dark_ratio < 0.10 or dark_ratio > 0.55:
                continue

            # Find dark shape contours
            dark_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for dc in dark_cnts:
                dc_area = cv2.contourArea(dc)
                if dc_area < 40:
                    continue

                hull = cv2.convexHull(dc)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue

                solidity = dc_area / hull_area
                # Arrows: 0.4-0.85 (not a filled rectangle, not too fragmented)
                if solidity < 0.35 or solidity > 0.88:
                    continue

                # Check convexity defects for the "notch" shape of arrows
                hull_idx = cv2.convexHull(dc, returnPoints=False)
                if len(hull_idx) < 3 or len(dc) < 5:
                    continue
                try:
                    defects = cv2.convexityDefects(dc, hull_idx)
                except cv2.error:
                    continue
                if defects is None:
                    continue

                sig_defects = 0
                for i in range(defects.shape[0]):
                    _, _, _, dist = defects[i, 0]
                    if dist > 400:
                        sig_defects += 1

                if sig_defects >= 1:
                    # Shape should have some elongation (arrow-like)
                    dx, dy, dw, dh = cv2.boundingRect(dc)
                    dc_aspect = max(dw, dh) / max(min(dw, dh), 1)
                    if dc_aspect >= 1.1:
                        return True
    return False


def image_has_arrow(img, signal_boxes):
    """Check if image contains arrow plates near any labeled signal."""
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (x1, y1, x2, y2) in signal_boxes:
        box_w = x2 - x1
        box_h = y2 - y1
        # Generous expansion around signal (arrows are typically nearby)
        margin_x = max(box_w * 2, 80)
        margin_y = max(box_h * 3, 120)

        sx1 = max(0, x1 - margin_x)
        sy1 = max(0, y1 - margin_y)
        sx2 = min(img_w, x2 + margin_x)
        sy2 = min(img_h, y2 + margin_y)

        roi = gray[sy1:sy2, sx1:sx2]
        if has_arrow_plate(roi):
            return True

    return False


def main():
    base_dir = Path(__file__).resolve().parent.parent
    images_dir = base_dir / "data" / "PERCEPT" / "images_signal_only"
    labels_dir = base_dir / "data" / "PERCEPT" / "labels"
    output_dir = images_dir / "with_arrows"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Signal class IDs: sig_free_left=8, sig_free_right=9, sig_free_straight=10, sig_stop=20
    target_class_ids = {8, 9, 10, 20}

    # Gather all images (excluding with_arrows subfolder)
    image_files = []
    for split in ["train", "val", "test"]:
        split_dir = images_dir / split
        if split_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                image_files.extend(split_dir.glob(ext))
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_files.extend(f for f in images_dir.glob(ext) if f.is_file())

    image_files = [f for f in image_files if "with_arrows" not in str(f)]
    image_files.sort()

    print(f"Total images to scan: {len(image_files)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(flush=True)

    copied = 0
    errors = 0

    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(image_files)} scanned, {copied} arrows found...", flush=True)

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue

            img_h, img_w = img.shape[:2]

            # Corresponding label file
            rel_parts = img_path.relative_to(images_dir)
            label_path = labels_dir / rel_parts.with_suffix('.txt')

            signal_boxes = get_signal_boxes(str(label_path), img_w, img_h, target_class_ids)

            if signal_boxes and image_has_arrow(img, signal_boxes):
                dest = output_dir / rel_parts
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(dest))
                copied += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {img_path.name}: {e}", flush=True)

    print(flush=True)
    print(f"=== Results ===", flush=True)
    print(f"Total images scanned: {len(image_files)}", flush=True)
    print(f"Images with arrows detected: {copied}", flush=True)
    print(f"Copied to with_arrows/: {copied}", flush=True)
    print(f"Errors: {errors}", flush=True)


if __name__ == "__main__":
    main()
