# detect_arrows.py — Method Documentation

## Overview

`detect_arrows.py` detects directional arrow plates (small signs mounted near railroad signals) in images using classical computer vision techniques. It does **not** use machine learning — instead it relies on thresholding, contour analysis, and geometric shape properties.

---

## Pipeline Summary

```
Image → Load signal bounding boxes from YOLO labels
      → Expand search region around each signal
      → In each ROI: find bright rectangular plates
      → Inside each plate: find dark shapes
      → Classify dark shapes as arrows using solidity + convexity defects
      → Output: copy image / draw bounding boxes
```

---

## Step-by-Step Method Explanation

### 1. Signal Localisation (`get_signal_boxes`)

Rather than scanning the entire 2064×1544 image (which is slow), the script leverages existing YOLO-format label files to know **where the signals are**. It reads the label `.txt` files and extracts bounding boxes for the four signal classes:

- `sig_free_left` (class ID 8)
- `sig_free_right` (class ID 9)
- `sig_free_straight` (class ID 10)
- `sig_stop` (class ID 20)

YOLO format stores boxes as normalised `(center_x, center_y, width, height)` — these are converted to absolute pixel coordinates `(x1, y1, x2, y2)`.

### 2. Region of Interest Expansion (`find_arrows_in_image`)

Arrow plates are physically mounted **near** signals (above, below, or beside them). The script expands the search area around each signal bounding box:

- **Horizontal margin**: `max(box_width × 2, 80px)`
- **Vertical margin**: `max(box_height × 3, 120px)`

This generous expansion ensures arrow plates within the signal's vicinity are captured without scanning the full image.

### 3. Plate Detection via Thresholding (`find_arrow_plates`)

Arrow plates have a distinctive appearance: a **bright/white rectangular background** with a **dark arrow symbol** on it.

#### 3.1 Bright Region Detection

The ROI (already converted to grayscale) is thresholded at multiple levels (170, 190, 210) to handle varying lighting conditions:

```python
cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
```

Multiple thresholds improve robustness — a plate that's slightly dirty or in shadow may not pass threshold 210 but will pass 170.

#### 3.2 Morphological Closing

A 5×5 rectangular structuring element closes small gaps in the bright regions, merging adjacent bright pixels into coherent plate shapes:

```python
cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
```

#### 3.3 Plate Candidate Filtering

External contours of the bright mask are found, then filtered by:

| Criterion | Condition | Rationale |
|-----------|-----------|-----------|
| Area | 150–6000 px² | Rejects noise (too small) and large background regions (too big) |
| Aspect ratio | ≤ 1.8 | Arrow plates are roughly square; elongated shapes are not plates |
| Minimum size | width ≥ 12px, height ≥ 12px | Ensures enough pixels to analyse the content |

The aspect ratio is computed from `cv2.minAreaRect` (rotated bounding rectangle), making it rotation-invariant.

### 4. Arrow Shape Detection Inside Plates

For each plate candidate, the interior is analysed for dark arrow-shaped content.

#### 4.1 Dark Content Extraction

```python
cv2.threshold(plate, 100, 255, cv2.THRESH_BINARY_INV)
```

Pixels darker than 100 are isolated. The **dark pixel ratio** must be between 10%–55% of the plate area (too few = empty plate, too many = plate is not actually bright).

#### 4.2 Contour Analysis of Dark Shapes

Each dark contour (minimum area 40 px²) is tested for arrow-like geometry using two key metrics:

##### Solidity

```
solidity = contour_area / convex_hull_area
```

- **Arrows**: 0.35–0.88 (they have concavities from the "notch" where the shaft meets the head)
- **Filled rectangles/circles**: ~1.0 (rejected — not arrows)
- **Very fragmented shapes**: < 0.35 (rejected — noise)

Solidity measures how "filled" a shape is relative to its convex hull. Arrows have a characteristic gap (the indentation at the tail/shaft junction) that reduces solidity below 1.0.

##### Convexity Defects

```python
defects = cv2.convexityDefects(contour, convex_hull_indices)
```

Convexity defects are the **inward-pointing pockets** between a contour and its convex hull. For an arrow shape, these correspond to the notches on either side of the arrowhead-shaft junction.

The algorithm counts defects with depth > 400 (in OpenCV's fixed-point representation, ×256). At least **one significant defect** must be present — this is the defining geometric feature of an arrow vs. other shapes like circles or rectangles.

##### Elongation (Aspect Ratio)

```python
dc_aspect = max(width, height) / min(width, height)
```

Must be ≥ 1.1. Arrows are elongated (they point in a direction), so perfectly circular shapes are rejected.

### 5. Coordinate Transformation

Detections are in ROI-local coordinates. They're mapped back to full-image coordinates by adding the ROI offset:

```python
(sx1 + bx, sy1 + by, sx1 + bx + bw, sy1 + by + bh)
```

### 6. Output

- **Default mode**: copies images with detected arrows to `with_arrows/`
- **`--draw-boxes`**: saves annotated copies with green bounding boxes around arrows to `with_arrows_annotated/`
- **`--draw-signals`**: additionally draws red bounding boxes around the signals themselves

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Multiple thresholds (170/190/210) | Handles varying illumination and plate cleanliness |
| Search only near signals, not full image | ~10× speed improvement (4029 images in ~5 min vs. timeout) |
| Solidity + convexity defects | Robust geometric features that distinguish arrows from other shapes |
| Plate-first, then arrow-inside | Two-stage approach reduces false positives from random dark shapes in the image |
| Return plate bounding box (not arrow contour) | The plate is the meaningful visual unit; easier to verify visually |

---

## Limitations

- **False positives**: Other small bright rectangular objects near signals with dark angular content may be detected
- **False negatives**: Very dirty/occluded plates, unusual lighting, or plates far from the signal may be missed
- **No direction classification**: The script detects the presence of an arrow but does not determine which direction it points (left/right/straight/diagonal)
- **Fixed thresholds**: May need tuning for different cameras or weather conditions

---

## Usage

```bash
# Detect and copy only
python scripts/detect_arrows.py

# With green bounding boxes on arrows
python scripts/detect_arrows.py --draw-boxes

# With green arrow boxes + red signal boxes
python scripts/detect_arrows.py --draw-boxes --draw-signals
```
