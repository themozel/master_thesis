#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Shared, dependency-light pieces of the zeus-cropped detect+classify
pipeline: the class/group definitions and the visualization renderer. Used
by both detect_classify_pipeline.py (PyTorch/GPU) and
detect_classify_pipeline_hailo.py (Hailo-8/HailoRT) so the two stay in sync
and this module itself only needs opencv/numpy (safe to import on either a
dev GPU box or a lean edge device with no PyTorch installed)."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Detection classes (in dataset category-id order, see
# data/zeus-cropped/classes.txt) merged into the three groups the detector
# is being evaluated on. Any detection class not listed here (i.e.
# "overexposed") is dropped by default.
GROUP_MAP = {
    "sig_stop": "signals",
    "sig_stop_occupied": "signals",
    "sig_free_straight": "signals",
    "sig_free_left": "signals",
    "sig_free_right": "signals",

    "sig_switch_straight_locked": "switches",
    "sig_switch_left_locked": "switches",
    "sig_switch_right_locked": "switches",
    "sig_switch_straight_free": "switches",
    "sig_switch_left_free": "switches",
    "sig_switch_right_free": "switches",
    "sig_switch_faulty_1": "switches",
    "sig_switch_faulty_2": "switches",
    "sig_switch_faulty_3": "switches",

    "sig_aux_arrow_right": "auxilary",
    "sig_aux_arrow_left": "auxilary",
    "sig_aux_arrow_right_diagonal": "auxilary",
    "sig_aux_arrow_left_diagonal": "auxilary",
    "sig_aux_arrow_straight": "auxilary",
    "sig_aux_tram_num_arrow_straight": "auxilary",
    "sig_aux_tram_num": "auxilary",
    "sig_aux_tram_arrow_straight": "auxilary",
    "sig_aux_tram_arrow_right": "auxilary",
    "sig_aux_tram_arrow_left": "auxilary",
    "sig_aux_tram_arrow_right_diagonal": "auxilary",
    "sig_aux_tram_arrow_left_diagonal": "auxilary",
    "sig_aux_tram": "auxilary",
    "sig_aux_bus_num_arrow": "auxilary",
    "sig_aux_bus_arrow_left": "auxilary",
    "sig_aux_bus_arrow_right": "auxilary",
    "sig_aux_bus": "auxilary",
    "sig_aux_right_forward_arrow": "auxilary",
}

# category-id order (1..33) from data/zeus-cropped/annotations/instances_*.json,
# i.e. detector class index i (0-indexed) corresponds to DETECTOR_CLASSES[i].
DETECTOR_CLASSES = [
    "sig_stop", "sig_stop_occupied", "sig_free_straight", "sig_free_left",
    "sig_free_right", "sig_switch_straight_locked", "sig_switch_left_locked",
    "sig_switch_right_locked", "sig_switch_straight_free", "sig_switch_left_free",
    "sig_switch_right_free", "sig_switch_faulty_1", "sig_switch_faulty_2",
    "sig_switch_faulty_3", "sig_aux_arrow_right", "sig_aux_arrow_left",
    "sig_aux_arrow_right_diagonal", "sig_aux_arrow_left_diagonal",
    "sig_aux_arrow_straight", "sig_aux_tram_num_arrow_straight", "sig_aux_tram_num",
    "sig_aux_tram_arrow_straight", "sig_aux_tram_arrow_right", "sig_aux_tram_arrow_left",
    "sig_aux_tram_arrow_right_diagonal", "sig_aux_tram_arrow_left_diagonal",
    "sig_aux_tram", "sig_aux_bus_num_arrow", "sig_aux_bus_arrow_left",
    "sig_aux_bus_arrow_right", "sig_aux_bus", "sig_aux_right_forward_arrow",
    "overexposed",
]

GROUP_COLORS = {
    "signals": (0, 0, 255),     # red (BGR)
    "switches": (0, 200, 0),    # green
    "auxilary": (255, 128, 0),  # blue-ish
}


def get_image_list(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)


LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.45
LABEL_LINE_HEIGHT = 20
MARGIN_WIDTH = 260


def _label_text(det):
    return f"{det['detector_group']}:{det['classifier_class']} {det['classifier_conf']:.2f}"


def _assign_label_rows(dets, height, line_height=LABEL_LINE_HEIGHT, pad=10):
    """Stack labels vertically in the margin without overlapping, keeping them
    as close as possible to their box's own vertical position. Returns
    {det_index: label_y}."""
    order = sorted(range(len(dets)), key=lambda i: (dets[i]["bbox"][1] + dets[i]["bbox"][3]) / 2)
    rows = {}
    last_y = -1e9
    for i in order:
        cy = (dets[i]["bbox"][1] + dets[i]["bbox"][3]) / 2
        y = max(cy, last_y + line_height)
        last_y = y
        rows[i] = y
    # if the stack overflowed the bottom, compress everything back into bounds
    overflow = last_y - (height - pad)
    if overflow > 0:
        for i in rows:
            rows[i] -= overflow
    return rows


def draw_annotated_image(img, dets):
    """Draw a plain box + a small number marker on each detected object (so
    the object itself stays fully visible) and list "N. class name" for
    every box in a margin added to the right of the image, matched to its
    box by that number instead of a leader line. Avoids labels being clipped
    at the image edge or overlapping each other on small/crowded crops."""
    h, w = img.shape[:2]
    order = sorted(range(len(dets)), key=lambda i: (dets[i]["bbox"][1] + dets[i]["bbox"][3]) / 2)
    numbers = {det_idx: n for n, det_idx in enumerate(order, start=1)}

    def row_text(i):
        return f"{numbers[i]}. {_label_text(dets[i])}"

    max_text_w = max(
        (cv2.getTextSize(row_text(i), LABEL_FONT, LABEL_FONT_SCALE, 1)[0][0] for i in range(len(dets))),
        default=0,
    )
    margin_width = max(MARGIN_WIDTH, max_text_w + 30)
    canvas = np.full((h, w + margin_width, 3), 255, dtype=np.uint8)
    canvas[:, :w] = img
    cv2.line(canvas, (w, 0), (w, h), (180, 180, 180), 1)

    for i, d in enumerate(dets):
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        color = GROUP_COLORS.get(d["detector_group"], (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        marker_center = (max(9, x1), max(9, y1))
        cv2.circle(canvas, marker_center, 9, color, -1)
        cv2.putText(canvas, str(numbers[i]),
                    (marker_center[0] - (5 if numbers[i] < 10 else 9), marker_center[1] + 4),
                    LABEL_FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    label_rows = _assign_label_rows(dets, h)
    for i, d in enumerate(dets):
        color = GROUP_COLORS.get(d["detector_group"], (200, 200, 200))
        label_y = int(round(label_rows[i]))
        label_y = max(12, min(h - 6, label_y))
        cv2.putText(canvas, row_text(i), (w + 12, label_y + 4), LABEL_FONT,
                    LABEL_FONT_SCALE, color, 1, cv2.LINE_AA)

    return canvas
