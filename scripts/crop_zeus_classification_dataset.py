"""Build a classification dataset by cropping each labeled bounding box out of
zeus-cropped's detection images, pooling instances by class and then re-splitting
them randomly into train/val/test (independent of the original detection split).

Reads YOLO-format labels from data/zeus-cropped/labels/{train,val,test}/*.txt
(class cx cy w h, normalized) and the matching images from
data/zeus-cropped/images/{train,val,test}/. Every labeled box, regardless of
which detection split its source image came from, is pooled per class and then
randomly assigned to a classification train/val/test split, writing:

    <output-dir>/{train,val,test}/<class_name>/<image_stem>__<idx>.<ext>

Class names come from data/zeus-cropped/classes.txt (line N -> class id N-1),
so folder names match the detection class names directly.

Run:
    python crop_zeus_classification_dataset.py \
        --dataset-dir /home/amo/zeus-training/master_thesis/data/zeus-cropped \
        --output-dir /home/amo/zeus-training/master_thesis/data/zeus-cropped-classification
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

DEFAULT_DATASET_DIR = "/home/amo/zeus-training/master_thesis/data/zeus-cropped"
DEFAULT_OUTPUT_DIR = "/home/amo/zeus-training/master_thesis/data/zeus-cropped-classification"
SOURCE_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop zeus-cropped bounding boxes into a classification dataset: pool all "
        "instances of each class into one place first, then randomly split each class's crops "
        "into train/val/test."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Fractional padding added around each box before cropping (0.1 = 10%% of "
        "the box's own width/height on each side). Default 0 (tight crop).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Skip crops whose width or height (in pixels, after padding/clipping) is "
        "below this value.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_class_names(classes_txt: Path) -> list[str]:
    return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yolo_box_to_pixels(cx, cy, w, h, img_w, img_h, padding):
    box_w = w * img_w
    box_h = h * img_h
    cx *= img_w
    cy *= img_h

    box_w *= 1.0 + 2 * padding
    box_h *= 1.0 + 2 * padding

    x1 = cx - box_w / 2.0
    y1 = cy - box_h / 2.0
    x2 = cx + box_w / 2.0
    y2 = cy + box_h / 2.0

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


def collect_boxes(dataset_dir, class_names, padding, min_size):
    """Scan every source split's labels and group valid boxes by class.

    Returns {class_name: [(image_path, stem, idx, x1, y1, x2, y2), ...]}.
    No cropping/image I/O happens here — only label parsing — so this is cheap
    even before we know which classification split each box will land in.
    """
    boxes_by_class = defaultdict(list)
    num_skipped = 0

    for source_split in SOURCE_SPLITS:
        images_dir = dataset_dir / "images" / source_split
        labels_dir = dataset_dir / "labels" / source_split

        if not labels_dir.exists():
            print(f"Skipping source split '{source_split}': {labels_dir} does not exist.")
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            image_path = find_image_path(images_dir, stem)
            if image_path is None:
                print(f"Warning: no image found for label {label_path}, skipping.")
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    num_skipped += 1
                    continue

                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(class_names):
                    print(f"Warning: class id {class_id} out of range in {label_path}, skipping box.")
                    num_skipped += 1
                    continue

                boxes_by_class[class_names[class_id]].append((image_path, stem, idx, *map(float, parts[1:5])))

    return boxes_by_class, num_skipped


def split_indices(n, train_ratio, val_ratio, rng):
    order = list(range(n))
    rng.shuffle(order)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def ensure_all_class_dirs(output_dir, class_names):
    """Create every class folder in every split up front, even with 0 instances,
    so train/val/test always list the same set of classes (matches classes.txt)."""
    for target_split in ("train", "val", "test"):
        for class_name in class_names:
            (output_dir / target_split / class_name).mkdir(parents=True, exist_ok=True)


def crop_and_save(entry, class_name, target_split, output_dir, padding, min_size):
    image_path, stem, idx, cx, cy, w, h = entry

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        img_w, img_h = image.size
        x1, y1, x2, y2 = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h, padding)

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return False

        class_dir = output_dir / target_split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        crop = image.crop((x1, y1, x2, y2))
        out_path = class_dir / f"{stem}__{idx}{image_path.suffix}"
        crop.save(out_path)

    return True


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise SystemExit("--train-ratio, --val-ratio, --test-ratio must sum to 1.0")

    class_names = load_class_names(dataset_dir / "classes.txt")
    print(f"Loaded {len(class_names)} classes from {dataset_dir / 'classes.txt'}")

    ensure_all_class_dirs(output_dir, class_names)

    boxes_by_class, num_skipped_parsing = collect_boxes(dataset_dir, class_names, args.padding, args.min_size)

    rng = random.Random(args.seed)
    total_crops = 0
    total_skipped = num_skipped_parsing

    for class_name in class_names:
        entries = boxes_by_class.get(class_name, [])
        if not entries:
            print(f"[{class_name}] no instances found, leaving empty folders in place.")
            continue

        assignment = split_indices(len(entries), args.train_ratio, args.val_ratio, rng)

        counts = {}
        for target_split, indices in assignment.items():
            written = 0
            for i in indices:
                if crop_and_save(entries[i], class_name, target_split, output_dir, args.padding, args.min_size):
                    written += 1
                else:
                    total_skipped += 1
            counts[target_split] = written
            total_crops += written

        print(
            f"[{class_name}] {len(entries)} instances -> "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )

    print(f"Done. Total crops: {total_crops}, total skipped: {total_skipped}.")
    print(f"Output dataset root: {output_dir}")


if __name__ == "__main__":
    main()
