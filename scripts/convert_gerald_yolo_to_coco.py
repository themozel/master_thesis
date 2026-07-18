"""Convert GERALD YOLO labels to COCO annotation JSON files.

This script reads class names from data/GERALD/data.yaml and creates:
  data/GERALD/annotations/instances_train2017.json
  data/GERALD/annotations/instances_val2017.json
  data/GERALD/annotations/instances_test2017.json

It does not modify classes or remap class IDs.
"""

from __future__ import annotations

import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(data_yaml_path: Path) -> list[str]:
    class_names: list[str] = []
    in_names = False
    for raw in data_yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line.startswith("-"):
                class_names.append(line[1:].strip())
            else:
                break
    return class_names


def get_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return im.size
    except Exception:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        return w, h


def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    w = width * img_w
    h = height * img_h
    x = (x_center * img_w) - (w / 2.0)
    y = (y_center * img_h) - (h / 2.0)
    return x, y, w, h


def convert_split(
    split: str,
    images_root: Path,
    labels_root: Path,
    categories: list[dict],
) -> dict:
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    split_images_dir = images_root / split
    split_labels_dir = labels_root / split

    image_files = sorted(
        p
        for p in split_images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_files:
        width, height = get_image_size(image_path)

        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.relative_to(images_root)),
                "width": width,
                "height": height,
            }
        )

        label_path = split_labels_dir / image_path.relative_to(split_images_dir).with_suffix(".txt")
        if label_path.exists():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:5])
                x, y, w, h = yolo_to_coco_bbox(x_center, y_center, bw, bh, width, height)

                # Keep bbox within image bounds to avoid invalid negative areas.
                x = max(0.0, min(x, float(width)))
                y = max(0.0, min(y, float(height)))
                w = max(0.0, min(w, float(width) - x))
                h = max(0.0, min(h, float(height) - y))
                area = w * h

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_root = base_dir / "data" / "GERALD"

    data_yaml = dataset_root / "data.yaml"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    annotations_root = dataset_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(data_yaml)
    if not class_names:
        raise RuntimeError(f"No class names found in {data_yaml}")

    categories = [
        {
            "id": idx + 1,
            "name": name,
            "supercategory": "none",
        }
        for idx, name in enumerate(class_names)
    ]

    for split, suffix in (("train", "train2017"), ("val", "val2017"), ("test", "test2017")):
        coco = convert_split(split, images_root, labels_root, categories)
        out_path = annotations_root / f"instances_{suffix}.json"
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(
            f"{split}: images={len(coco['images'])}, "
            f"annotations={len(coco['annotations'])}, out={out_path}"
        )


if __name__ == "__main__":
    main()
