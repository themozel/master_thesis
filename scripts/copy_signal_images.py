"""Build signal/switch subsets for PERCEPT or GERALD datasets.

Creates a trainable YOLO subset with:
    - images/{train,val,test}
    - labels/{train,val,test} remapped to contiguous class IDs
    - classes.txt
    - data.yaml
"""

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


DATASET_TARGETS = {
    "PERCEPT": [
        "sig_free_left",
        "sig_free_right",
        "sig_free_straight",
        "sig_stop",
        "sig_switch_left_free",
        "sig_switch_left_locked",
        "sig_switch_right_free",
        "sig_switch_right_locked",
        "sig_switch_straight_free",
        "sig_switch_straight_locked",
    ],
    "GERALD": [
        "Hp_0_HV",
        "Hp_0_Ks",
        "Hp_0_Sh",
        "Hp_1",
        "Hp_2",
        "Ks_1",
        "Ks_2",
        "Sh_0",
        "Sh_1",
        "Sh_2",
        "Vr_0",
        "Vr_1",
        "Vr_2",
        "Zs_2",
        "Zs_2v",
        "Zs_3",
        "Zs_3v",
        "Zs_6",
        "Zs_Off",
        "Signal_Identifier_Sign",
        "Signal_Invalid",
        "Signal_Off",
    ],
}


def find_image_path(images_root: Path, stem_rel: Path):
    for ext in IMAGE_EXTENSIONS:
        img_path = images_root / stem_rel.with_suffix(ext)
        if img_path.exists():
            return img_path

    base = stem_rel.name
    candidates = list(images_root.rglob(f"{base}.*"))
    img_candidates = [c for c in candidates if c.suffix.lower() in IMAGE_EXTENSIONS]
    return img_candidates[0] if img_candidates else None


def write_classes_file(classes_path: Path, class_names):
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def write_data_yaml(data_yaml_path: Path, output_root: Path, class_names):
    data_yaml_path.write_text(
        "\n".join(
            [
                f"train: {output_root / 'images' / 'train'}",
                f"val: {output_root / 'images' / 'val'}",
                f"test: {output_root / 'images' / 'test'}",
                f"nc: {len(class_names)}",
                "names:",
                *[f"- {name}" for name in class_names],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def load_classes(classes_file: Path, data_yaml_file: Path):
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    classes = []
    in_names = False
    with open(data_yaml_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line.startswith("-"):
                    classes.append(line[1:].strip())
                elif line:
                    break
    return classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="PERCEPT",
        choices=sorted(DATASET_TARGETS.keys()),
        help="Dataset to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    base_dir = Path(__file__).resolve().parent.parent
    dataset_root = base_dir / "data" / dataset
    classes_file = dataset_root / "classes.txt"
    data_yaml_file = dataset_root / "data.yaml"
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"
    output_root = dataset_root / "images_signal_and_switch"
    output_images_root = output_root / "images"
    output_labels_root = output_root / "labels"

    output_images_root.mkdir(parents=True, exist_ok=True)
    output_labels_root.mkdir(parents=True, exist_ok=True)

    if not labels_root.exists() or not images_root.exists():
        raise FileNotFoundError(f"Missing dataset folders under {dataset_root}")

    target_names = DATASET_TARGETS[dataset]

    all_classes = load_classes(classes_file, data_yaml_file)
    if not all_classes:
        raise RuntimeError(f"Could not load classes for dataset: {dataset}")

    target_ids = set()
    for name in target_names:
        if name in all_classes:
            target_ids.add(all_classes.index(name))
        else:
            print(f"WARNING: Class '{name}' not found in class list")

    target_id_map = {
        all_classes.index(name): subset_id
        for subset_id, name in enumerate(target_names)
        if name in all_classes
    }

    target_names_present = [name for name in target_names if name in all_classes]

    print(f"Dataset: {dataset}")
    print(f"Target class names requested: {target_names}")
    print(f"Target class names present: {target_names_present}")
    print(f"Target class IDs: {sorted(target_ids)}")
    print()

    # Find all label files containing any target class
    matched_labels = []
    for label_path in sorted(labels_root.rglob("*.txt")):
        filtered_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in target_ids:
                        parts[0] = str(target_id_map[cls_id])
                        filtered_lines.append(" ".join(parts))

        if filtered_lines:
            matched_labels.append((label_path, filtered_lines))

    print(f"Matched label files: {len(matched_labels)}")

    # Copy corresponding images and write remapped label files
    copied = 0
    missing = 0
    written_labels = 0

    for label_path, filtered_lines in matched_labels:
        # Label path: labels/{split}/filename.txt
        # Image path: images/{split}/filename.{ext}
        rel = label_path.relative_to(labels_root)
        stem_rel = rel.with_suffix('')

        dest_label = output_labels_root / rel
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_label.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        written_labels += 1

        img_path = find_image_path(images_root, stem_rel)
        if img_path:
            dest = output_images_root / rel.with_suffix(img_path.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(img_path), str(dest))
            copied += 1
        else:
            missing += 1

    write_classes_file(output_root / "classes.txt", target_names_present)
    write_data_yaml(output_root / "data.yaml", output_root, target_names_present)

    print(f"Copied images: {copied}")
    print(f"Written label files: {written_labels}")
    print(f"Missing image matches: {missing}")


if __name__ == "__main__":
    main()
