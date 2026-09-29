import argparse
import os
import random
import re
from collections import defaultdict

DEFAULT_IMAGE_ROOT = (
    "/home/amo/zeus-training/master_thesis/data/zeus-cropped/images/"
    "mee119_cvat_upload_ready_images_only/obj_train_data"
)
DEFAULT_LABEL_ROOT = "/home/amo/zeus-training/master_thesis/data/zeus-cropped/labels/obj_train_data"
DEFAULT_OUTPUT_DIR = "/home/amo/zeus-training/master_thesis/data/zeus-cropped"

# Groups e.g. "20260521T044706Z_weichenstellungsmelder_1544_frame_000000__center.png"
# into sequence "20260521T044706Z_weichenstellungsmelder_1544" so that frames from the
# same recording never leak across train/val/test (mirrors split_GERALD.py's marker strategy,
# but zeus filenames use "_frame_<n>__<center|left|right>" instead of "#t=").
SEQUENCE_RE = re.compile(r"^(.*)_frame_\d+__(?:center|left|right)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively collect zeus-cropped images/labels from their job* "
        "subfolders and split them into a flat train/val/test structure, stratified "
        "per recording sequence, matching the GERALD-cropped/percept-cropped layout."
    )
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--label-ext", default=".txt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sequence_of(filename):
    match = SEQUENCE_RE.match(filename)
    if not match:
        raise ValueError(f"Filename did not match expected pattern: {filename}")
    return match.group(1)


def find_files(root, ext):
    """Recursively find files under root, returning {basename: full_path}."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(ext):
                found[filename] = os.path.join(dirpath, filename)
    return found


def main():
    args = parse_args()
    random.seed(args.seed)

    split_ratio = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if abs(sum(split_ratio.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio.values())}")

    images = find_files(args.image_root, args.image_ext)
    labels = find_files(args.label_root, args.label_ext)

    missing_labels = [f for f in images if f[: -len(args.image_ext)] + args.label_ext not in labels]
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} images have no matching label, e.g. {missing_labels[:5]}")

    sequences = defaultdict(list)
    for filename in images:
        stem = filename[: -len(args.image_ext)]
        sequences[sequence_of(stem)].append(filename)

    splits = {"train": [], "val": [], "test": []}
    for files in sequences.values():
        files = files[:]
        random.shuffle(files)

        train_end = int(len(files) * split_ratio["train"])
        val_end = train_end + int(len(files) * split_ratio["val"])

        splits["train"].extend(files[:train_end])
        splits["val"].extend(files[train_end:val_end])
        splits["test"].extend(files[val_end:])

    for split in splits:
        os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)

    for split, files in splits.items():
        print(f"Processing {split}: {len(files)} files")
        for filename in files:
            img_src = images[filename]
            img_dst = os.path.join(args.output_dir, "images", split, filename)
            os.rename(img_src, img_dst)

            lbl_name = filename[: -len(args.image_ext)] + args.label_ext
            lbl_src = labels[lbl_name]
            lbl_dst = os.path.join(args.output_dir, "labels", split, lbl_name)
            os.rename(lbl_src, lbl_dst)

    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
