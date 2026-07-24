import argparse
import os
import random
import re
import shutil
from collections import defaultdict

from tqdm import tqdm

DEFAULT_IMAGE_DIR = "/home/amo/zeus-training/datasets/GERALD/dataset/JPEGImages"
DEFAULT_LABEL_DIRS = ["/home/amo/zeus-training/master_thesis/data/GERALD/output_labels"]
DEFAULT_OUTPUT_DIR = "/home/amo/zeus-training/master_thesis/data/GERALD/split_dataset_per_sequence"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split an image/label dataset into train/val/test, "
        "optionally stratified per sequence so each sequence is split at the same ratio."
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--label-dirs", nargs="+", default=DEFAULT_LABEL_DIRS)
    parser.add_argument("--label-ext", default=".txt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence-strategy",
        choices=["flat", "marker", "regex"],
        default="marker",
        help="How to derive each image's sequence for stratified splitting: "
        "'flat' does no grouping (plain random split over all images), "
        "'marker' groups by the filename prefix before --sequence-marker, "
        "'regex' groups by the first capture group of --sequence-regex "
        "(or the whole match if the pattern has no group).",
    )
    parser.add_argument(
        "--sequence-marker",
        default="#t=",
        help="Used with --sequence-strategy=marker. E.g. 'Aachen_Duesseldorf.mp4#t=43.4.jpg' "
        "with marker '#t=' groups as sequence 'Aachen_Duesseldorf.mp4'.",
    )
    parser.add_argument(
        "--sequence-regex",
        default=None,
        help="Used with --sequence-strategy=regex, e.g. '^(.*)_\\d+\\.jpg$'.",
    )
    return parser.parse_args()


def make_sequence_fn(args):
    if args.sequence_strategy == "flat":
        return lambda filename: ""

    if args.sequence_strategy == "marker":
        marker = args.sequence_marker
        return lambda filename: filename.split(marker)[0]

    if args.sequence_strategy == "regex":
        if not args.sequence_regex:
            raise ValueError("--sequence-regex is required when --sequence-strategy=regex")
        pattern = re.compile(args.sequence_regex)

        def sequence_fn(filename):
            match = pattern.search(filename)
            if not match:
                raise ValueError(f"Sequence regex did not match filename: {filename}")
            return match.group(1) if match.groups() else match.group(0)

        return sequence_fn

    raise ValueError(f"Unknown sequence strategy: {args.sequence_strategy}")


def load_label_contents(label_dirs, label_ext):
    contents = {}
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith(label_ext):
                with open(os.path.join(label_dir, filename), "r") as f:
                    contents[filename] = f.read()
    return contents


def main():
    args = parse_args()
    random.seed(args.seed)

    split_ratio = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if abs(sum(split_ratio.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio.values())}")

    sequence_fn = make_sequence_fn(args)
    label_contents = load_label_contents(args.label_dirs, args.label_ext)

    images = [f for f in os.listdir(args.image_dir) if f.endswith(args.image_ext)]

    sequences = defaultdict(list)
    for image in images:
        sequences[sequence_fn(image)].append(image)

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
        img_dir = os.path.join(args.output_dir, "images", split)
        lbl_dir = os.path.join(args.output_dir, "labels", split)
        shutil.rmtree(img_dir, ignore_errors=True)
        shutil.rmtree(lbl_dir, ignore_errors=True)
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

    for split, files in splits.items():
        for file in tqdm(files, desc=f"Processing {split}"):
            img_src = os.path.join(args.image_dir, file)
            img_dst = os.path.join(args.output_dir, "images", split, file)
            shutil.copy(img_src, img_dst)

            lbl_name = file[: -len(args.image_ext)] + args.label_ext
            if lbl_name in label_contents:
                lbl_dst = os.path.join(args.output_dir, "labels", split, lbl_name)
                with open(lbl_dst, "w") as f:
                    f.write(label_contents[lbl_name])


if __name__ == "__main__":
    main()
