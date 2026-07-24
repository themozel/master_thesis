import yaml

# read classes
with open(
    "/home/amo/zeus-training/master_thesis/data/GERALD/split_dataset_per_sequence/classes.txt"
) as f:
    class_names = [line.strip() for line in f.readlines()]

YOLO_DATASET_PATH = (
    "/home/amo/zeus-training/master_thesis/data/GERALD/split_dataset_per_sequence"
)

data = {
    "train": f"{YOLO_DATASET_PATH}/images/train",
    "val": f"{YOLO_DATASET_PATH}/images/val",
    "nc": len(class_names),
    "names": class_names,
}

with open(f"{YOLO_DATASET_PATH}/data.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("✅ data.yaml created!")
