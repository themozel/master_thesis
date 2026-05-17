import yaml

# read classes
with open(
    "/mnt/c/Amid/Uni/Master Thesis/Dataset/dataset_full_backup/dataset_full_backup/classes.txt"
) as f:
    class_names = [line.strip() for line in f.readlines()]

YOLO_DATASET_PATH = (
    "/home/themozel/Projects/master_thesis/signal_detection/data/PERCEPT"
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
