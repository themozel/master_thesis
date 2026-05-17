import os
import random
import shutil
from pathlib import Path

random.seed(42)

# --------------------------------
# CONFIG
# --------------------------------
IMAGE_ROOT = "/mnt/c/Amid/Uni/Master Thesis/Dataset/dataset_full_backup/dataset_full_backup/processed/images_rgb_tonemapped"
LABEL_ROOT = "/mnt/c/Amid/Uni/Master Thesis/Dataset/dataset_full_backup/dataset_full_backup/labels"
OUTPUT_ROOT = "data/PERCEPT_SPLIT"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

# --------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------
for split in SPLITS:
    os.makedirs(f"{OUTPUT_ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/labels/{split}", exist_ok=True)

# --------------------------------
# FIND ALL IMAGES RECURSIVELY
# --------------------------------
all_images = []

for ext in IMAGE_EXTENSIONS:
    all_images.extend(Path(IMAGE_ROOT).rglob(f"*{ext}"))

print(f"✅ Found {len(all_images)} images")

# shuffle images
random.shuffle(all_images)

# --------------------------------
# SPLIT DATASET
# --------------------------------
n_total = len(all_images)

n_train = int(n_total * SPLITS["train"])
n_val = int(n_total * SPLITS["val"])

split_data = {
    "train": all_images[:n_train],
    "val": all_images[n_train : n_train + n_val],
    "test": all_images[n_train + n_val :],
}

# --------------------------------
# COPY FILES
# --------------------------------
missing_labels = 0

for split, image_paths in split_data.items():

    print(f"\n📂 Processing {split}...")

    for img_path in image_paths:

        img_path = Path(img_path)

        img_name = img_path.name
        label_name = img_path.stem + ".txt"

        # labels are flat
        label_path = Path(LABEL_ROOT) / label_name

        # destination
        dst_img = Path(f"{OUTPUT_ROOT}/images/{split}") / img_name
        dst_lbl = Path(f"{OUTPUT_ROOT}/labels/{split}") / label_name

        # copy image
        shutil.copy(img_path, dst_img)

        # copy label if exists
        if label_path.exists():
            shutil.copy(label_path, dst_lbl)
        else:
            missing_labels += 1

print("\n✅ Dataset split complete!")
print(f"⚠️ Missing labels: {missing_labels}")
