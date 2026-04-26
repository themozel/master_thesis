import os
import random
import shutil
from tqdm import tqdm

random.seed(42)

IMAGE_DIR = "data/images/all"
LABEL_DIR = "data/labels/all"

OUTPUT_DIR = "data"

SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
random.shuffle(images)

train_split = int(len(images) * SPLIT_RATIO["train"])
val_split = int(len(images) * SPLIT_RATIO["val"])

splits = {
    "train": images[:train_split],
    "val": images[train_split : train_split + val_split],
    "test": images[train_split + val_split :],
}

for split in splits:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

for split, files in splits.items():
    for file in tqdm(files, desc=f"Processing {split}"):
        img_src = os.path.join(IMAGE_DIR, file)
        lbl_src = os.path.join(LABEL_DIR, file.replace(".jpg", ".txt"))

        img_dst = os.path.join(OUTPUT_DIR, "images", split, file)
        lbl_dst = os.path.join(
            OUTPUT_DIR, "labels", split, file.replace(".jpg", ".txt")
        )

        shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
