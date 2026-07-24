import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)

# --------------------------------
# CONFIG
# --------------------------------
IMAGE_ROOT = "/home/amo/zeus-training/images_rgb_tonemapped"
XML_DIR = "/home/amo/zeus-training/labels_processed/cvat_rgb_xml"
OUTPUT_ROOT = "/home/amo/zeus-training/percept_split"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

# --------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------
for split in SPLITS:
    os.makedirs(f"{OUTPUT_ROOT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/labels/{split}", exist_ok=True)

# --------------------------------
# PARSE CVAT XML -> YOLO LABELS PER IMAGE STEM
# --------------------------------
def to_yolo_box(xtl, ytl, xbr, ybr, img_w, img_h):
    x_center = (xtl + xbr) / 2.0 / img_w
    y_center = (ytl + ybr) / 2.0 / img_h
    width = (xbr - xtl) / img_w
    height = (ybr - ytl) / img_h
    return x_center, y_center, width, height


xml_files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))

print(f"🔍 Scanning {len(xml_files)} CVAT XML files for classes...")

class_set = set()
for xml_file in xml_files:
    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    for box in tree.getroot().iter("box"):
        class_set.add(box.get("label"))

class_list = sorted(class_set)
CLASSES = {name: idx for idx, name in enumerate(class_list)}

print("✅ Classes found:")
for name, idx in CLASSES.items():
    print(f"  {idx}: {name}")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(f"{OUTPUT_ROOT}/classes.txt", "w") as f:
    f.write("\n".join(class_list))

print("\n🔁 Converting CVAT XML -> YOLO labels...")

annotations = {}  # image stem -> list of YOLO label lines

for xml_file in xml_files:
    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    for image_el in tree.getroot().iter("image"):
        stem = Path(image_el.get("name")).stem
        img_w = float(image_el.get("width"))
        img_h = float(image_el.get("height"))

        lines = []
        for box in image_el.findall("box"):
            cls_id = CLASSES[box.get("label")]
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            x, y, w, h = to_yolo_box(xtl, ytl, xbr, ybr, img_w, img_h)
            lines.append(f"{cls_id} {x} {y} {w} {h}")

        annotations[stem] = lines

print(f"✅ Parsed annotations for {len(annotations)} images")

# --------------------------------
# FIND ALL IMAGES RECURSIVELY
# --------------------------------
all_images = []

for ext in IMAGE_EXTENSIONS:
    all_images.extend(Path(IMAGE_ROOT).rglob(f"*{ext}"))

print(f"✅ Found {len(all_images)} images")

# keep only images that have a matching label, so images/labels counts stay in sync
n_found = len(all_images)
all_images = [p for p in all_images if p.stem in annotations]
print(f"✅ {len(all_images)}/{n_found} images have a matching label ({n_found - len(all_images)} skipped)")

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
# COPY IMAGES + WRITE YOLO LABELS
# --------------------------------
for split, image_paths in split_data.items():

    print(f"\n📂 Processing {split}...")

    for img_path in image_paths:

        img_path = Path(img_path)

        img_name = img_path.name
        label_name = img_path.stem + ".txt"

        # destination
        dst_img = Path(f"{OUTPUT_ROOT}/images/{split}") / img_name
        dst_lbl = Path(f"{OUTPUT_ROOT}/labels/{split}") / label_name

        # copy image
        shutil.copy(img_path, dst_img)

        # write matching YOLO label (by filename stem)
        dst_lbl.write_text("\n".join(annotations[img_path.stem]))

print("\n✅ Dataset split complete!")
