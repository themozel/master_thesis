import os
import xml.etree.ElementTree as ET
from collections import defaultdict

XML_DIR = "/home/amo/zeus-training/labels_processed/cvat_rgb_xml"
OUTPUT_LABELS = "/home/amo/zeus-training/percept_split"  # Output directory for YOLO labels

os.makedirs(OUTPUT_LABELS, exist_ok=True)

# -----------------------------
# Step 1: Extract all classes
# -----------------------------
class_set = set()

print("🔍 Scanning XML files for classes...")

for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()

    for obj in root.iter("object"):
        cls_name = obj.find("name").text.strip()
        cls_name = cls_name.replace(" ", "_")  # normalize
        class_set.add(cls_name)

# sort for consistency
class_list = sorted(list(class_set))

# create mapping
CLASSES = {name: idx for idx, name in enumerate(class_list)}

print("\n✅ Classes found:")
for name, idx in CLASSES.items():
    print(f"{idx}: {name}")

# save classes to file
with open("classes.txt", "w") as f:
    for name in class_list:
        f.write(name + "\n")

print("\n📁 classes.txt saved!")


# -----------------------------
# Step 2: Convert annotations
# -----------------------------
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0

    width = box[2] - box[0]
    height = box[3] - box[1]

    return (
        x_center * dw,
        y_center * dh,
        width * dw,
        height * dh,
    )


print("\n🔁 Converting XML → YOLO format...")

empty_files = 0
total_files = 0

for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    total_files += 1

    tree = ET.parse(os.path.join(XML_DIR, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_filename = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(OUTPUT_LABELS, txt_filename)

    lines = []

    for obj in root.iter("object"):
        cls_name = obj.find("name").text.strip()
        cls_name = cls_name.replace(" ", "_")

        cls_id = CLASSES[cls_name]

        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)

        bb = convert((w, h), (xmin, ymin, xmax, ymax))

        line = f"{cls_id} {' '.join(map(str, bb))}"
        lines.append(line)

    if len(lines) == 0:
        empty_files += 1

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

print("\n✅ Conversion done!")
print(f"📊 Total files: {total_files}")
print(f"📭 Empty label files: {empty_files}")
