#!/usr/bin/env python3
"""
Train tf_efficientnet_lite3 (Hailo-8 compatible) on a custom classification dataset
and export it to ONNX for the hailomz parse/optimize/compile pipeline.

Usage:
    python train_efficientnet_lite3.py "C:\\Users\\amo\\Downloads\\zeus-cropped-classification"

Expected data layout (this is exactly what your zeus-cropped-classification folder has):

    <data_root>/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png

Requirements:
    pip install timm onnx

--------------------------------------------------------------------------------------
IMPORTANT — issues found in your actual data when this script was (re-)checked
--------------------------------------------------------------------------------------
A re-scan of "C:\\Users\\amo\\Downloads\\zeus-cropped-classification" after your
restructure shows val/ and test/ are now fully readable (0 zero-byte files — the
earlier 100% corruption there is fixed). But train/ has shifted, not disappeared:
6 whole classes are now 100% zero-byte (sig_free_straight, sig_stop, sig_switch_
right_free, sig_switch_right_locked, sig_switch_straight_free, sig_switch_straight_
locked), plus sig_switch_faulty_1 at ~90% zero-byte. Notably sig_stop — your single
largest class — now has ZERO usable training images.

The pattern (whole classes flipping between fully-valid and fully-placeholder across
scans, always 0 bytes rather than truncated/garbled) looks like an active cloud-sync
client (OneDrive/Drive "files on demand") still materializing folders in the
background, rather than random corruption. If that's the cause, make sure the sync
is fully finished (or force "always keep on this device") before your next run —
otherwise you may find a *different* set of classes empty next time you check.

This script automatically detects and skips unreadable/zero-byte files, reports
per-class counts so this kind of gap is impossible to miss, and if val/ ends up
empty it carves a stratified slice out of train/ to validate against instead. What
it can't do is invent training data: a class with 0 usable train images will never
be predicted correctly no matter how long you train, so re-run the scan report
below and confirm every class you care about has a nonzero train count before you
trust the results.

1. Class balance is extremely long-tailed even where data is intact: sig_stop has
   ~3200 train images on paper (currently 0 usable, see above), several
   classes (sig_aux_bus_arrow_right, sig_switch_faulty_3, ...) have exactly 1. A
   WeightedRandomSampler is used to fight this, but a class with 1 example cannot be
   meaningfully validated or generalized — expect near-zero recall on those until you
   collect more examples. Classes with a literal handful of images are worth merging
   into a broader class or excluding until you have more data.

2. Crops are tiny — median ~20x17 px, 90th percentile ~61x48 px. They get upscaled to
   280x280 (matching Hailo's efficientnet_lite3 input) with bicubic interpolation.
   This is inherent to your detector's crop sizes, not a bug, but it's worth knowing
   that a lot of the 280x280 the classifier sees is interpolated, not real detail —
   if accuracy plateaus lower than you'd like, this resolution ceiling is likely why,
   and a smaller model (mobilenet_v2_1.0 / efficientnet_lite0) may do just as well
   for less compute since neither has real high-frequency detail to exploit anyway.
--------------------------------------------------------------------------------------
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T

try:
    import timm
except ImportError:
    sys.exit("timm is required: pip install timm")


# --------------------------------------------------------------------------------
# 1. Dataset scanning — find every usable image, skip zero-byte/corrupt ones, and
#    report exactly what was found so a bad export doesn't fail silently.
# --------------------------------------------------------------------------------

def is_readable_image(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def load_classes_file(path: Path):
    """Read a fixed class-name-to-ID ordering from a classes.txt (one name per
    line, blank lines ignored). Position in the file = class index."""
    names = [line.strip() for line in path.read_text().splitlines()]
    return [n for n in names if n]


def scan_split(split_dir: Path, canonical_classes=None):
    """Returns (dict[class_name] -> list[Path] of *valid* images, stats dict)."""
    stats = {"total_seen": 0, "valid": 0, "zero_byte_or_corrupt": 0, "missing_classes": []}
    class_files = {}

    if canonical_classes is None:
        if not split_dir.is_dir():
            return class_files, stats
        canonical_classes = sorted(
            p.name for p in split_dir.iterdir() if p.is_dir()
        )

    for cls in canonical_classes:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            stats["missing_classes"].append(cls)
            class_files[cls] = []
            continue
        valid = []
        for f in cls_dir.iterdir():
            if not f.is_file():
                continue
            stats["total_seen"] += 1
            if is_readable_image(f):
                valid.append(f)
                stats["valid"] += 1
            else:
                stats["zero_byte_or_corrupt"] += 1
        class_files[cls] = valid

    return class_files, stats


def print_scan_report(name, class_files, stats):
    total_valid = sum(len(v) for v in class_files.values())
    print(f"\n[{name}] seen={stats['total_seen']}  valid={stats['valid']}  "
          f"zero_byte_or_corrupt={stats['zero_byte_or_corrupt']}")
    if stats["missing_classes"]:
        print(f"  classes with NO folder in this split: {stats['missing_classes']}")
    if total_valid == 0:
        print(f"  *** WARNING: {name} has ZERO usable images. ***")
        return
    counts = sorted(((c, len(v)) for c, v in class_files.items()), key=lambda x: -x[1])
    worst = [c for c, n in counts if n == 0]
    if worst:
        print(f"  classes with a folder but 0 usable images: {worst}")
    print(f"  largest class: {counts[0]}   smallest non-empty: "
          f"{min(((c, n) for c, n in counts if n > 0), key=lambda x: x[1])}")


def make_val_from_train(train_files: dict, frac=0.15, seed=42):
    """Stratified split-off from train when val/ has nothing usable."""
    rng = random.Random(seed)
    new_train, new_val = {}, {}
    for cls, files in train_files.items():
        files = files[:]
        rng.shuffle(files)
        if len(files) <= 1:
            new_train[cls] = files
            new_val[cls] = []
            continue
        n_val = max(1, int(len(files) * frac))
        new_val[cls] = files[:n_val]
        new_train[cls] = files[n_val:]
    return new_train, new_val


# --------------------------------------------------------------------------------
# 2. Dataset / DataLoader
# --------------------------------------------------------------------------------

class CropDataset(Dataset):
    def __init__(self, class_files: dict, class_to_idx: dict, transform):
        self.samples = []
        for cls, files in class_files.items():
            idx = class_to_idx[cls]
            for f in files:
                self.samples.append((f, idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # Defensive: if a file that passed the initial verify() scan still fails to
        # decode at read time, don't crash the whole run — fall back to another
        # random sample and move on. Given this dataset's known corruption, this is
        # worth having rather than losing hours of training to one bad file.
        for _ in range(5):
            path, label = self.samples[i]
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                return self.transform(im), label
            except Exception as e:
                print(f"  [warn] failed to read {path} at runtime ({e}); "
                      f"substituting a random sample")
                i = random.randrange(len(self.samples))
        raise RuntimeError("Too many unreadable samples in a row — check your data.")


def build_sampler(class_files: dict, class_to_idx: dict):
    """Inverse-frequency weighted sampler so rare classes aren't drowned out by
    sig_stop's ~3200 examples."""
    counts = {class_to_idx[c]: max(len(v), 1) for c, v in class_files.items()}
    weights = []
    for cls, files in class_files.items():
        w = 1.0 / counts[class_to_idx[cls]]
        weights.extend([w] * len(files))
    if not weights:
        return None
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# --------------------------------------------------------------------------------
# 3. Transforms — EfficientNet-Lite uses [-1, 1] scaling, not ImageNet mean/std.
#    Pulled from the model's own pretrained_cfg so it's always correct, not hardcoded.
# --------------------------------------------------------------------------------

def build_transforms(img_size, mean, std):
    train_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        T.RandomRotation(degrees=7),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------------

def set_backbone_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if "classifier" in name or "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def evaluate(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    train_dir, val_dir, test_dir = data_root / "train", data_root / "val", data_root / "test"

    print("=" * 88)
    print("Scanning dataset ...")

    if args.classes_file:
        classes = load_classes_file(Path(args.classes_file))
        print(f"\nUsing fixed class order from {args.classes_file} ({len(classes)} classes)")
        train_files, train_stats = scan_split(train_dir, canonical_classes=classes)
        on_disk = sorted(p.name for p in train_dir.iterdir() if p.is_dir()) if train_dir.is_dir() else []
        extra = sorted(set(on_disk) - set(classes))
        if extra:
            sys.exit(f"{args.classes_file} is missing class(es) found under {train_dir}: {extra}. "
                      f"Update the classes file so every folder is accounted for before training.")
    else:
        train_files, train_stats = scan_split(train_dir)
        classes = sorted(train_files.keys())
        print(f"\n*** No --classes-file given: class IDs derived from alphabetical folder "
              f"order. These IDs will only match another run/consumer if it uses the exact "
              f"same derivation — pass --classes-file to pin a fixed, portable ID scheme. ***")
    if not classes:
        sys.exit(f"No class folders found under {train_dir}")
    val_files, val_stats = scan_split(val_dir, canonical_classes=classes)
    test_files, test_stats = scan_split(test_dir, canonical_classes=classes)

    print_scan_report("train", train_files, train_stats)
    print_scan_report("val", val_files, val_stats)
    print_scan_report("test", test_files, test_stats)

    zero_train_classes = [c for c, files in train_files.items() if len(files) == 0]
    if zero_train_classes:
        print(f"\n*** {len(zero_train_classes)} class(es) have ZERO usable TRAINING "
              f"images: {zero_train_classes} ***")
        print("    These cannot be learned this run, however long you train — the model "
              "will never predict them correctly. This is a data problem (files read as "
              "placeholders/corrupt), not something more epochs will fix. Re-check the "
              "source folders for these classes before trusting this run's results.")

    if sum(len(v) for v in val_files.values()) == 0:
        print(f"\n*** val/ has nothing usable — carving {args.val_fallback_frac:.0%} "
              f"out of train/ instead. Fix your val export; this is a stopgap. ***")
        train_files, val_files = make_val_from_train(train_files, frac=args.val_fallback_frac,
                                                       seed=args.seed)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print(f"\nLoading tf_efficientnet_lite3 (ImageNet-pretrained, head resized to "
          f"{len(classes)} classes) ...")
    model = timm.create_model("tf_efficientnet_lite3", pretrained=True, num_classes=len(classes))
    cfg = getattr(model, "pretrained_cfg", None) or model.default_cfg
    mean, std = cfg["mean"], cfg["std"]
    print(f"  normalization from model cfg: mean={mean} std={std}")
    model.to(device)

    train_tf, eval_tf = build_transforms(args.img_size, mean, std)
    train_ds = CropDataset(train_files, class_to_idx, train_tf)
    val_ds = CropDataset(val_files, class_to_idx, eval_tf)
    test_ds = CropDataset(test_files, class_to_idx, eval_tf)

    sampler = build_sampler(train_files, class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.workers, drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers) if len(val_ds) else None)
    test_loader = (DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers) if len(test_ds) else None)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        frozen_phase = epoch < args.freeze_epochs
        set_backbone_trainable(model, trainable=not frozen_phase)
        lr = args.head_lr if frozen_phase else args.finetune_lr
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        model.train()
        t0 = time.time()
        running_loss, running_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        val_acc = evaluate(model, val_loader, device)
        phase = "frozen-backbone" if frozen_phase else "fine-tune"
        val_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
        print(f"epoch {epoch+1:03d}/{args.epochs} [{phase}] "
              f"loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_str} "
              f"({time.time()-t0:.1f}s)")

        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "img_size": args.img_size}, best_path)
        elif val_acc is None:
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "img_size": args.img_size}, best_path)

    print(f"\nBest val_acc: {best_acc if best_acc >= 0 else 'n/a (no usable val set)'}")

    if test_loader is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_acc = evaluate(model, test_loader, device)
        print(f"Test accuracy (best checkpoint): {test_acc}")
    else:
        print("Skipping test evaluation — test/ had zero usable images.")

    return best_path, classes


# --------------------------------------------------------------------------------
# 5. ONNX export — raw logits, no softmax baked in (Hailo applies softmax on-device
#    per the efficientnet_lite3.yaml postprocessing config).
# --------------------------------------------------------------------------------

def export_onnx(best_path, args):
    ckpt = torch.load(best_path, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    model = timm.create_model("tf_efficientnet_lite3", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    onnx_path = os.path.join(args.output_dir, "efficientnet_lite3_custom.onnx")
    export_kwargs = dict(
        input_names=["images"], output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch=1, matches Hailo compile expectations
    )
    try:
        # dynamo=False forces the classic TorchScript-based exporter: no onnxscript
        # dependency, and a simpler/more predictable node-naming scheme for the
        # Hailo parser than the newer dynamo-based exporter (torch >= 2.5-ish).
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions don't have the `dynamo` kwarg at all — just use
        # whatever exporter that version ships with.
        torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
    print(f"\nExported ONNX: {onnx_path}")
    print(f"Classes ({len(classes)}): {classes}")
    print("\nNext steps: adapt efficientnet_lite3.yaml (network_path -> this .onnx, "
          "info.output_shape -> the class count above, parser.nodes -> 'logits'), "
          "then hailomz parse / optimize / compile.")
    return onnx_path


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", help="Path to the dataset root (contains train/val/test)")
    ap.add_argument("--output-dir", required=True, help="Where to save best_model.pt, classes.json, and the ONNX export")
    ap.add_argument("--classes-file", default=None,
                     help="Path to a classes.txt with one class name per line, fixing class "
                          "ID = line position (e.g. master_thesis/data/zeus-cropped/classes.txt). "
                          "Without this, IDs are derived by sorting train/ folder names "
                          "alphabetically, which won't match another run's ID scheme.")
    ap.add_argument("--export-onnx", action="store_false", help="Export the model to ONNX format")
    ap.add_argument("--img-size", type=int, default=280)  # matches Hailo's efficientnet_lite3 cfg
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--val-fallback-frac", type=float, default=0.15,
                     help="Fraction of train/ used as val if val/ is unusable")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    best_path, classes = train(args)
    if should_export_onnx := getattr(args, "export_onnx", False):
        export_onnx(best_path, args)


if __name__ == "__main__":
    main()
