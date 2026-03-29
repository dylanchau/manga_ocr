# tools/download_roboflow.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads the Manga Text Detection dataset from Roboflow Universe and
# places it exactly where train_detect.py expects it.
#
# Usage (on EC2):
#   cd ~/manga_ocr
#   pip install roboflow
#   python tools/download_roboflow.py --api-key YOUR_KEY_HERE
#
# After running, your folder will look like:
#   data/processed/detection/
#     train/
#       images/   ← manga pages
#       labels/   ← YOLO .txt files
#     valid/
#       images/
#       labels/
#     test/
#       images/
#       labels/
#     data.yaml   ← ready to pass to YOLOv8
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import shutil
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PATH as BASE_DIR_CONFIG

# ── Dataset identifiers (from the Roboflow URL) ───────────────────────────────
# URL: universe.roboflow.com/ocr-9ocgg/manga-text-detection-xyvbw-iipaw
WORKSPACE   = "mangadetector-listt"
PROJECT     = "my_data-pdeqe"
VERSION     = 1          # increment if you want a newer exported version

# Option B — sam64 (5240 images, best choice)
#WORKSPACE = "sam64-t4u3d"
#PROJECT   = "manga-translator-detection-r1kli"
#VERSION   = 1   # check with project.versions() if this fails too

# ── Where to put it in the project ───────────────────────────────────────────
BASE_DIR     = os.path.expanduser(BASE_DIR_CONFIG)
DETECT_DIR   = os.path.join(BASE_DIR, "data", "processed", "detection")


def download(api_key: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow package not found. Installing...")
        os.system(f"{sys.executable} -m pip install -q roboflow")
        from roboflow import Roboflow

    print(f"Connecting to Roboflow...")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    for v in project.versions():
        print(f"{PROJECT} versions: ", v.version)
    VERSION = project.versions()[0].version

    print(f"Downloading dataset v{VERSION} in YOLOv8 format...")
    dataset = project.version(VERSION).download(
        model_format = "yolov8",
        location     = DETECT_DIR,
        overwrite    = True,
    )

    print(f"\nDownloaded to: {dataset.location}")
    _verify_and_fix_yaml(dataset.location)
    _print_summary(dataset.location)


def _verify_and_fix_yaml(dataset_dir: str):
    """
    Roboflow writes absolute paths into data.yaml that point to wherever
    the download happened.  We rewrite them to be relative so the file
    works on any machine or after moving the folder.
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[warn] data.yaml not found at {yaml_path}")
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Roboflow sometimes writes absolute paths — make them relative
    for key in ("train", "val", "valid", "test"):
        if key in cfg and os.path.isabs(cfg[key]):
            cfg[key] = os.path.relpath(cfg[key], dataset_dir)

    # Normalise "valid" → "val" so YOLOv8 recognises it
    if "valid" in cfg and "val" not in cfg:
        cfg["val"] = cfg.pop("valid")

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"data.yaml updated: {yaml_path}")
    print(f"  train : {cfg.get('train')}")
    print(f"  val   : {cfg.get('val')}")
    print(f"  nc    : {cfg.get('nc')}  classes: {cfg.get('names')}")


def _print_summary(dataset_dir: str):
    """Count images and labels in each split."""
    print("\n── Dataset summary ──────────────────────────────────")
    total_images = 0
    for split in ("train", "valid", "val", "test"):
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue
        n_img = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        n_lbl = len([f for f in os.listdir(lbl_dir)
                     if f.endswith(".txt")]) if os.path.exists(lbl_dir) else 0
        print(f"  {split:6s}  {n_img:4d} images   {n_lbl:4d} labels")
        total_images += n_img

    print(f"\n  Total: {total_images} images")
    print("─────────────────────────────────────────────────────")
    print("\nNext step — train the detection model:")
    print(f"  cd ~/manga_ocr && python train_detect.py")


# ── Bonus: auto-crop detected regions to build recognition dataset ─────────────
def crop_for_recognition(dataset_dir: str, output_dir: str):
    """
    The Roboflow dataset has bounding boxes but NO text labels.
    This function crops each box from the image so you can manually
    type the text content into a labels.json later.

    It produces:
      output_dir/
        crops/
          000001.png
          000002.png
          ...
        to_annotate.json   ← list of crop paths, text field empty — fill these in!
    """
    import cv2
    import json

    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    entries    = []
    crop_idx   = 0
    skipped    = 0

    for split in ("train", "valid", "val"):
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir,
                                    os.path.splitext(img_file)[0] + ".txt")

            if not os.path.exists(lbl_path):
                continue

            image    = cv2.imread(img_path)
            if image is None:
                continue
            ih, iw   = image.shape[:2]

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue

                # YOLO format: class cx cy w h (all normalised)
                _, cx, cy, w, h = map(float, parts)

                x1 = int((cx - w / 2) * iw) - 4
                y1 = int((cy - h / 2) * ih) - 4
                x2 = int((cx + w / 2) * iw) + 4
                y2 = int((cy + h / 2) * ih) + 4

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)

                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                    skipped += 1
                    continue

                crop_name = f"{crop_idx:06d}.png"
                cv2.imwrite(os.path.join(crops_dir, crop_name), crop)

                entries.append({
                    "image": os.path.join(crops_dir, crop_name),
                    "text":  "",    # ← YOU fill this in (see instructions below)
                    "source_image": img_path,
                })
                crop_idx += 1

    # Save the stub JSON
    out_json = os.path.join(output_dir, "to_annotate.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\n── Crop summary ─────────────────────────────────────")
    print(f"  {crop_idx} crops saved to: {crops_dir}")
    if skipped:
        print(f"  {skipped} crops skipped (too small)")
    print(f"\n  Stub JSON saved to: {out_json}")
    print()
    print("  The 'text' field in to_annotate.json is empty.")
    print("  Fill it in using one of these approaches:")
    print()
    print("  Option A (fast): Use manga-ocr to pre-fill text automatically,")
    print("    then manually correct mistakes:")
    print("    python tools/prefill_text.py --json", out_json)
    print()
    print("  Option B (manual): Open each crop image, read it, type the text.")
    print("    Then rename to_annotate.json → labels.json when done.")
    print("─────────────────────────────────────────────────────")


# ── Pre-fill text using manga-ocr (run after crop_for_recognition) ────────────
def prefill_text_with_manga_ocr(json_path: str):
    """
    Uses the pre-trained manga-ocr model to automatically fill the 'text'
    field in to_annotate.json.

    This gives you a starting point — manga-ocr is not perfect, so review
    and correct each entry before using for training.

    Usage:
        python tools/download_roboflow.py --prefill --json to_annotate.json
    """
    import json
    from PIL import Image

    try:
        from manga_ocr import MangaOcr
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q manga-ocr")
        from manga_ocr import MangaOcr

    print("Loading manga-ocr model (~400 MB download on first run)...")
    mocr = MangaOcr()

    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)

    empty   = [e for e in entries if not e["text"].strip()]
    total   = len(empty)
    print(f"Pre-filling {total} empty entries...")

    for i, entry in enumerate(entries):
        if entry["text"].strip():
            continue   # already has text — skip

        try:
            img  = Image.open(entry["image"]).convert("RGB")
            text = mocr(img)
            entry["text"] = text
        except Exception as ex:
            print(f"  [warn] {entry['image']}: {ex}")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} done...")

    # Save back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    filled = sum(1 for e in entries if e["text"].strip())
    print(f"\nDone. {filled}/{len(entries)} entries now have text.")
    print(f"Saved: {json_path}")
    print()
    print("IMPORTANT: Review the text values — manga-ocr makes mistakes.")
    print("When you're satisfied, rename the file to labels.json.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Roboflow manga dataset and prepare for training"
    )
    parser.add_argument("--api-key",  help="Your Roboflow API key")
    parser.add_argument("--crop",     action="store_true",
                        help="Crop detected regions for recognition dataset")
    parser.add_argument("--prefill",  action="store_true",
                        help="Pre-fill text using manga-ocr (run after --crop)")
    parser.add_argument("--json",     default=None,
                        help="Path to to_annotate.json (used with --prefill)")
    args = parser.parse_args()

    if args.prefill:
        json_path = args.json or os.path.join(
            BASE_DIR, "data", "processed", "recognition", "to_annotate.json"
        )
        prefill_text_with_manga_ocr(json_path)

    elif args.crop:
        rec_dir = os.path.join(BASE_DIR, "data", "processed", "recognition")
        os.makedirs(rec_dir, exist_ok=True)
        crop_for_recognition(DETECT_DIR, rec_dir)

    elif args.api_key:
        download(args.api_key)

    else:
        parser.print_help()