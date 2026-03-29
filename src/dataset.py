# src/dataset.py
"""
PyTorch Dataset classes.

Why do we need this?
  PyTorch models learn from batches of images.
  A Dataset object tells PyTorch: "here's how to load one example".
  A DataLoader then batches them automatically.
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from config import CHARSET, RECOG_HEIGHT, RECOG_WIDTH
from src.preprocess import preprocess_for_recognition


# ── Helpers for encoding / decoding text ───────────────────────────────

CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARSET)}   # blank=0
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARSET)}

def encode_text(text: str) -> list[int]:
    """Convert a string to a list of integer indices."""
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]

def decode_indices(indices: list[int]) -> str:
    """Convert a list of indices back to a string (removes blanks & repeats)."""
    result = []
    prev = None
    for idx in indices:
        if idx != 0 and idx != prev:       # 0 = CTC blank
            result.append(IDX_TO_CHAR.get(idx, "?"))
        prev = idx
    return "".join(result)


# ── Detection Dataset ───────────────────────────────────────────────────

class MangaDetectionDataset(Dataset):
    """
    Dataset for the TEXT DETECTION model.
    
    Expects a folder with:
      - images/   (*.jpg or *.png)
      - labels/   (*.txt  — YOLO format: class cx cy w h, all normalized)
    
    YOLO label format (one line per text box):
      0 0.512 0.340 0.180 0.095
      ^class  ^center_x  ^center_y  ^width  ^height
      (all values are 0–1, relative to image size)
    """

    def __init__(self, root: str, img_size=640, augment=False):
        self.img_dir   = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.img_size  = img_size
        self.augment   = augment

        # Collect all image paths that have a matching label file
        self.samples = [
            f for f in os.listdir(self.img_dir)
            if f.endswith((".jpg", ".png")) and
               os.path.exists(os.path.join(self.label_dir, f.replace(".jpg", ".txt").replace(".png", ".txt")))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        img_path   = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir, fname.rsplit(".", 1)[0] + ".txt")

        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # HWC → CHW

        # Load labels (each row: class cx cy w h)
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(p) for p in parts])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        return img, boxes


# ── Recognition Dataset ─────────────────────────────────────────────────

class MangaRecognitionDataset(Dataset):
    """
    Dataset for the TEXT RECOGNITION model.
    
    Expects a JSON file with entries like:
      [{"image": "crops/001.png", "text": "Hello world"}, ...]
    
    Images should be pre-cropped text regions.
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"[warn] Cannot read image: {entry['image']} — using blank")
            img = np.ones((RECOG_HEIGHT, RECOG_WIDTH), dtype=np.uint8) * 255


        img = preprocess_for_recognition(img, target_h=RECOG_HEIGHT, target_w=RECOG_WIDTH)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)   # (1, H, W)

        label = encode_text(entry["text"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor, len(label)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    return (
        torch.stack(images),
        torch.cat(labels),
        torch.tensor(lengths, dtype=torch.long)
    )

# Alias — train_recog.py imports it by this name
collate_recognition = collate_fn
