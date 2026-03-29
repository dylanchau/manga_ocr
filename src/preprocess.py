# src/preprocess.py
"""
Prepares manga images for training.

What this does:
  1. Converts to grayscale (manga is usually black & white)
  2. Applies adaptive thresholding to separate ink from paper
  3. Resizes to a standard size
  4. Normalizes pixel values to [0, 1]
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_image(path: str) -> np.ndarray:
    """Load image from disk as a NumPy array (BGR format)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding: makes text pure black, background pure white.
    Better than a fixed threshold because manga scans vary in brightness.
    """
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,    # Neighbourhood size (must be odd)
        C=8              # Constant subtracted from mean
    )


def denoise(img: np.ndarray) -> np.ndarray:
    """Remove small noise dots from scanned manga pages."""
    # fastNlMeansDenoising only works on grayscale
    return cv2.fastNlMeansDenoising(img, h=10)


def resize_for_detection(img: np.ndarray, size=(640, 640)) -> np.ndarray:
    """Resize image to the detection model's expected input size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0,255] to [0,1] as float32."""
    return img.astype(np.float32) / 255.0


def preprocess_for_recognition(
    crop: np.ndarray,
    target_h: int = None,
    target_w: int = None,
) -> np.ndarray:

    # Resolve defaults here, not at definition time
    if target_h is None:
        from config import RECOG_HEIGHT
        target_h = RECOG_HEIGHT
    if target_w is None:
        from config import RECOG_WIDTH
        target_w = RECOG_WIDTH

    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = crop.shape

    # Guard: if height is 0 return a blank image instead of crashing
    if h == 0 or w == 0:
        return np.ones((target_h, target_w), dtype=np.float32)

    new_w = min(int(w * (target_h / h)), target_w)

    # Guard: new_w must be at least 1 pixel
    new_w = max(new_w, 1)

    crop = cv2.resize(crop, (new_w, target_h))

    if new_w < target_w:
        pad  = np.full((target_h, target_w - new_w), 255, dtype=np.uint8)
        crop = np.hstack([crop, pad])

    return normalize(crop)

def full_pipeline(image_path: str) -> dict:
    """
    Run all preprocessing steps and return a dict with each result.
    Useful for debugging — you can visualize each stage.
    """
    original = load_image(image_path)
    gray     = to_grayscale(original)
    denoised = denoise(gray)
    binary   = binarize(denoised)
    resized  = resize_for_detection(binary)
    normalized = normalize(resized)

    return {
        "original":   original,
        "gray":       gray,
        "denoised":   denoised,
        "binary":     binary,
        "resized":    resized,
        "normalized": normalized,
    }


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.jpg"
    stages = full_pipeline(path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["original", "gray", "binary", "resized"]
    for ax, key in zip(axes, titles):
        img = stages[key]
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(key)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("preprocess_stages.png")
    print("Saved: preprocess_stages.png")
