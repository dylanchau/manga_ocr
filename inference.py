# inference.py
"""
Run the full manga OCR pipeline on a new image.

Usage:
    cd ~/manga_ocr
    python inference.py path/to/manga_page.jpg
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, RECOG_MODEL_DIR
from src.pipeline import MangaOCR


DETECT_MODEL = os.path.join(MODELS_DIR,      "detect", "weights", "best.pt")
RECOG_MODEL  = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/manga_page.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(DETECT_MODEL):
        print(f"Detection model not found: {DETECT_MODEL}")
        print("Run train_detect.py first.")
        sys.exit(1)

    if not os.path.exists(RECOG_MODEL):
        print(f"Recognition model not found: {RECOG_MODEL}")
        print("Run train_recog.py first.")
        sys.exit(1)

    ocr     = MangaOCR(DETECT_MODEL, RECOG_MODEL)
    results = ocr.run(image_path)

    print(f"\nFound {len(results)} text region(s):\n")
    for i, item in enumerate(results, 1):
        print(f"  [{i}] \"{item['text']}\"  (conf: {item['conf']:.2f})")

    out_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
    ocr.visualize(image_path, results, save_path=out_path)


if __name__ == "__main__":
    main()
