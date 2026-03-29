# tools/evaluate.py
"""
Measure OCR accuracy using CER and WER.

Usage:
    cd ~/manga_ocr
    python tools/evaluate.py --json data/test_labels.json

test_labels.json format:
    [{"image": "path/to/page.jpg", "texts": ["bubble1 text", "bubble2 text"]}, ...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import editdistance

from config import MODELS_DIR, RECOG_MODEL_DIR
from src.pipeline import MangaOCR


DETECT_MODEL = os.path.join(MODELS_DIR,      "detect", "weights", "best.pt")
RECOG_MODEL  = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")


def cer(pred: str, truth: str) -> float:
    """Character Error Rate — fraction of characters that are wrong."""
    if not truth:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, truth) / len(truth)


def wer(pred: str, truth: str) -> float:
    """Word Error Rate — fraction of words that are wrong."""
    p, t = pred.split(), truth.split()
    if not t:
        return 0.0
    return editdistance.eval(p, t) / len(t)


def evaluate(test_json: str):
    ocr = MangaOCR(DETECT_MODEL, RECOG_MODEL)

    with open(test_json, encoding="utf-8") as f:
        data = json.load(f)

    all_cer, all_wer = [], []

    for item in data:
        results = ocr.run(item["image"])
        preds   = [r["text"] for r in results]

        for pred, truth in zip(preds, item["texts"]):
            all_cer.append(cer(pred, truth))
            all_wer.append(wer(pred, truth))

    avg_cer = sum(all_cer) / len(all_cer) if all_cer else 0.0
    avg_wer = sum(all_wer) / len(all_wer) if all_wer else 0.0

    print(f"Samples evaluated : {len(all_cer)}")
    print(f"CER               : {avg_cer:.4f}  ({avg_cer*100:.1f}% char errors)")
    print(f"WER               : {avg_wer:.4f}  ({avg_wer*100:.1f}% word errors)")
    print()
    print("Target: CER < 0.10 (below 10% is solid for a first model)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,
                        help="Path to test labels JSON")
    args = parser.parse_args()
    evaluate(args.json)
