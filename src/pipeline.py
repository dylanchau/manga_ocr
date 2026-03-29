# src/pipeline.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from config import DEVICE, CONF_THRESHOLD, NUM_CLASSES, RECOG_HEIGHT, RECOG_WIDTH
from src.recognize.model import CRNN
from src.dataset import decode_indices
from src.preprocess import preprocess_for_recognition


class MangaOCR:
    def __init__(self, detect_model_path: str, recog_model_path: str):
        print("Loading detection model...")
        self.detector = YOLO(detect_model_path)

        print("Loading recognition model...")
        self.recognizer = CRNN(NUM_CLASSES).to(DEVICE)
        self.recognizer.load_state_dict(
            torch.load(recog_model_path, map_location=DEVICE)
        )
        self.recognizer.eval()
        print("Models loaded.")

    def detect(self, image: np.ndarray) -> list:
        """Return [(x1, y1, x2, y2, confidence), ...]"""
        results = self.detector(image, conf=CONF_THRESHOLD)[0]
        boxes   = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    def recognize(self, crop: np.ndarray) -> str:
        img    = preprocess_for_recognition(crop, RECOG_HEIGHT, RECOG_WIDTH)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            logits = self.recognizer(tensor)          # (T, 1, C)
            preds  = logits.softmax(2).argmax(2)      # (T, 1)
            preds  = preds.squeeze(1).tolist()        # [T]
        return decode_indices(preds)

    def sort_reading_order(self, boxes: list) -> list:
        """Right-to-left, top-to-bottom — standard manga reading order."""
        if not boxes:
            return boxes

        sorted_boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        rows, cur    = [], [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            cy      = (box[1] + box[3]) / 2
            last_cy = (cur[-1][1] + cur[-1][3]) / 2
            if abs(cy - last_cy) < 60:
                cur.append(box)
            else:
                rows.append(cur)
                cur = [box]
        rows.append(cur)

        ordered = []
        for row in rows:
            row.sort(key=lambda b: b[0], reverse=True)
            ordered.extend(row)
        return ordered

    def run(self, image_path: str) -> list:
        """
        Full pipeline on one image.
        Returns [{"box": (x1,y1,x2,y2), "conf": float, "text": str}, ...]
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        boxes   = self.detect(image)
        boxes   = self.sort_reading_order(boxes)
        results = []

        for (x1, y1, x2, y2, conf) in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results.append({
                "box":  (x1, y1, x2, y2),
                "conf": conf,
                "text": self.recognize(crop),
            })
        return results

    def visualize(self, image_path: str, results: list,
                  save_path: str = None) -> np.ndarray:
        img = cv2.imread(image_path)
        for item in results:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, item["text"], (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Saved → {save_path}")
        return img
