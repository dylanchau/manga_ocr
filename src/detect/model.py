# src/detect/model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from config import MODELS_DIR, BATCH_SIZE, CONF_THRESHOLD


def get_detection_model(pretrained: bool = True) -> YOLO:
    """
    Load a YOLOv8n model.
    pretrained=True  → start from Ultralytics weights (recommended)
    pretrained=False → random weights, needs much more data
    """
    return YOLO("yolov8n.pt" if pretrained else "yolov8n.yaml")


def train_detection(data_yaml: str, epochs: int = 50, img_size: int = 640):
    """
    Fine-tune YOLOv8 on your manga dataset.
    Call this from train_detect.py, not directly.
    """
    model = get_detection_model(pretrained=True)

    results = model.train(
        data        = data_yaml,
        epochs      = epochs,
        imgsz       = img_size,
        batch       = BATCH_SIZE,
        patience    = 10,
        save        = True,
        save_period = 5,
        project     = MODELS_DIR,
        name        = "detect",
        device      = "0",
        augment     = True,
        exist_ok    = True,
    )
    return results


def detect_text_regions(model_path: str, image_path: str,
                        conf: float = CONF_THRESHOLD) -> list:
    """
    Run detection on one image.
    Returns [(x1, y1, x2, y2, confidence), ...]
    """
    model   = YOLO(model_path)
    results = model(image_path, conf=conf)[0]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0])))
    return boxes
