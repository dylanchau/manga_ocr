"""
Train YOLOv8 text detection model on EC2.
 
Usage:
    cd ~/manga-ocr
    python train_detect.py
 
What's different from Kaggle version:
  - Full 50 epochs (no 9-hr time cap)
  - Logs saved to ~/manga-ocr/logs/
  - Automatic S3 backup when training finishes (optional)
  - Checkpoint resume via RESUME_DETECT in config.py
"""
 
import os, sys
 
from ultralytics import YOLO
from config import (DATASET_ROOT, MODELS_DIR, DETECT_MODEL_DIR,
                    IMG_SIZE, BATCH_SIZE, EPOCHS_DETECT,
                    DEVICE, RESUME_DETECT, PATH as BASE_DIR_CONFIG)
                    
sys.path.insert(0, os.path.expanduser(BASE_DIR_CONFIG))                    
 
DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")
 
if not os.path.exists(DATA_YAML):
    print(f"ERROR: data.yaml not found at {DATA_YAML}")
    print("Run tools/convert_annotations.py first.")
    sys.exit(1)
 
# Resume from last checkpoint or start fresh
best_weights = os.path.join(DETECT_MODEL_DIR, "weights", "best.pt")
last_weights = os.path.join(DETECT_MODEL_DIR, "weights", "last.pt")
 
if RESUME_DETECT and os.path.exists(last_weights):
    print(f"Resuming from: {last_weights}")
    model = YOLO(last_weights)
    resume = True
else:
    print("Starting fresh from pre-trained yolov8n.pt")
    model = YOLO("yolov8n.pt")
    resume = False
 
print(f"Dataset  : {DATA_YAML}")
print(f"Epochs   : {EPOCHS_DETECT}")
print(f"Device   : {DEVICE}")
print()
 
results = model.train(
    data        = DATA_YAML,
    epochs      = EPOCHS_DETECT,
    imgsz       = IMG_SIZE,
    batch       = BATCH_SIZE,
    patience    = 15,
    save        = True,
    save_period = 5,       # checkpoint every 5 epochs
    project     = MODELS_DIR,
    name        = "detect",
    device      = "0" if DEVICE == "cuda" else "cpu",
    augment     = True,
    exist_ok    = True,    # don't create detect2/, detect3/ on re-runs
    resume      = resume,
    # Logging
    plots       = True,    # saves training curves as images
    verbose     = True,
)
 
print("\n============================")
print("Detection training complete!")
print(f"Best model: {best_weights}")
print("============================")
 
# ── Optional: back up model to S3 ─────────────────────────────────────────
S3_BUCKET = ""   # e.g. "my-manga-ocr-bucket" — leave empty to skip
if S3_BUCKET and os.path.exists(best_weights):
    import subprocess
    cmd = f"aws s3 cp {best_weights} s3://{S3_BUCKET}/models/detect_best.pt"
    print(f"\nBacking up to S3: {cmd}")
    subprocess.run(cmd, shell=True)