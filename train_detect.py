"""
train_detect.py — YOLOv8 text detection training for EC2
─────────────────────────────────────────────────────────
Usage:
    cd ~/manga_ocr
    python train_detect.py
 
Fix applied: all training code is inside if __name__ == '__main__'
because YOLOv8 uses multiprocessing internally for data loading.
Without this guard Python tries to re-import the script in each
worker process, which causes the RuntimeError you saw.
"""
 
import os
import sys
 
# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
 
def main():
    from ultralytics import YOLO
    from config import (
        MODELS_DIR, DETECT_MODEL_DIR,
        IMG_SIZE, BATCH_SIZE, EPOCHS_DETECT,
        DEVICE, RESUME_DETECT,
        PATH as BASE_DIR_CONFIG,
    )
 
    BASE_DIR  = os.path.expanduser(BASE_DIR_CONFIG)
    DATA_YAML = os.path.join(BASE_DIR, "data", "processed", "detection", "data.yaml")
 
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        print("Run tools/convert_annotations.py or tools/download_roboflow.py first.")
        sys.exit(1)
 
    # ── Resume or fresh start ─────────────────────────────────────────────────
    last_weights = os.path.join(DETECT_MODEL_DIR, "weights", "last.pt")
    best_weights = os.path.join(DETECT_MODEL_DIR, "weights", "best.pt")
 
    if RESUME_DETECT and os.path.exists(last_weights):
        print(f"Resuming from: {last_weights}")
        model  = YOLO(last_weights)
        resume = True
    else:
        print("Starting fresh from pre-trained yolov8n.pt")
        model  = YOLO("yolov8n.pt")
        resume = False
 
    print(f"Dataset : {DATA_YAML}")
    print(f"Epochs  : {EPOCHS_DETECT}")
    print(f"Device  : {DEVICE}")
    print()
 
    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS_DETECT,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        patience    = 15,
        save        = True,
        save_period = 5,
        project     = MODELS_DIR,
        name        = "detect",
        device      = "0" if DEVICE == "cuda" else "cpu",
        augment     = True,
        exist_ok    = True,
        resume      = resume,
        plots       = True,
        verbose     = True,
        # workers=0 is a safe fallback if multiprocessing causes issues
        # on your OS. Uncomment the line below only if you still get errors.
        # workers     = 0,
    )
 
    print("\n============================")
    print("Detection training complete!")
    print(f"Best model: {best_weights}")
    print("============================")
 
    # ── Optional S3 backup ────────────────────────────────────────────────────
    S3_BUCKET = ""   # e.g. "my-manga-ocr-bucket" — leave empty to skip
    if S3_BUCKET and os.path.exists(best_weights):
        import subprocess
        subprocess.run(
            f"aws s3 cp {best_weights} s3://{S3_BUCKET}/detect/best.pt",
            shell=True,
        )
        print(f"Backed up to s3://{S3_BUCKET}/detect/best.pt")
 
 
# ── This guard is the fix ─────────────────────────────────────────────────────
# YOLOv8 uses Python's multiprocessing to load data in parallel.
# On Linux with "spawn" or "forkserver" start methods (and always on Windows),
# Python re-imports this script in every worker process.
# Without the guard, each worker tries to call model.train() again → crash.
# With the guard, workers import the file but skip straight to __name__ check
# and do nothing — only the original process runs main().
if __name__ == "__main__":
    main()
