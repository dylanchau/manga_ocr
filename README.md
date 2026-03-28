```
#!/bin/bash
# =============================================================================
# EC2 Setup Script for Manga OCR Training
# =============================================================================
# STEP 0: Launch your EC2 instance first (do this in AWS Console):
#
#   AMI    : "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (search in AMI catalog)
#            This AMI has CUDA, PyTorch, and conda pre-installed — saves 30 min setup.
#   Type   : g4dn.xlarge
#   Storage: 50 GB gp3 EBS (root volume) — manga images + models need space
#   Key    : Create or select a .pem key pair — you need this to SSH in
#   Security group: allow SSH (port 22) from your IP only
#
# STEP 1: SSH into your instance
#   chmod 400 your-key.pem
#   ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
#
# STEP 2: Run this script
#   curl -O https://... (or paste it manually)
#   bash ec2_setup.sh
# =============================================================================

set -e   # stop on first error

echo "======================================================"
echo " Manga OCR — EC2 environment setup"
echo "======================================================"

# ── Verify GPU is visible ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi
# You should see a T4 GPU listed. If this fails, you're on the wrong instance type.

# ── Update system packages ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    tmux \
    htop \
    tree \
    unzip \
    awscli     # for uploading results to S3 (optional but useful)

# ── Install Python packages ────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python packages..."
# The Deep Learning AMI uses conda. Activate the PyTorch env it ships with.
source activate pytorch   2>/dev/null || true

pip install -q \
    ultralytics \
    opencv-python-headless \
    editdistance \
    manga-ocr \
    roboflow \
    pyyaml \
    matplotlib \
    tqdm \
    boto3      # AWS SDK — for optional S3 backup

# ── Create project directory structure ────────────────────────────────────────
echo ""
echo "[4/6] Creating project structure..."
mkdir -p ~/manga_ocr/{data/{raw,processed/{detection/{train/{images,labels},val/{images,labels}},recognition/crops}},models/{detect/weights,recog},logs,tools,src/{detect,recognize}}

echo "Project structure created at ~/manga_ocr/"
tree ~/manga_ocr/ -L 3 2>/dev/null || find ~/manga_ocr -type d | head -30

# ── Verify PyTorch + CUDA ──────────────────────────────────────────────────────
echo ""
echo "[5/6] Verifying PyTorch + CUDA..."
python3 -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.cuda.is_available()}')
print(f'GPU name : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')
"

# ── tmux tip ──────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "======================================================"
echo " IMPORTANT: Use tmux for long training runs"
echo "======================================================"
echo " If your SSH connection drops, training keeps running"
echo " inside a tmux session."
echo ""
echo " Start a session : tmux new -s train"
echo " Detach (keep running): Ctrl+B then D"
echo " Reattach later  : tmux attach -t train"
echo " List sessions   : tmux ls"
echo "======================================================"


# =============================================================================
# config.py  — EC2 paths (replaces the Kaggle version)
# Save this as ~/manga_ocr/config.py
# =============================================================================
cat > ~/manga_ocr/config.py << 'PYTHON'
import os
import torch

# ── Paths (EC2) ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.expanduser("~/manga_ocr")
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED= os.path.join(BASE_DIR, "data", "processed")

DATASET_ROOT  = os.path.join(DATA_PROCESSED, "detection")
RECOG_JSON    = os.path.join(DATA_PROCESSED, "recognition", "labels.json")

MODELS_DIR        = os.path.join(BASE_DIR, "models")
DETECT_MODEL_DIR  = os.path.join(MODELS_DIR, "detect")
RECOG_MODEL_DIR   = os.path.join(MODELS_DIR, "recog")
LOG_DIR           = os.path.join(BASE_DIR, "logs")

for d in [MODELS_DIR, DETECT_MODEL_DIR, RECOG_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Image sizes ───────────────────────────────────────────────────────────────
IMG_SIZE  = 640
RECOG_H   = 32
RECOG_W   = 128

# ── Character set — paste output of charset_builder.py here ──────────────────
CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
    # Add your Japanese characters here after running charset_builder.py
)
NUM_CLASSES = len(CHARSET) + 1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS_DETECT  = 50    # No time limit on EC2 — use full epochs
EPOCHS_RECOG   = 100
LR             = 1e-3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.5

# ── Checkpoint resume ─────────────────────────────────────────────────────────
RESUME_DETECT  = False
RESUME_RECOG   = False

print(f"Config loaded. Device: {DEVICE}  |  Charset size: {NUM_CLASSES - 1}")
PYTHON

echo "config.py written to ~/manga_ocr/config.py"


# =============================================================================
# train_detect.py  — Detection training script (EC2 version)
# Save as ~/manga_ocr/train_detect.py
# =============================================================================
cat > ~/manga_ocr/train_detect.py << 'PYTHON'
"""
Train YOLOv8 text detection model on EC2.

Usage:
    cd ~/manga_ocr
    python train_detect.py

What's different from Kaggle version:
  - Full 50 epochs (no 9-hr time cap)
  - Logs saved to ~/manga_ocr/logs/
  - Automatic S3 backup when training finishes (optional)
  - Checkpoint resume via RESUME_DETECT in config.py
"""

import os, sys
sys.path.insert(0, os.path.expanduser("~/manga_ocr"))

from ultralytics import YOLO
from config import (DATASET_ROOT, MODELS_DIR, DETECT_MODEL_DIR,
                    IMG_SIZE, BATCH_SIZE, EPOCHS_DETECT,
                    DEVICE, RESUME_DETECT)

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
PYTHON

echo "train_detect.py written."


# =============================================================================
# train_recog.py  — Recognition training script (EC2 version)
# Save as ~/manga_ocr/train_recog.py
# =============================================================================
cat > ~/manga_ocr/train_recog.py << 'PYTHON'
"""
Train CRNN text recognition model on EC2.

Usage:
    cd ~/manga_ocr
    python train_recog.py

What's different from Kaggle version:
  - Full 100 epochs (no time cap)
  - Live loss logging to logs/recog_loss.csv  (easy to plot)
  - AMP (mixed precision) for faster training
  - Checkpoint every 10 epochs + best model saved separately
  - Optional S3 backup
"""

import os, sys, csv
sys.path.insert(0, os.path.expanduser("~/manga_ocr"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from config import (RECOG_JSON, RECOG_MODEL_DIR, LOG_DIR,
                    BATCH_SIZE, EPOCHS_RECOG, LR,
                    NUM_CLASSES, DEVICE, RESUME_RECOG)

# ── Import project modules ─────────────────────────────────────────────────
from src.dataset import MangaRecognitionDataset, collate_recognition
from src.recognize.model import CRNN

# ── Validate dataset exists ────────────────────────────────────────────────
if not os.path.exists(RECOG_JSON):
    print(f"ERROR: labels.json not found at {RECOG_JSON}")
    print("Run tools/convert_annotations.py first.")
    sys.exit(1)

# ── Dataset ────────────────────────────────────────────────────────────────
dataset  = MangaRecognitionDataset(RECOG_JSON)
val_size = max(1, int(len(dataset) * 0.1))
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

print(f"Dataset  : {len(dataset)} crops  "
      f"(train: {len(train_ds)}, val: {val_size})")
print(f"Device   : {DEVICE}")
print(f"Epochs   : {EPOCHS_RECOG}")
print()

train_loader = DataLoader(
    train_ds, BATCH_SIZE, shuffle=True,
    collate_fn=collate_recognition,
    num_workers=4,       # EC2 has more CPUs than Kaggle — use them
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds, BATCH_SIZE, shuffle=False,
    collate_fn=collate_recognition,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# ── Model ──────────────────────────────────────────────────────────────────
crnn      = CRNN(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(crnn.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=7, factor=0.5, verbose=True
)
ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

use_amp   = (DEVICE == "cuda")
scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

# ── Resume from checkpoint ─────────────────────────────────────────────────
checkpoint_path = os.path.join(RECOG_MODEL_DIR, "checkpoint_latest.pt")
best_path       = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")
start_epoch     = 1
best_val_loss   = float("inf")

if RESUME_RECOG and os.path.exists(checkpoint_path):
    ckpt          = torch.load(checkpoint_path, map_location=DEVICE)
    crnn.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed from epoch {ckpt['epoch']}  "
          f"(best val loss: {best_val_loss:.4f})")

# ── CSV logger — open in Excel to see training curves ─────────────────────
log_path = os.path.join(LOG_DIR, "recog_loss.csv")
log_file = open(log_path, "a", newline="")
log_csv  = csv.writer(log_file)
if start_epoch == 1:
    log_csv.writerow(["epoch", "train_loss", "val_loss", "lr"])

# ── Training loop ──────────────────────────────────────────────────────────
print(f"Logging losses to: {log_path}")
print()

for epoch in range(start_epoch, EPOCHS_RECOG + 1):

    # Train
    crnn.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS_RECOG} [train]",
                leave=False, ncols=80)

    for imgs, labels, label_lens in pbar:
        imgs       = imgs.to(DEVICE, non_blocking=True)
        labels     = labels.to(DEVICE, non_blocking=True)
        label_lens = label_lens.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits     = crnn(imgs)
            log_probs  = logits.log_softmax(2)
            input_lens = torch.full(
                (imgs.size(0),), logits.size(0),
                dtype=torch.long, device=DEVICE
            )
            loss = ctc_loss(log_probs, labels, input_lens, label_lens)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(crnn.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    # Validate
    crnn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels, label_lens in val_loader:
            imgs       = imgs.to(DEVICE, non_blocking=True)
            labels     = labels.to(DEVICE, non_blocking=True)
            label_lens = label_lens.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits     = crnn(imgs)
                log_probs  = logits.log_softmax(2)
                input_lens = torch.full(
                    (imgs.size(0),), logits.size(0),
                    dtype=torch.long, device=DEVICE
                )
                val_loss += ctc_loss(
                    log_probs, labels, input_lens, label_lens
                ).item()
    val_loss /= len(val_loader)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS_RECOG}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}  "
          f"lr={current_lr:.2e}")

    log_csv.writerow([epoch, f"{train_loss:.6f}",
                      f"{val_loss:.6f}", f"{current_lr:.2e}"])
    log_file.flush()   # write immediately so you can tail -f the log

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(crnn.state_dict(), best_path)
        print(f"  ✓ New best model saved → {best_path}")

    # Checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            "epoch":         epoch,
            "model":         crnn.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scaler":        scaler.state_dict(),
            "best_val_loss": best_val_loss,
        }, checkpoint_path)
        print(f"  [checkpoint saved] epoch {epoch}")

log_file.close()
print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
print(f"Best model : {best_path}")
print(f"Loss log   : {log_path}")

# ── Optional: back up to S3 ────────────────────────────────────────────────
S3_BUCKET = ""   # e.g. "my-manga-ocr-bucket" — leave empty to skip
if S3_BUCKET and os.path.exists(best_path):
    import subprocess
    subprocess.run(
        f"aws s3 cp {best_path} s3://{S3_BUCKET}/models/crnn_best.pt",
        shell=True
    )
PYTHON

echo "train_recog.py written."
echo ""
echo "======================================================"
echo " Setup complete. Next steps:"
echo "======================================================"
echo ""
echo " 1. Upload your dataset:"
echo "    scp -i your-key.pem -r data/processed/ ubuntu@<ip>:~/manga_ocr/data/"
echo ""
echo " 2. Start a tmux session so training survives SSH drops:"
echo "    tmux new -s train"
echo ""
echo " 3. Train detection model:"
echo "    cd ~/manga_ocr && python train_detect.py"
echo ""
echo " 4. Train recognition model (in same or new tmux window):"
echo "    cd ~/manga_ocr && python train_recog.py"
echo ""
echo " 5. Monitor training live (in another terminal):"
echo "    tail -f ~/manga_ocr/logs/recog_loss.csv"
echo "    watch -n 5 nvidia-smi"
echo ""
echo " 6. Download models when done:"
echo "    scp -i your-key.pem ubuntu@<ip>:~/manga_ocr/models/ ./models/ -r"
echo ""
echo " REMEMBER: Stop your EC2 instance when not training!"
echo "   AWS Console → EC2 → Instances → Stop"
echo "   (you are NOT charged for stopped instances, only for running ones)"
echo "======================================================"
```