import os
import torch
 
# ── Paths (EC2) ───────────────────────────────────────────────────────────────
PATH          = '~\\Desktop\\manga-ocr'
BASE_DIR      = os.path.expanduser(PATH)
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