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
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
 
from config import (RECOG_JSON, RECOG_MODEL_DIR, LOG_DIR,
                    BATCH_SIZE, EPOCHS_RECOG, LR,
                    NUM_CLASSES, DEVICE, RESUME_RECOG, PATH as BASE_DIR_CONFIG)
                    
sys.path.insert(0, os.path.expanduser(PATH)                    
 
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