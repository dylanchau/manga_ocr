"""
train_recog.py  — EC2 version with S3 backup + safe Spot resume
─────────────────────────────────────────────────────────────────
Changes from original:
  1. TOTAL_EPOCHS derived from config so extending epochs works correctly
  2. Checkpoint saved every 5 epochs (safer for Spot, was 10)
  3. S3 backup after every checkpoint AND after best model is saved
  4. On startup: pulls latest checkpoint from S3 if local one is missing
     (handles the case where the instance was terminated, not just stopped)
"""

import os
import sys
import csv
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from config import (
    RECOG_JSON, RECOG_MODEL_DIR, LOG_DIR,
    BATCH_SIZE, EPOCHS_RECOG, LR,
    NUM_CLASSES, DEVICE, RESUME_RECOG,
    PATH as BASE_DIR_CONFIG,
)
from src.dataset import MangaRecognitionDataset, collate_recognition
from src.recognize.model import CRNN

BASE_DIR = os.path.expanduser(BASE_DIR_CONFIG)

# ── S3 config — set your bucket name here ─────────────────────────────────────
# Create the bucket first:
#   aws s3 mb s3://your-manga-ocr-bucket --region us-east-1
# Then set the name below. Leave empty "" to disable S3 backup.
S3_BUCKET = ""          # e.g. "my-manga-ocr-models"
S3_PREFIX = "recog/"    # folder inside the bucket


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_upload(local_path: str, s3_key: str):
    """Upload a file to S3. Silently skips if S3_BUCKET is not set."""
    if not S3_BUCKET:
        return
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"  [S3] uploaded → s3://{S3_BUCKET}/{s3_key}")
    except NoCredentialsError:
        print("  [S3] No credentials found — check IAM role is attached to instance")
    except ClientError as e:
        print(f"  [S3] Upload failed: {e}")


def s3_download(s3_key: str, local_path: str) -> bool:
    """
    Download a file from S3 to local_path.
    Returns True if successful, False if the key doesn't exist or S3 is disabled.
    """
    if not S3_BUCKET:
        return False
    try:
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, s3_key, local_path)
        print(f"  [S3] downloaded ← s3://{S3_BUCKET}/{s3_key}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False   # file doesn't exist yet — that's fine on first run
        print(f"  [S3] Download failed: {e}")
        return False
    except NoCredentialsError:
        print("  [S3] No credentials — skipping S3 restore")
        return False

def main():
    # ── Validate dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(RECOG_JSON):
        print(f"ERROR: labels.json not found at {RECOG_JSON}")
        print("Run tools/convert_annotations.py first.")
        sys.exit(1)

    # ── Dataset ────────────────────────────────────────────────────────────────────
    dataset  = MangaRecognitionDataset(RECOG_JSON)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    print(f"Dataset  : {len(dataset)} crops  (train: {len(train_ds)}, val: {val_size})")
    print(f"Device   : {DEVICE}")
    print(f"S3 backup: {'enabled → ' + S3_BUCKET if S3_BUCKET else 'disabled'}")
    print()

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        collate_fn=collate_recognition,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False,
        collate_fn=collate_recognition,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model, optimizer, loss ────────────────────────────────────────────────────
    crnn      = CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp   = (DEVICE == "cuda")
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Checkpoint paths ──────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(RECOG_MODEL_DIR, "checkpoint_latest.pt")
    best_path       = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")
    S3_CHECKPOINT   = S3_PREFIX + "checkpoint_latest.pt"
    S3_BEST         = S3_PREFIX + "crnn_best.pt"

    # ── Resume logic ──────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if RESUME_RECOG:
        # Step 1: if local checkpoint is missing, try to restore from S3
        # This handles the "Spot was terminated and disk was wiped" scenario
        if not os.path.exists(checkpoint_path):
            print("Local checkpoint not found — checking S3...")
            s3_download(S3_CHECKPOINT, checkpoint_path)

        # Step 2: load checkpoint if it now exists (locally or restored from S3)
        if os.path.exists(checkpoint_path):
            ckpt          = torch.load(checkpoint_path, map_location=DEVICE)
            crnn.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            print(f"Resumed from epoch {ckpt['epoch']}  "
                  f"(best val loss: {best_val_loss:.4f})")
        else:
            print("No checkpoint found locally or in S3 — starting from scratch.")

    # ── TOTAL_EPOCHS: always run at least EPOCHS_RECOG epochs total ───────────────
    # If you want to extend training, increase EPOCHS_RECOG in config.py
    # and set RESUME_RECOG = True — the loop will continue from start_epoch.
    TOTAL_EPOCHS = max(EPOCHS_RECOG, start_epoch - 1)
    if start_epoch > TOTAL_EPOCHS:
        print(f"Already completed {start_epoch - 1} epochs "
              f"(EPOCHS_RECOG={EPOCHS_RECOG}). "
              f"Increase EPOCHS_RECOG in config.py to train more.")
        sys.exit(0)

    print(f"Training epochs {start_epoch} → {TOTAL_EPOCHS}")
    print()

    # ── CSV loss log ──────────────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "recog_loss.csv")
    log_file = open(log_path, "a", newline="")
    log_csv  = csv.writer(log_file)
    if start_epoch == 1:
        log_csv.writerow(["epoch", "train_loss", "val_loss", "lr"])

    # ── Training loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        # Train
        crnn.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:3d}/{TOTAL_EPOCHS} [train]",
                    leave=False, ncols=80)

        for imgs, labels, label_lens in pbar:
            imgs       = imgs.to(DEVICE, non_blocking=True)
            labels     = labels.to(DEVICE, non_blocking=True)
            label_lens = label_lens.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits     = crnn(imgs)
                log_probs  = logits.log_softmax(2)
                input_lens = torch.full(
                    (imgs.size(0),), logits.size(0),
                    dtype=torch.long, device=DEVICE,
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
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits     = crnn(imgs)
                    log_probs  = logits.log_softmax(2)
                    input_lens = torch.full(
                        (imgs.size(0),), logits.size(0),
                        dtype=torch.long, device=DEVICE,
                    )
                    val_loss += ctc_loss(
                        log_probs, labels, input_lens, label_lens
                    ).item()
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{TOTAL_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={current_lr:.2e}")

        log_csv.writerow([epoch, f"{train_loss:.6f}",
                          f"{val_loss:.6f}", f"{current_lr:.2e}"])
        log_file.flush()

        # ── Save best model ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(crnn.state_dict(), best_path)
            print(f"  ✓ New best model saved  (val_loss={val_loss:.4f})")
            # Back up best model to S3 immediately
            s3_upload(best_path, S3_BEST)

        # ── Checkpoint every 5 epochs (reduced from 10 for Spot safety) ────────
        if epoch % 5 == 0:
            torch.save({
                "epoch":         epoch,
                "model":         crnn.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scaler":        scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"  [checkpoint] epoch {epoch} saved locally")
            # Back up checkpoint to S3 so a new instance can restore it
            s3_upload(checkpoint_path, S3_CHECKPOINT)

    log_file.close()

    print(f"\nTraining complete.")
#print(f"Best val loss : {best_val_loss:.4f}")
#print(f"Best model    : {best_path}")
#print(f"Loss log      : {log_path}")
#if S3_BUCKET:
#    print(f"S3 backup     : s3://{S3_BUCKET}/{S3_PREFIX}")
if __name__ == '__main__':
    main()
