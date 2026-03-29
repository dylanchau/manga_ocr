# src/recognize/train.py
"""
Internal training module called by the top-level train_recog.py.
Can also be run standalone:
    cd ~/manga_ocr
    python -m src.recognize.train
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_

from config import (
    DEVICE, BATCH_SIZE, EPOCHS_RECOG, LR,
    NUM_CLASSES, RECOG_MODEL_DIR, RECOG_JSON,
)
from src.dataset import MangaRecognitionDataset, collate_fn
from src.recognize.model import CRNN


def train_recognition(json_path: str = RECOG_JSON):
    # ── Dataset ───────────────────────────────────────────────────────────
    dataset  = MangaRecognitionDataset(json_path)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size]
    )

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn
    )
    val_loader   = DataLoader(
        val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model     = CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5
    )
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(RECOG_MODEL_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS_RECOG + 1):
        # Train
        model.train()
        train_loss = 0.0

        for imgs, labels, label_lens in train_loader:
            imgs       = imgs.to(DEVICE)
            labels     = labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            logits    = model(imgs)
            log_probs = logits.log_softmax(2)
            input_lens = torch.full(
                (imgs.size(0),), logits.size(0), dtype=torch.long
            )
            loss = ctc_loss(log_probs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels, label_lens in val_loader:
                imgs       = imgs.to(DEVICE)
                labels     = labels.to(DEVICE)
                label_lens = label_lens.to(DEVICE)
                logits     = model(imgs)
                log_probs  = logits.log_softmax(2)
                input_lens = torch.full(
                    (imgs.size(0),), logits.size(0), dtype=torch.long
                )
                val_loss += ctc_loss(
                    log_probs, labels, input_lens, label_lens
                ).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS_RECOG}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_recognition()
