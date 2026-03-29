# src/recognize/model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """conv → batchnorm → relu, with optional max-pool."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    CNN + Bidirectional LSTM + linear classifier.

    Input : (batch, 1, 32, 128)   — grayscale crop
    Output: (T, batch, num_classes) — for CTCLoss
    """

    def __init__(self, num_classes: int, img_h: int = 32, rnn_hidden: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1,   64,  3, pool=(2, 2)),   # → (B, 64,  16, 64)
            ConvBlock(64,  128, 3, pool=(2, 2)),   # → (B, 128,  8, 32)
            ConvBlock(128, 256, 3),                # → (B, 256,  8, 32)
            ConvBlock(256, 256, 3, pool=(2, 1)),   # → (B, 256,  4, 32)
            ConvBlock(256, 512, 3),                # → (B, 512,  4, 32)
            ConvBlock(512, 512, 3, pool=(2, 1)),   # → (B, 512,  2, 32)
            ConvBlock(512, 512, 3, pool=(2, 1)),   # → (B, 512,  1, 32)
        )

        self.rnn = nn.LSTM(
            input_size  = 512,
            hidden_size = rnn_hidden,
            num_layers  = 2,
            bidirectional = True,
            batch_first = False,
            dropout     = 0.3,
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)           # (B, 512, 1, W')
        f = f.squeeze(2)          # (B, 512, W')
        f = f.permute(2, 0, 1)    # (W', B, 512)  ← sequence-first for LSTM
        out, _ = self.rnn(f)      # (W', B, hidden*2)
        return self.fc(out)       # (W', B, num_classes)
