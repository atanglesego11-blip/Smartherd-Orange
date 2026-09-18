"""1D CNN-GRU classifier for collar telemetry windows.

Why this architecture:

  The convolutional front end learns short local motifs - a fever ramp, a
  hesitant gait, a burst of acceleration - without having to remember them
  across the whole window. The GRU then reads the sequence of those motifs
  and decides what the two hours as a whole mean. That division of labour
  matters here because the classes are separated by ORDER, not by level: a
  resting animal and a sick animal have the same average speed, and a
  grazing animal and a fleeing animal both show high speed. Only the
  temporal shape distinguishes them.

  A GRU rather than an LSTM because it has ~25% fewer parameters for the same
  job, and the target for this model is eventually an ESP32-class device,
  where every kilobyte of flash is a battery-life decision.

Model size at default settings: ~50k parameters, ~200 KB in float32,
~50 KB after int8 quantisation - within reach of TFLite Micro on an ESP32-S3.
"""

import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    def __init__(self, in_channels=10, n_classes=5, conv1=32, conv2=64,
                 gru_hidden=64, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, conv1, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv1),
            nn.ReLU(),
            nn.Conv1d(conv1, conv2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv2),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # 24 -> 12 timesteps
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(conv2, gru_hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        # Attention pooling over GRU outputs: a 2-hour window is usually
        # normal with a short informative stretch inside it, so taking the
        # last hidden state alone throws away the part that matters.
        self.attn = nn.Linear(gru_hidden, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, n_classes),
        )

    def forward(self, x):                          # x: (B, C, T)
        h = self.features(x)                       # (B, conv2, T/2)
        h = h.transpose(1, 2)                      # (B, T/2, conv2)
        out, _ = self.gru(h)                       # (B, T/2, H)
        w = torch.softmax(self.attn(out), dim=1)   # (B, T/2, 1)
        ctx = (out * w).sum(dim=1)                 # (B, H)
        return self.head(ctx)

    @staticmethod
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
