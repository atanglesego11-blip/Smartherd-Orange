"""Generate the figures used in the SmartHerd.ai submission and pitch.

Reads only ml/artifacts/metrics.json - it never recomputes anything, so the
figures can never disagree with the reported numbers.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
INK, GRID = "#0F1C22", "#D8DEE1"
ACCENT = "#1B6E8C"
WARN = "#C8642A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

M = json.load(open(os.path.join(ART, "metrics.json")))
CLASSES = M["dataset"]["classes"]
SHORT = ["grazing", "resting", "illness", "theft", "flight"]


def confusion(tag, fname, title):
    cm = np.array(M["experiments"][tag]["confusion_matrix"], dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(SHORT)), SHORT, rotation=35, ha="right")
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10, pad=10)
    for i in range(len(SHORT)):
        for j in range(len(SHORT)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout(); fig.savefig(os.path.join(ART, fname), dpi=170)
    plt.close(fig)


def ablation():
    labels = SHORT
    full = [M["experiments"]["cnn_gru_full"]["per_class"][c]["f1-score"] for c in CLASSES]
    mov = [M["experiments"]["cnn_gru_movement_only"]["per_class"][c]["f1-score"] for c in CLASSES]
    rf = [M["experiments"]["random_forest_baseline"]["per_class"][c]["f1-score"] for c in CLASSES]
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.bar(x - w, full, w, label="CNN-GRU, movement + physiology", color=ACCENT)
    ax.bar(x, mov, w, label="CNN-GRU, movement only", color=WARN)
    ax.bar(x + w, rf, w, label="Random forest baseline", color="#9AA7AD")
    ax.set_xticks(x, labels); ax.set_ylabel("F1 (held-out animals)")
    ax.set_ylim(0.80, 1.005)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Per-class F1: what the physiological channels buy you",
                 fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ablation.png"), dpi=170)
    plt.close(fig)


def curves():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for tag, col, lab in [("cnn_gru_full", ACCENT, "movement + physiology"),
                          ("cnn_gru_movement_only", WARN, "movement only")]:
        h = M["experiments"][tag]["history"]
        ax.plot([d["epoch"] for d in h], [d["val_macro_f1"] for d in h],
                color=col, lw=1.8, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1")
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Training", fontsize=10, pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(ART, "training_curve.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    confusion("cnn_gru_full", "confusion_full.png",
              "CNN-GRU, all channels - held-out animals")
    confusion("cnn_gru_movement_only", "confusion_movement_only.png",
              "CNN-GRU, movement only - held-out animals")
    ablation()
    curves()
    print("figures written to", ART)
