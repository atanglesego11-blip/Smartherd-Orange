"""Dataset loading, splitting and normalisation for SmartHerd.ai.

Two rules are enforced here because breaking either of them silently inflates
every number downstream:

  1. Splits are by ANIMAL, never by window. Consecutive windows overlap in
     time, so a random window split would put near-duplicate windows in both
     train and test and report an accuracy that cannot be reproduced in the
     field.
  2. Normalisation statistics are fitted on the training animals only and
     then applied to val and test.
"""

import numpy as np

CLASSES = ["grazing", "resting", "illness_suspect", "theft_suspect", "predator_flight"]
MOVEMENT_ONLY_CHANNELS = [0, 1, 2, 3, 4, 8, 9]   # drops temp_c, hr_bpm, activity_idx


def load(path="data/telemetry.npz"):
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["animal"].astype(np.int64), \
        [str(c) for c in d["channels"]], [str(c) for c in d["classes"]]


def animal_split(animal, seed=42, train_frac=0.70, val_frac=0.10):
    rng = np.random.default_rng(seed)
    ids = np.unique(animal)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(train_frac * n))
    n_va = max(1, int(round(val_frac * n)))
    tr_ids, va_ids, te_ids = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
    return (np.isin(animal, tr_ids),
            np.isin(animal, va_ids),
            np.isin(animal, te_ids),
            {"train_animals": tr_ids.tolist(), "val_animals": va_ids.tolist(),
             "test_animals": te_ids.tolist()})


def fit_scaler(X_train):
    """Per-channel mean/std over train windows. X: (N, C, T)."""
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def class_weights(y, n_classes=len(CLASSES)):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)
