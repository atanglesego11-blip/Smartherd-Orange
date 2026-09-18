"""Train and evaluate the SmartHerd.ai 1D CNN-GRU, with an ablation and a baseline.

Runs three experiments and writes every number it computes to
ml/artifacts/metrics.json. Nothing in that file is typed by hand.

  A. CNN-GRU, all 10 channels (movement + physiology + time)
  B. CNN-GRU, movement channels only - the ablation that tests whether the
     physiological channels are actually carrying the illness signal
  C. Random Forest on hand-crafted window statistics - a classical baseline,
     reported whether or not it flatters the deep model

Usage:
    python ml/train.py --data data/telemetry.npz --epochs 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, balanced_accuracy_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (load, animal_split, fit_scaler, apply_scaler,
                     class_weights, MOVEMENT_ONLY_CHANNELS, CLASSES)
from model import CNNGRU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_torch(Xtr, ytr, Xva, yva, Xte, yte, epochs, lr, seed, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CNNGRU(in_channels=Xtr.shape[1], n_classes=len(CLASSES)).to(DEVICE)
    w = torch.tensor(class_weights(ytr)).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    loader = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True, drop_last=False)
    Xva_t = torch.tensor(Xva).to(DEVICE)
    Xte_t = torch.tensor(Xte).to(DEVICE)

    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).cpu().numpy()
        f1 = f1_score(yva, pv, average="macro", zero_division=0)
        history.append({"epoch": ep + 1, "train_loss": tot / len(ytr), "val_macro_f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep+1:02d}/{epochs}  loss {tot/len(ytr):.4f}  val_macroF1 {f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1)

    rep = classification_report(yte, pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    result = {
        "tag": tag,
        "params": CNNGRU.count_params(model),
        "train_seconds": round(time.time() - t0, 1),
        "best_val_macro_f1": float(best_f1),
        "test_macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "test_accuracy": float((pred == yte).mean()),
        "per_class": {c: {k: float(v) for k, v in rep[c].items()} for c in CLASSES},
        "confusion_matrix": confusion_matrix(yte, pred, labels=range(len(CLASSES))).tolist(),
        "history": history,
    }
    return model, result, probs, pred


def window_stats(X):
    """Hand-crafted summary features for the classical baseline."""
    return np.concatenate([X.mean(2), X.std(2), X.min(2), X.max(2),
                           X[:, :, -1] - X[:, :, 0]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telemetry.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ml/artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y, animal, channels, classes = load(args.data)
    m_tr, m_va, m_te, split_ids = animal_split(animal, seed=args.seed)
    print(f"windows: train {m_tr.sum()}  val {m_va.sum()}  test {m_te.sum()}")

    mu, sd = fit_scaler(X[m_tr])
    Xn = apply_scaler(X, mu, sd)

    metrics = {
        "dataset": {
            "path": args.data, "synthetic": True,
            "windows": int(len(y)), "channels": channels, "classes": classes,
            "train_windows": int(m_tr.sum()), "val_windows": int(m_va.sum()),
            "test_windows": int(m_te.sum()),
            "split": "by animal (no window leakage)",
            "animal_split": split_ids,
            "test_class_counts": {classes[i]: int((y[m_te] == i).sum())
                                  for i in range(len(classes))},
        },
        "experiments": {},
    }

    # --- A. full channels -------------------------------------------------
    model, res_full, probs, pred = run_torch(
        Xn[m_tr], y[m_tr], Xn[m_va], y[m_va], Xn[m_te], y[m_te],
        args.epochs, args.lr, args.seed, "cnn_gru_full")
    metrics["experiments"]["cnn_gru_full"] = res_full
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "channels": channels, "classes": classes,
                "in_channels": Xn.shape[1]},
               os.path.join(args.out, "smartherd_cnn_gru.pt"))
    np.save(os.path.join(args.out, "test_probs.npy"), probs)
    np.save(os.path.join(args.out, "test_true.npy"), y[m_te])

    # --- B. movement-only ablation ---------------------------------------
    Xm = Xn[:, MOVEMENT_ONLY_CHANNELS, :]
    _, res_mov, _, _ = run_torch(
        Xm[m_tr], y[m_tr], Xm[m_va], y[m_va], Xm[m_te], y[m_te],
        args.epochs, args.lr, args.seed, "cnn_gru_movement_only")
    metrics["experiments"]["cnn_gru_movement_only"] = res_mov

    # --- C. classical baseline -------------------------------------------
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=args.seed, n_jobs=-1)
    rf.fit(window_stats(Xn[m_tr]), y[m_tr])
    rp = rf.predict(window_stats(Xn[m_te]))
    rep = classification_report(y[m_te], rp, target_names=classes,
                                output_dict=True, zero_division=0)
    metrics["experiments"]["random_forest_baseline"] = {
        "tag": "random_forest_baseline",
        "train_seconds": round(time.time() - t0, 1),
        "test_macro_f1": float(f1_score(y[m_te], rp, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y[m_te], rp)),
        "test_accuracy": float((rp == y[m_te]).mean()),
        "per_class": {c: {k: float(v) for k, v in rep[c].items()} for c in classes},
        "confusion_matrix": confusion_matrix(y[m_te], rp, labels=range(len(classes))).tolist(),
    }

    metrics["headline"] = {
        "cnn_gru_full_macro_f1": res_full["test_macro_f1"],
        "cnn_gru_movement_only_macro_f1": res_mov["test_macro_f1"],
        "random_forest_macro_f1": metrics["experiments"]["random_forest_baseline"]["test_macro_f1"],
        "illness_recall_full": res_full["per_class"]["illness_suspect"]["recall"],
        "illness_recall_movement_only": res_mov["per_class"]["illness_suspect"]["recall"],
        "theft_recall_full": res_full["per_class"]["theft_suspect"]["recall"],
        "model_parameters": res_full["params"],
    }
    metrics["caveat"] = (
        "All figures are computed on held-out SYNTHETIC animals from "
        "simulator/simulate_collar.py (seed 42). They measure whether the "
        "architecture can learn the encoded signature, not field accuracy. "
        "Re-run this exact script on real collar data before quoting any "
        "number as a field result."
    )

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n" + json.dumps(metrics["headline"], indent=2))
    print("\nwrote", os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
