"""
train_classifier.py  —  Train the OOP LSTM classifier on extracted features.

Run from solution/ after extract_features.py:
    python train_classifier.py
    python train_classifier.py --features features.npz --epochs 40 --out oop_classifier.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from oop_model import OOPClassifier, WINDOW_SIZE, FEAT_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Sliding-window dataset over frame-level features.

    Each sample is a (WINDOW_SIZE, FEAT_DIM) tensor labelled by the last
    frame in the window.  Windows that straddle two videos are skipped so
    the LSTM never sees fake temporal transitions.
    """

    def __init__(
        self,
        X:        np.ndarray,   # (N, FEAT_DIM)
        y:        np.ndarray,   # (N,)
        video_id: np.ndarray,   # (N,)  which video each frame belongs to
        window:   int = WINDOW_SIZE,
    ) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[int]        = []

        for end in range(window - 1, len(X)):
            start = end - window + 1
            # Skip windows that cross a video boundary
            if video_id[start] != video_id[end]:
                continue
            self.X.append(X[start : end + 1])
            self.y.append(int(y[end]))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_by_video(
    X: np.ndarray, y: np.ndarray, video_id: np.ndarray, val_ratio: float = 0.1
) -> tuple:
    """
    Hold out the last val_ratio of unique videos as validation set.
    val_ratio=0 trains on all data (evaluation done via main.py output videos).
    """
    unique_vids = np.unique(video_id)
    n_val       = int(len(unique_vids) * val_ratio)  # 0 is valid: no validation split
    if n_val == 0:
        empty = np.array([], dtype=X.dtype)
        return X, y, video_id, empty.reshape(0, X.shape[1]), empty, empty.astype(np.int16)
    val_vids    = set(unique_vids[-n_val:])

    mask_val = np.array([vid in val_vids for vid in video_id])
    mask_tr  = ~mask_val

    return (X[mask_tr], y[mask_tr], video_id[mask_tr],
            X[mask_val], y[mask_val], video_id[mask_val])


def _eval(model, loader, criterion, device) -> tuple[float, list, list]:
    model.eval()
    total_loss, preds, truths = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb   = xb.to(device), yb.to(device)
            logits   = model(xb)
            total_loss += criterion(logits, yb).item()
            preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
            truths.extend(yb.long().cpu().tolist())
    return total_loss / max(len(loader), 1), preds, truths


def _metrics(truths, preds) -> dict:
    tp = sum(t == 1 and p == 1 for t, p in zip(truths, preds))
    fp = sum(t == 0 and p == 1 for t, p in zip(truths, preds))
    fn = sum(t == 1 and p == 0 for t, p in zip(truths, preds))
    tn = sum(t == 0 and p == 0 for t, p in zip(truths, preds))
    acc  = (tp + tn) / max(len(truths), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # --- load data ---
    data     = np.load(args.features)
    X        = data["X"].astype(np.float32)
    y        = data["y"].astype(np.float32)
    video_id = data["video_id"]

    pos = int(y.sum()); neg = len(y) - pos
    print(f"Loaded {len(X)} frames  (pos={pos}  neg={neg})\n")

    X_tr, y_tr, vid_tr, X_val, y_val, vid_val = _split_by_video(
        X, y, video_id, val_ratio=args.val_ratio
    )
    print(f"Train : {len(X_tr)} frames  |  Val : {len(X_val)} frames")

    tr_ds  = WindowDataset(X_tr,  y_tr,  vid_tr)
    val_ds = WindowDataset(X_val, y_val, vid_val)
    print(f"Train windows: {len(tr_ds)}  |  Val windows: {len(val_ds)}\n")

    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  drop_last=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=0)

    # --- model + optimiser ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}\n")
    model  = OOPClassifier().to(device)

    # Weighted loss to handle class imbalance
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    has_val     = len(val_ds) > 0
    best_f1     = 0.0
    best_epoch  = 0

    # --- loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        if has_val:
            val_loss, preds, truths = _eval(model, val_dl, criterion, device)
            sched.step(val_loss)
            m = _metrics(truths, preds)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"tr={tr_loss/max(len(tr_dl),1):.4f}  "
                      f"val={val_loss:.4f}  "
                      f"acc={m['acc']:.3f}  prec={m['precision']:.3f}  "
                      f"rec={m['recall']:.3f}  f1={m['f1']:.3f}")
            cur_f1 = m["f1"]
        else:
            # No validation set — save every epoch; report training loss only
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs}  "
                      f"tr={tr_loss/max(len(tr_dl),1):.4f}  (no validation split)")
            cur_f1 = 1.0   # always save when no val set

        if cur_f1 >= best_f1:
            best_f1, best_epoch = cur_f1, epoch
            torch.save({
                "model_state": model.state_dict(),
                "window_size": WINDOW_SIZE,
                "feat_dim":    FEAT_DIM,
            }, args.out)

    # --- final report ---
    print(f"\nBest checkpoint: epoch {best_epoch}  →  {args.out}")

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, preds, truths = _eval(model, val_dl, criterion, device)
    m = _metrics(truths, preds)

    print("\n── Validation results (best checkpoint) ──────────────────")
    print(f"  Accuracy  : {m['acc']:.3f}")
    print(f"  Precision : {m['precision']:.3f}  (of predicted OOP, how many were real)")
    print(f"  Recall    : {m['recall']:.3f}  (of real OOP frames, how many detected)")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print("───────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train OOP LSTM classifier")
    ap.add_argument("--features", default="features.npz",       help="Feature file from extract_features.py")
    ap.add_argument("--out",      default="oop_classifier.pt",  help="Output model path")
    ap.add_argument("--epochs",    type=int,   default=40)
    ap.add_argument("--batch",     type=int,   default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="Fraction of videos held out for validation (0 = train on all)")
    train(ap.parse_args())
