"""
oop_model.py  —  LSTM classifier + shared feature-extraction utility.

Shared by extract_features.py (offline), train_classifier.py (offline),
and detector.py (online inference).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Feature constants
# ---------------------------------------------------------------------------

# Keypoint indices (COCO-17) included in the feature vector.
# Ordered deliberately: shoulders, elbows, hips, knees, ankles.
FEAT_KP_INDICES = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]   # 10 keypoints
FEAT_DIM        = len(FEAT_KP_INDICES) * 3                  # x, y, conf  = 30
WINDOW_SIZE     = 16    # frames per sliding window  (~0.8 s at 20 fps)
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(kpts: np.ndarray, conf_thresh: float = 0.30) -> np.ndarray:
    """
    Build a normalised FEAT_DIM-dim feature vector from a (17, 3) keypoint array.

    Normalisation
    -------------
    origin : hip midpoint  (position-invariant)
    scale  : shoulder-to-hip distance  (size-invariant)

    Missing or low-confidence keypoints are encoded as zeros.

    Parameters
    ----------
    kpts        : (17, 3) array  [x, y, confidence]
    conf_thresh : minimum confidence to include a keypoint

    Returns
    -------
    feat : (FEAT_DIM,) float32 array
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)

    # Hip centre
    hip_pts = [kpts[i, :2] for i in (11, 12) if kpts[i, 2] >= conf_thresh]
    hip_c   = np.mean(hip_pts, axis=0) if hip_pts else np.array([0.5, 0.5], dtype=np.float32)

    # Torso scale (shoulder-to-hip distance)
    sh_pts = [kpts[i, :2] for i in (5, 6) if kpts[i, 2] >= conf_thresh]
    if sh_pts and hip_pts:
        sh_c  = np.mean(sh_pts, axis=0)
        scale = float(np.linalg.norm(sh_c - hip_c))
    else:
        scale = 1.0
    scale = max(scale, 1e-4)

    for slot, kp_idx in enumerate(FEAT_KP_INDICES):
        kp = kpts[kp_idx]
        if kp[2] >= conf_thresh:
            feat[slot * 3    ] = (kp[0] - hip_c[0]) / scale
            feat[slot * 3 + 1] = (kp[1] - hip_c[1]) / scale
            feat[slot * 3 + 2] = float(kp[2])

    return feat


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OOPClassifier(nn.Module):
    """
    Bidirectional LSTM binary classifier.

    Input  : (batch, WINDOW_SIZE, FEAT_DIM)  — normalised keypoint windows
    Output : (batch,)  raw logit  (positive = OOP)
    """

    def __init__(
        self,
        feat_dim:    int = FEAT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = feat_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3 if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        # Concatenate last forward and backward hidden states
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)   # (batch, hidden*2)
        return self.head(h_cat).squeeze(-1)          # (batch,)
