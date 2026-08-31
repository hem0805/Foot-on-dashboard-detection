"""
extract_features.py  —  Run YOLOv8-Pose on every input video and save a
labelled keypoint dataset for LSTM training.

Label strategy
--------------
Negative videos  (negative_*.mp4) : every frame → 0  (ground truth — no OOP)
Positive videos  (positive_*.mp4) : per-frame heuristic → 1 if foot is on
    dashboard in that frame, else 0.

Using per-frame labels rather than the video-level label prevents the LSTM
from learning "which video is this" instead of "is the foot on the dashboard".
The README states that positive videos contain OOP posture "in some or every
frames", so a frame-level label is required.

Run from solution/:
    python extract_features.py
    python extract_features.py --input ../input --output features.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from detector import FootOnDashboardDetector, _torso_center_x, _to_bgr
from oop_model import extract_features, FEAT_DIM


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def _extract_video(
    detector:    FootOnDashboardDetector,
    video_path:  Path,
    is_positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one feature vector and one label per frame for *video_path*.

    Parameters
    ----------
    detector    : shared detector instance (reused across videos)
    video_path  : path to the .mp4 file
    is_positive : True for positive_*.mp4 videos

    Returns
    -------
    feats  : (N, FEAT_DIM) float32  — normalised keypoint features
    labels : (N,)          int8     — per-frame OOP label (0 or 1)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feats:  list[np.ndarray] = []
    labels: list[int]        = []

    with tqdm(total=n_frames, desc=f"  {video_path.name}", unit="fr",
              ncols=80, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bgr     = _to_bgr(detector._preprocess(frame))
            results = detector.model(bgr, verbose=False)

            kpts_data = results[0].keypoints
            if kpts_data is None or len(kpts_data.data) == 0:
                feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
                labels.append(0)
                pbar.update(1)
                continue

            all_kpts = kpts_data.data.cpu().numpy()
            boxes    = results[0].boxes

            # Select passenger (rightmost hip)
            best_idx, best_x = None, -1.0
            for i, kp in enumerate(all_kpts):
                if boxes is not None and float(boxes.conf[i]) < config.PERSON_CONF_THRESHOLD:
                    continue
                cx = _torso_center_x(kp, boxes, i)
                if cx is not None and cx > best_x:
                    best_x, best_idx = cx, i

            if best_idx is None:
                feats.append(np.zeros(FEAT_DIM, dtype=np.float32))
                labels.append(0)
            else:
                kpts = all_kpts[best_idx]
                feats.append(extract_features(kpts, config.KEYPOINT_CONF_THRESHOLD))

                if is_positive:
                    # Per-frame pseudo-label: the ankle is in the dashboard zone
                    # (above floor level, not raised toward the ceiling).
                    # Using the zone gate rather than the strict dashboard line
                    # (DASHBOARD_Y) gives more positive training examples while
                    # still excluding floor-level and ceiling-level ankle positions.
                    # Negative videos provide the 0-class signal, so the LSTM
                    # learns elevated-ankle-in-zone = OOP vs floor-level = normal.
                    H = float(bgr.shape[0])
                    ankle_conf = max(config.KEYPOINT_CONF_THRESHOLD * 0.5, 0.12)
                    frame_label = 0
                    for ankle_idx in (15, 16):   # L_ANKLE, R_ANKLE (COCO-17)
                        kp = kpts[ankle_idx]
                        if float(kp[2]) >= ankle_conf:
                            ay_norm = float(kp[1]) / H
                            if config.OOP_ANKLE_ZONE_TOP < ay_norm < config.ANKLE_PARTIAL_THRESHOLD:
                                frame_label = 1
                                break
                else:
                    # Negative videos: every frame is definitively normal.
                    frame_label = 0

                labels.append(frame_label)

            pbar.update(1)

    cap.release()
    detector.reset()
    return (
        np.array(feats,  dtype=np.float32),
        np.array(labels, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract pose keypoint features from input videos for LSTM training"
    )
    ap.add_argument("--input",  "-i",
                    default=str(Path(__file__).resolve().parents[1] / "input"),
                    help="Root directory of input videos")
    ap.add_argument("--output", "-o", default="features.npz",
                    help="Output .npz file path")
    ap.add_argument("--model",  "-m", default=config.DEFAULT_MODEL,
                    help="YOLOv8-pose model weights")
    args = ap.parse_args()

    input_root = Path(args.input)
    videos     = sorted(input_root.rglob("*.mp4"))
    if not videos:
        print(f"[ERROR] No .mp4 files found under: {args.input}")
        sys.exit(1)

    print(f"\nFound {len(videos)} video(s).\n")
    detector = FootOnDashboardDetector(model_path=args.model)

    all_feats:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_vids:   list[np.ndarray] = []

    for vid_id, vp in enumerate(videos):
        is_positive = "positive" in vp.stem.lower()
        tag         = "[POS]" if is_positive else "[NEG]"
        print(f"{tag}  {vp.relative_to(input_root)}")

        feats, frame_labels = _extract_video(detector, vp, is_positive)
        all_feats.append(feats)
        all_labels.append(frame_labels)
        all_vids.append(np.full(len(feats), vid_id, dtype=np.int16))

    X        = np.concatenate(all_feats)
    y        = np.concatenate(all_labels)
    video_id = np.concatenate(all_vids)

    np.savez_compressed(args.output, X=X, y=y, video_id=video_id)

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"\nDataset saved → {args.output}")
    print(f"  Total frames : {len(X)}")
    print(f"  Positive     : {pos}  ({pos/len(X):.1%})")
    print(f"  Negative     : {neg}  ({neg/len(X):.1%})")


if __name__ == "__main__":
    main()
