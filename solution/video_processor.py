"""
video_processor.py — Video I/O and batch processing utilities.

Walks the input directory tree, processes each .mp4, and writes the
annotated result to the mirrored output directory tree.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

from detector import FootOnDashboardDetector
import config


# ---------------------------------------------------------------------------
# Timeline graph helper
# ---------------------------------------------------------------------------

def _draw_timeline(frame: np.ndarray, history: list) -> np.ndarray:
    """
    Append a scrolling timeline strip below *frame*.

    Each frame in the visible window is rendered as a thin vertical bar:
      green  → NEGATIVE (normal posture)
      red    → POSITIVE (foot on dashboard)
      grey   → UNKNOWN  (no detection)

    A white cursor marks the current (latest) frame.
    """
    h, w      = frame.shape[:2]
    gh        = config.GRAPH_HEIGHT_PX
    window    = config.GRAPH_WINDOW_FRAMES
    visible   = history[-window:]
    n         = len(visible)

    strip = np.full((gh, w, 3), (30, 30, 30), dtype=np.uint8)

    if n > 0:
        bar_w = max(1, w // window)

        for i, label in enumerate(visible):
            x1 = i * bar_w
            x2 = min(x1 + bar_w, w - 1)
            if label == "POSITIVE":
                color = config.POSITIVE_COLOR
            elif label == "NEGATIVE":
                color = config.NEGATIVE_COLOR
            else:
                color = config.UNKNOWN_COLOR
            cv2.rectangle(strip, (x1, 6), (x2, gh - 6), color, -1)

        # White cursor at the current frame
        cx = min((n - 1) * bar_w + bar_w // 2, w - 1)
        cv2.line(strip, (cx, 0), (cx, gh), (255, 255, 255), 2)

    # Divider line between video and graph
    cv2.line(strip, (0, 0), (w, 0), (80, 80, 80), 1)

    # Label
    cv2.putText(strip, "Timeline", (6, gh - 8),
                config.FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    # Current frame count
    cv2.putText(strip, f"Frame {len(history)}", (w - 140, gh - 8),
                config.FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([frame, strip])


# ---------------------------------------------------------------------------
# Single-video processing
# ---------------------------------------------------------------------------

def process_video(
    detector:    FootOnDashboardDetector,
    input_path:  str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Read *input_path*, classify every frame, write annotated *output_path*.

    Returns a statistics dict:
        total          – total frame count
        positive       – number of POSITIVE-labelled frames
        negative       – number of NEGATIVE-labelled frames
        unknown        – frames where no person was detected
        positive_ratio – positive / total
        per_frame      – list of per-frame smoothed labels
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    raw_fps  = cap.get(cv2.CAP_PROP_FPS)
    if not raw_fps or raw_fps <= 0:
        print(f"  [WARNING] Could not read FPS from '{Path(input_path).name}', defaulting to 25.0")
        raw_fps = 25.0
    fps = raw_fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise IOError(
            f"Invalid frame dimensions ({w}x{h}) reported for: {input_path}"
        )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h = h + config.GRAPH_HEIGHT_PX if config.SHOW_TIMELINE_GRAPH else h

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, out_h))
    if not writer.isOpened():
        # mp4v unavailable — fall back to avc1 (H.264) then XVID
        for codec in ("avc1", "XVID"):
            writer.release()
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, out_h))
            if writer.isOpened():
                break
        if not writer.isOpened():
            cap.release()
            raise IOError(f"No suitable video codec found to write: {output_path}")

    # Reset per-video state (temporal smoothing history, undistort cache)
    detector.reset()

    stats: Dict[str, Any] = {
        "total":    0,
        "positive": 0,
        "negative": 0,
        "unknown":  0,
        "per_frame": [],
    }

    with tqdm(
        total=n_frames,
        desc=f"  {Path(input_path).name}",
        unit="fr",
        ncols=80,
        leave=False,
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_label, smoothed_label, annotated = detector.classify_frame(frame, stats["total"])

            if config.SHOW_TIMELINE_GRAPH:
                annotated = _draw_timeline(annotated, stats["per_frame"] + [smoothed_label])

            writer.write(annotated)

            stats["total"] += 1
            stats["per_frame"].append(smoothed_label)
            if smoothed_label == "POSITIVE":
                stats["positive"] += 1
            elif smoothed_label == "NEGATIVE":
                stats["negative"] += 1
            else:
                stats["unknown"] += 1

            pbar.update(1)

    cap.release()
    writer.release()

    total = max(stats["total"], 1)
    stats["positive_ratio"] = round(stats["positive"] / total, 4)
    return stats


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all(
    input_dir:  str,
    output_dir: str,
    model_path: str = config.DEFAULT_MODEL,
    stats_path: str | None = None,
) -> Dict[str, Any]:
    """
    Process every .mp4 found under *input_dir*, preserving the directory
    structure under *output_dir*.

    Expected input structure
    ├── profile1/
    │   ├── positive_1.mp4
    │   └── negative_1.mp4
    └── profile2/
        └── ...

    Parameters
    ----------
    input_dir  : root of the input video tree
    output_dir : root of the output video tree
    model_path : YOLOv8-pose weights file or model name
    stats_path : optional path to write JSON statistics

    Returns
    -------
    all_stats  : {relative_video_path: stats_dict, ...}
    """
    detector   = FootOnDashboardDetector(model_path=model_path)
    input_root  = Path(input_dir)
    output_root = Path(output_dir)
    all_stats: Dict[str, Any] = {}

    # Gather all .mp4 files
    video_files = sorted(input_root.rglob("*.mp4"))
    if not video_files:
        print(f"[WARNING] No .mp4 files found under: {input_dir}")
        return all_stats

    print(f"\nFound {len(video_files)} video(s) to process.\n")

    for video_path in video_files:
        rel        = video_path.relative_to(input_root)
        out_path   = output_root / rel
        rel_str    = str(rel)

        print(f"[{rel_str}]")
        try:
            stats = process_video(detector, str(video_path), str(out_path))
            all_stats[rel_str] = stats
            print(
                f"  ✓ {stats['positive']}/{stats['total']} frames POSITIVE "
                f"({stats['positive_ratio']:.1%})  →  {out_path}"
            )
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            all_stats[rel_str] = {"error": str(exc)}

    if stats_path:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as fh:
            json.dump(all_stats, fh, indent=2)
        print(f"\nStatistics written to: {stats_path}")

    return all_stats