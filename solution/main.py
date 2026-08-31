"""
main.py — Entry point for the Euro NCAP OOP Feet-on-Dashboard Detector.

Usage
-----
# Process all videos (default paths mirror the assignment structure)
python main.py

# Custom paths
python main.py --input path/to/input --output path/to/output

# Save per-frame statistics to JSON
python main.py --stats results/stats.json

# Use a larger (more accurate) model
python main.py --model yolov8m-pose.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from video_processor import process_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oop_detector",
        description="Euro NCAP OOP Case 1 — Feet-on-Dashboard Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=str(Path(__file__).resolve().parents[1] / "input"),
        help="Root directory containing profile sub-folders with input videos.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="Root directory where annotated output videos will be written.",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n-pose.pt",
        help=(
            "YOLOv8-pose model to use. "
            "Options (speed ↔ accuracy trade-off): "
            "yolov8n-pose.pt | yolov8s-pose.pt | yolov8m-pose.pt | yolov8l-pose.pt"
        ),
    )
    parser.add_argument(
        "--stats", "-s",
        default=None,
        metavar="PATH",
        help="Optional path to write per-video statistics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  Euro NCAP OOP — Feet-on-Dashboard Detector")
    print("=" * 62)
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Model  : {args.model}")
    if args.stats:
        print(f"  Stats  : {args.stats}")
    print("=" * 62)

    # Validate input directory
    if not Path(args.input).is_dir():
        print(f"\n[ERROR] Input directory not found: {args.input}")
        sys.exit(1)

    t0 = time.perf_counter()

    all_stats = process_all(
        input_dir  = args.input,
        output_dir = args.output,
        model_path = args.model,
        stats_path = args.stats,
    )

    elapsed = time.perf_counter() - t0

    # Summary
    total_videos  = len(all_stats)
    total_frames  = sum(s.get("total", 0) for s in all_stats.values())
    total_positive = sum(s.get("positive", 0) for s in all_stats.values())

    print("\n" + "=" * 62)
    print("  Done!")
    print(f"  Videos processed : {total_videos}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Positive frames  : {total_positive} "
          f"({total_positive / max(total_frames, 1):.1%})")
    print(f"  Wall-clock time  : {elapsed:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()