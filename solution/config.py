"""
config.py — Camera intrinsics and detection hyperparameters.

Camera hardware (from datasheet):
  Model       : Intensirics IR camera (STMicro VG1762, GEO GW5410 ISP)
  Resolution  : 1920 × 1080 @ 20 fps (IR mode)
  Lens FOV    : 170° (D) × 170° (H) × 106.3° (V)  — ultra-wide fisheye
  IR LED      : OSRAM SFH4726AS A01, 940 nm peak wavelength
  Output      : YUV422 8-bit

All detection thresholds are normalised to [0, 1] relative to frame dimensions
so the pipeline is resolution-agnostic.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Camera intrinsic parameters (Intensirics IR camera, Hyundai Santa Fe)
# ---------------------------------------------------------------------------
CAMERA_MATRIX = np.array(
    [
        [595.8036047830891,  0.0,                924.65430795264774],
        [0.0,               598.33827405037948,  580.41919770485049],
        [0.0,               0.0,                 1.0],
    ],
    dtype=np.float64,
)

# Distortion coefficients as provided in the assignment: [k1, k2, p1, p2]
# NOTE: The lens has a 170° FOV which is fisheye geometry. These coefficients
# were provided in standard pinhole format. Two undistortion strategies are
# available and selectable via USE_FISHEYE_MODEL below:
#
#   False → cv2.undistort  (standard pinhole + Brown–Conrady model)
#   True  → cv2.fisheye.*  (equidistant fisheye model; coefficients reused
#                           as (k1, k2, k3, k4) — treat k3,k4 as residuals)
#
# Start with False; switch to True if the undistorted image looks warped.
USE_FISHEYE_MODEL = False

# Set to True only if the input video is horizontally mirrored (driver appears
# on the RIGHT of the raw frame). The provided videos are already in the correct
# orientation, so this is False by default.
FLIP_HORIZONTAL = False

# Apply lens-distortion correction before inference.
# With the provided coefficients the remap produces heavy black borders on the
# 170° fisheye frame and offers little benefit for cabin pose estimation.
# Leave False so the original clean frame is used for both inference and output.
APPLY_UNDISTORTION = False

DIST_COEFFS = np.array(
    [
        -0.015349419086740696,
        -0.0536764772521049,
         0.061315407683887907,
        -0.026142516909791854,
    ],
    dtype=np.float64,
)

# Reshaped (4,1) array required by cv2.fisheye functions
DIST_COEFFS_FISHEYE = DIST_COEFFS.reshape(4, 1)

# ---------------------------------------------------------------------------
# YOLOv8-Pose model
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "yolov8n-pose.pt"   # auto-downloaded on first run (~6 MB)
# Upgrade for accuracy (slower): yolov8s-pose.pt / yolov8m-pose.pt / yolov8l-pose.pt

# ---------------------------------------------------------------------------
# Detection confidence thresholds
# ---------------------------------------------------------------------------
PERSON_CONF_THRESHOLD   = 0.30   # minimum YOLO box confidence
KEYPOINT_CONF_THRESHOLD = 0.30   # minimum keypoint confidence

# ---------------------------------------------------------------------------
# Steering-wheel axis — OOP threshold
# ---------------------------------------------------------------------------
# The steering wheel is always visible in the lower-left of the raw (non-
# undistorted) frame.  Its vertical centre in normalised image coordinates
# acts as a horizontal threshold line.
#
# OOP condition:  ankle_y_norm < DASHBOARD_Y
#   (ankle has risen ABOVE this line = foot is at or above dashboard level)
#
# Calibrated on the raw fisheye frame (APPLY_UNDISTORTION = False):
#   steering-wheel centre ≈ 0.88 × H  (very low in the wide-angle view).
#   Dashboard surface where the foot rests is slightly above the wheel,
#   so the effective threshold is ~0.87.  A raised foot appears at ~0.85.
#
# NOTE: DASHBOARD_Y is now used only as a VISUAL REFERENCE LINE on the output
# video.  It no longer drives OOP detection directly.  Detection is based on
# the full leg posture (ankle gate + body confirmation checks below).
DASHBOARD_Y = 0.87

# Draw the threshold line across the full frame width for visual tuning.
# Set to False once the line is positioned correctly.
SHOW_STEERING_AXIS = True

# ---------------------------------------------------------------------------
# Temporal smoothing  (at 20 fps, 9 frames ≈ 450 ms)
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW = 5   # history window for per-keypoint smoothing (deque maxlen)

# OOP state-machine hysteresis
# Trigger: need OOP_TRIGGER_COUNT positive detections within OOP_TRIGGER_WINDOW frames
OOP_TRIGGER_WINDOW = 5    # look-back window (frames)
OOP_TRIGGER_COUNT  = 1    # enter OOP on the first confirmed positive frame
                           # the ankle zone gate is the noise filter, not this count
# Release: need OOP_RELEASE_FRAMES *consecutive* negatives to exit OOP state
# At 20 fps: 20 frames = 1 s of sustained normal posture before switching back
OOP_RELEASE_FRAMES = 20

# ---------------------------------------------------------------------------
# Heuristic fallback thresholds — used ONLY when the HAR model is not loaded.
# ---------------------------------------------------------------------------
# These two values are derived from the physical camera geometry, not from
# observing individual videos.  They are the ONLY thresholds the heuristic uses.
#
# OOP_ANKLE_ZONE_TOP  : physical lower bound on ankle y/H for a foot resting on
#   the dashboard.  The dashboard appears at y ≈ 0.87 in this fisheye frame.
#   A raised-folded knee sends the ankle toward the ceiling (y < 0.78), well
#   above the dashboard.  Derived from the calibrated camera position — not tuned.
#
# ANKLE_PARTIAL_THRESHOLD : floor level in this frame.  The floor appears at
#   y ≈ 0.97; anything above 0.92 is clearly not at floor level.  Derived from
#   camera geometry — not tuned.
#
# The heuristic rule is intentionally simple:
#   ankle in zone (OOP_ANKLE_ZONE_TOP < y/H < ANKLE_PARTIAL_THRESHOLD)
#   AND ankle at or above calibrated dashboard line (y/H < DASHBOARD_Y)
#   → OOP
#
# All nuanced cases (bent knee, partial visibility, any angle) are handled by
# the trained HAR LSTM model.  Do NOT add more thresholds here.
OOP_ANKLE_ZONE_TOP      = 0.78   # camera-geometry bound: raised-folded knee is above this
ANKLE_PARTIAL_THRESHOLD = 0.92   # camera-geometry bound: floor level is below this

# ---------------------------------------------------------------------------
# HAR (Human Activity Recognition) LSTM classifier
# ---------------------------------------------------------------------------
# Path is relative to the solution/ directory; the model is auto-loaded when
# the file exists after running train_classifier.py.  Falls back to the
# heuristic (_is_oop) when the file is absent or fails to load.
HAR_CLASSIFIER = "oop_classifier.pt"

# ---------------------------------------------------------------------------
# Output video rendering
# ---------------------------------------------------------------------------
POSITIVE_COLOR   = (0,  50, 220)    # BGR – red
NEGATIVE_COLOR   = (30, 180,  30)   # BGR – green
UNKNOWN_COLOR    = (130, 130, 130)  # BGR – grey (no detection)

FONT             = 1                # cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 1.4
LABEL_THICKNESS  = 2
BANNER_ALPHA     = 0.45            # transparency of banner overlay
BANNER_HEIGHT_PX = 56

# Human-readable labels shown in the output video banner
DISPLAY_LABELS = {
    "POSITIVE": "FOOT ON DASHBOARD",
    "NEGATIVE": "NORMAL POSTURE",
    "UNKNOWN":  "NO DETECTION",
}

# Set to True to produce a clean output video: only the colour banner and
# label text are drawn — no skeleton, no joint dots, no threshold lines.
CLEAN_OUTPUT = True

# ---------------------------------------------------------------------------
# Scrolling timeline graph (appended below each frame)
# ---------------------------------------------------------------------------
# Draws a coloured bar strip below the video showing POSITIVE/NEGATIVE history
# across the most recent GRAPH_WINDOW_FRAMES frames.
SHOW_TIMELINE_GRAPH  = True
GRAPH_HEIGHT_PX      = 72     # pixel height of the graph strip
GRAPH_WINDOW_FRAMES  = 300    # number of frames visible in the scrolling window