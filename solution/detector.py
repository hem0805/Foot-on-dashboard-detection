"""
detector.py — FootOnDashboardDetector

Wraps YOLOv8-Pose to produce per-frame OOP (Out-of-Position) decisions.

Camera context
──────────────
The Intensirics camera is a 170° ultra-wide fisheye IR module mounted at
the rear-view mirror position facing rearward into the Hyundai Santa Fe cabin.
The recorded video is *horizontally mirrored* relative to the true cabin
orientation (driver appears on the right of the raw frame).

Pipeline per frame
──────────────────
1.  Undistort  — standard pinhole (default) or fisheye model (USE_FISHEYE_MODEL)
2.  Horizontal flip  — un-mirror: driver → LEFT, passenger → RIGHT  (LHD vehicle)
3.  Grayscale → 3-channel BGR  — IR single-channel → model-compatible format
4.  YOLOv8-Pose inference  — COCO-17 skeleton keypoints
5.  Passenger selection  — person whose hip centre is furthest right in frame
6.  OOP heuristic  — ankle elevation relative to hip + absolute height check
7.  Temporal smoothing  — sliding-window majority vote
8.  Render  — colour banner + skeleton overlay
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ultralytics import YOLO

import config
from oop_model import OOPClassifier, extract_features, WINDOW_SIZE, FEAT_DIM


# ---------------------------------------------------------------------------
# COCO-17 keypoint indices
# ---------------------------------------------------------------------------
class KP:
    NOSE         = 0
    L_EAR        = 3
    R_EAR        = 4
    L_SHOULDER   = 5
    R_SHOULDER   = 6
    L_ELBOW      = 7
    R_ELBOW      = 8
    L_HIP        = 11
    R_HIP        = 12
    L_KNEE       = 13
    R_KNEE       = 14
    L_ANKLE      = 15
    R_ANKLE      = 16


# Skeleton connection pairs for visualisation (lower-body focus)
_SKELETON_PAIRS = [
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_HIP),
    (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP,      KP.R_HIP),
    (KP.L_HIP,      KP.L_KNEE),
    (KP.L_KNEE,     KP.L_ANKLE),
    (KP.R_HIP,      KP.R_KNEE),
    (KP.R_KNEE,     KP.R_ANKLE),
]


class FootOnDashboardDetector:
    """
    Detects Euro NCAP OOP Case 1 — foot/feet on dashboard — for the
    front passenger in IR in-cabin video.
    """

    def __init__(
        self,
        model_path: str = config.DEFAULT_MODEL,
        smoothing_window: int = config.SMOOTHING_WINDOW,
    ) -> None:
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_path}'. "
                "Ensure the model file exists or that you have an internet connection "
                "for auto-download on first run."
            ) from exc
        self.camera_matrix = config.CAMERA_MATRIX
        self.dist_coeffs   = config.DIST_COEFFS
        self.use_fisheye   = config.USE_FISHEYE_MODEL

        # Per-video state (reset between videos)
        self._undistort_maps: Optional[tuple] = None

        # Sticky ankle position cache — retains last valid y-pixel values so that
        # brief YOLO confidence drops on static scenes don't lose the ankle position.
        self._l_ankle_y_cache: deque[float] = deque(maxlen=smoothing_window)
        self._r_ankle_y_cache: deque[float] = deque(maxlen=smoothing_window)

        # HAR LSTM classifier — loaded if checkpoint exists, else falls back to heuristic
        self._har_model:  Optional[OOPClassifier] = None
        self._har_device: torch.device            = torch.device("cpu")
        self._feat_buffer: deque                  = deque(maxlen=WINDOW_SIZE)

        # OOP state machine (fast trigger, slow release — prevents flickering)
        self._oop_state:   bool        = False
        self._neg_streak:  int         = 0
        self._trigger_buf: deque[int]  = deque(maxlen=config.OOP_TRIGGER_WINDOW)
        # Set by _is_oop each frame: True only when an ankle is positively detected
        # at floor level (y > ANKLE_PARTIAL_THRESHOLD).  Used by the exit logic so
        # that a static foot on the dashboard (YOLO confidence drop = missed frames)
        # does not cause the state machine to exit OOP state.
        self._ankle_at_floor: bool     = False

        har_path = Path(config.HAR_CLASSIFIER)
        if not har_path.is_absolute():
            har_path = Path(__file__).parent / har_path
        if har_path.exists():
            self._load_har_model(har_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-video state between videos."""
        self._undistort_maps = None
        self._l_ankle_y_cache.clear()
        self._r_ankle_y_cache.clear()
        self._feat_buffer.clear()
        self._oop_state      = False
        self._neg_streak     = 0
        self._ankle_at_floor = False
        self._trigger_buf.clear()

    # ------------------------------------------------------------------
    # HAR classifier loading
    # ------------------------------------------------------------------

    def _load_har_model(self, path: Path) -> None:
        """Load a trained OOPClassifier checkpoint from *path*."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt   = torch.load(str(path), map_location=device, weights_only=False)
            model  = OOPClassifier()
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._har_model  = model.to(device)
            self._har_device = device
            print(f"[INFO] HAR classifier loaded: {path}  (device={device})")
        except Exception as exc:
            print(f"[WARN] Could not load HAR classifier '{path}': {exc}")
            self._har_model = None

    def classify_frame(
        self, raw_frame: np.ndarray, frame_num: int = 0
    ) -> Tuple[str, str, np.ndarray]:
        """
        Classify a single raw video frame.

        Parameters
        ----------
        raw_frame  : BGR or grayscale frame from cv2.VideoCapture.
        frame_num  : 0-based frame index used for the on-screen counter.

        Returns
        -------
        raw_label      : 'POSITIVE' | 'NEGATIVE' | 'UNKNOWN'
        smoothed_label : temporally smoothed label (majority vote)
        annotated      : BGR frame with classification banner + skeleton
        """
        # Steps 1–3: preprocess
        processed = self._preprocess(raw_frame)
        bgr       = _to_bgr(processed)

        # Step 4: pose inference
        results = self.model(bgr, verbose=False)

        # Step 5: pick the passenger's keypoints
        kpts, person_idx = self._select_passenger(results)

        # Step 6: OOP decision — HAR classifier when available, else heuristic fallback
        if kpts is None:
            raw_label = "UNKNOWN"
            self._feat_buffer.append(np.zeros(FEAT_DIM, dtype=np.float32))
        else:
            feat = extract_features(kpts, config.KEYPOINT_CONF_THRESHOLD)
            self._feat_buffer.append(feat)

            if self._har_model is not None and len(self._feat_buffer) == WINDOW_SIZE:
                window = np.stack(list(self._feat_buffer), axis=0)          # (W, D)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, W, D)
                with torch.no_grad():
                    logit = self._har_model(x.to(self._har_device))
                    prob  = torch.sigmoid(logit).item()
                raw_label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
            else:
                raw_label = "POSITIVE" if self._is_oop(kpts, bgr.shape) else "NEGATIVE"

        # Step 7: OOP state machine — fast trigger, slow release
        # UNKNOWN is treated as NEGATIVE for the state machine.
        raw_positive = raw_label == "POSITIVE"
        self._trigger_buf.append(int(raw_positive))

        if not self._oop_state:
            # Enter OOP: need OOP_TRIGGER_COUNT positives within the trigger window
            if sum(self._trigger_buf) >= config.OOP_TRIGGER_COUNT:
                self._oop_state  = True
                self._neg_streak = 0
        else:
            # Exit OOP — only when there is positive evidence the foot has returned
            # to floor level.  A heuristic False caused by YOLO losing confidence on
            # a static ankle does NOT count as a negative — that would incorrectly
            # flip a resting foot on the dashboard back to NORMAL POSTURE.
            if raw_positive:
                self._neg_streak = 0
            elif kpts is None:
                # No person detected at all — count toward exit
                self._neg_streak += 1
                if self._neg_streak >= config.OOP_RELEASE_FRAMES:
                    self._oop_state  = False
                    self._neg_streak = 0
            elif self._ankle_at_floor:
                # Ankle positively detected at floor level — clear evidence of normal posture
                self._neg_streak += 1
                if self._neg_streak >= config.OOP_RELEASE_FRAMES:
                    self._oop_state  = False
                    self._neg_streak = 0
            # else: ankle elevated (in zone) but no OOP signal — foot may still be on
            # dashboard, YOLO just uncertain.  Hold OOP state.

        smoothed_label = "POSITIVE" if self._oop_state else "NEGATIVE"

        # Step 8: render annotated frame
        annotated = self._render(bgr, results, smoothed_label, raw_label, kpts, person_idx, frame_num)

        return raw_label, smoothed_label, annotated

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _build_undistort_maps(self, shape: tuple) -> tuple:
        """
        Build (map1, map2) LUTs for the chosen distortion model.

        Standard model  : cv2.getOptimalNewCameraMatrix + initUndistortRectifyMap
        Fisheye model   : cv2.fisheye.estimateNewCameraMatrixForUndistortRectify
                          + cv2.fisheye.initUndistortRectifyMap

        The LUTs are cached after the first call so remap is O(1) per frame.
        """
        h, w = shape[:2]

        if self.use_fisheye:
            # Fisheye equidistant model (suitable for 170° FOV lenses)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix,
                config.DIST_COEFFS_FISHEYE,
                (w, h),
                np.eye(3),
                balance=1.0,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                config.DIST_COEFFS_FISHEYE,
                np.eye(3),
                new_K,
                (w, h),
                cv2.CV_16SC2,
            )
        else:
            # Standard pinhole + Brown–Conrady model
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
            )

        return map1, map2

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Optionally undistort frame and/or flip horizontally."""
        if config.APPLY_UNDISTORTION:
            if self._undistort_maps is None:
                self._undistort_maps = self._build_undistort_maps(frame.shape)
            map1, map2 = self._undistort_maps
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        return frame

    # ------------------------------------------------------------------
    # Passenger selection
    # ------------------------------------------------------------------

    def _select_passenger(
        self, results: List[Any]
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Identify the front passenger among all detected persons.

        In a LHD vehicle viewed from the rear-view mirror (no flip):
          - Driver    → LEFT  side of frame
          - Passenger → RIGHT side of frame

        Strategy: pick the person whose hip-centre X is the largest (rightmost).
        """
        kpts_data = results[0].keypoints
        if kpts_data is None or len(kpts_data.data) == 0:
            return None, None

        all_kpts = kpts_data.data.cpu().numpy()   # (N, 17, 3)
        boxes    = results[0].boxes

        best_idx = None
        best_x   = -1.0

        for i, kpts in enumerate(all_kpts):
            if boxes is not None and float(boxes.conf[i]) < config.PERSON_CONF_THRESHOLD:
                continue

            cx = _torso_center_x(kpts, boxes, i)
            if cx is None:
                continue

            if cx > best_x:
                best_x   = cx
                best_idx = i

        if best_idx is None:
            return None, None

        return all_kpts[best_idx], best_idx

    # ------------------------------------------------------------------
    # OOP detection
    # ------------------------------------------------------------------

    def _is_oop(self, kpts: np.ndarray, frame_shape: tuple) -> bool:
        """
        Heuristic fallback — used only when the HAR LSTM model is not loaded.

        Rule: ankle is in the physical dashboard zone AND is at or above the
        calibrated dashboard line.  Both bounds come from the camera geometry,
        not from tuning against individual videos.

        Zone gate
        ---------
        OOP_ANKLE_ZONE_TOP (0.78) < ankle_y/H < ANKLE_PARTIAL_THRESHOLD (0.92)
          - Upper bound: ankle is off the floor  (floor appears at ~0.97)
          - Lower bound: ankle is not too high   (raised-folded knee sends the
            ankle toward the ceiling, typically y/H < 0.78)

        Direct check
        ------------
        ankle_y/H < DASHBOARD_Y (0.87)
          - Ankle is at or above the calibrated dashboard surface.
          - This is the only check.  All nuanced cases are handled by the LSTM.
        """
        H    = float(frame_shape[0])
        conf = config.KEYPOINT_CONF_THRESHOLD
        oop  = False
        self._ankle_at_floor = False

        ankle_conf = max(conf * 0.5, 0.12)

        for ankle_idx, ankle_cache in (
            (KP.L_ANKLE, self._l_ankle_y_cache),
            (KP.R_ANKLE, self._r_ankle_y_cache),
        ):
            ankle_vis = float(kpts[ankle_idx][2]) >= ankle_conf

            if ankle_vis:
                ankle_y = float(kpts[ankle_idx][1])
                ankle_cache.append(ankle_y)
                if (ankle_y / H) >= config.ANKLE_PARTIAL_THRESHOLD:
                    self._ankle_at_floor = True
            elif ankle_cache:
                ankle_y = float(np.mean(ankle_cache))
            else:
                ankle_y = None

            if ankle_y is None:
                continue

            ankle_norm = ankle_y / H

            # Gate: ankle must be in the physical dashboard zone
            if not (config.OOP_ANKLE_ZONE_TOP < ankle_norm < config.ANKLE_PARTIAL_THRESHOLD):
                continue

            # Single check: ankle is at or above the dashboard surface
            if ankle_norm < config.DASHBOARD_Y:
                oop = True

        return oop

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        frame:      np.ndarray,
        results:    List[Any],
        label:      str,
        raw_label:  str,
        pass_kpts:  Optional[np.ndarray],
        pass_idx:   Optional[int],
        frame_num:  int = 0,
    ) -> np.ndarray:
        """Overlay skeleton and classification banner on the processed frame."""
        out         = frame.copy()
        label_color = _label_color(label)

        if not config.CLEAN_OUTPUT:
            # Draw all detected skeletons; highlight the passenger's in cyan
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                all_kpts = results[0].keypoints.data.cpu().numpy()
                for i, kpts in enumerate(all_kpts):
                    is_passenger = i == pass_idx
                    color        = (0, 210, 210) if is_passenger else (160, 160, 160)
                    thickness    = 2 if is_passenger else 1
                    _draw_skeleton(out, kpts, color, thickness)

            # Highlight passenger ankles with label colour
            if pass_kpts is not None:
                for ankle_idx in (KP.L_ANKLE, KP.R_ANKLE):
                    pt = pass_kpts[ankle_idx]
                    if float(pt[2]) > config.KEYPOINT_CONF_THRESHOLD:
                        cx_px, cy_px = int(pt[0]), int(pt[1])
                        cv2.circle(out, (cx_px, cy_px), 10, label_color, -1)
                        cv2.circle(out, (cx_px, cy_px), 12, (255, 255, 255), 2)

        # Semi-transparent banner at the top
        banner = out.copy()
        cv2.rectangle(banner, (0, 0), (out.shape[1], config.BANNER_HEIGHT_PX), label_color, -1)
        cv2.addWeighted(banner, config.BANNER_ALPHA, out, 1 - config.BANNER_ALPHA, 0, out)

        # Centred status text
        display_text = config.DISPLAY_LABELS.get(label, label)
        (tw, th), _ = cv2.getTextSize(
            display_text, config.FONT, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS
        )
        tx = (out.shape[1] - tw) // 2
        ty = (config.BANNER_HEIGHT_PX + th) // 2
        cv2.putText(
            out, display_text, (tx, ty),
            config.FONT, config.LABEL_FONT_SCALE,
            (255, 255, 255), config.LABEL_THICKNESS, cv2.LINE_AA,
        )

        # Dashboard threshold line + per-ankle position readout
        if not config.CLEAN_OUTPUT and config.SHOW_STEERING_AXIS:
            H_px, W_px = out.shape[:2]
            dash_px = int(config.DASHBOARD_Y * H_px)

            # Full-width horizontal line at the dashboard threshold
            cv2.line(out, (0, dash_px), (W_px, dash_px), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"dashboard Y={config.DASHBOARD_Y:.2f}",
                        (8, dash_px - 6), config.FONT, 0.5, (0, 215, 255), 1, cv2.LINE_AA)

            # Per-ankle readout: normalised y position and OOP flag
            if pass_kpts is not None:
                ankle_conf = max(config.KEYPOINT_CONF_THRESHOLD * 0.5, 0.12)
                leg_render = [
                    (KP.L_KNEE, KP.L_ANKLE, (0, 140, 255), "L"),
                    (KP.R_KNEE, KP.R_ANKLE, (0, 220, 100), "R"),
                ]
                for knee_i, ankle_i, color, side in leg_render:
                    if float(pass_kpts[ankle_i][2]) < ankle_conf:
                        continue
                    ax = int(pass_kpts[ankle_i][0])
                    ay = int(pass_kpts[ankle_i][1])
                    ay_norm   = ay / H_px
                    oop_ankle = ay_norm < config.DASHBOARD_Y
                    flag      = " OOP" if oop_ankle else ""
                    cv2.putText(out, f"{side}:y={ay_norm:.2f}{flag}",
                                (ax + 12, ay - 8),
                                config.FONT, 0.45, color, 1, cv2.LINE_AA)

                    # Highlight knee-to-ankle shin segment
                    knee_ok = float(pass_kpts[knee_i][2]) >= config.KEYPOINT_CONF_THRESHOLD
                    if knee_ok:
                        kx = int(pass_kpts[knee_i][0]); ky = int(pass_kpts[knee_i][1])
                        seg_color = (0, 0, 220) if oop_ankle else color
                        cv2.line(out, (kx, ky), (ax, ay), seg_color, 3, cv2.LINE_AA)

        if not config.CLEAN_OUTPUT:
            # Raw per-frame label — bottom-left (shows unsmoothed single-frame result)
            raw_display = config.DISPLAY_LABELS.get(raw_label, raw_label)
            cv2.putText(
                out, f"Raw: {raw_display}",
                (8, out.shape[0] - 16),
                config.FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )

            # Frame counter — bottom-right corner
            cv2.putText(
                out, f"Frame {frame_num}",
                (out.shape[1] - 200, out.shape[0] - 16),
                config.FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
            )

        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert single-channel IR frame to 3-channel BGR for the pose model."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _torso_center_x(
    kpts: np.ndarray, boxes: Optional[Any], idx: int
) -> Optional[float]:
    """Estimate horizontal torso centre from hip keypoints (or bbox fallback)."""
    conf = config.KEYPOINT_CONF_THRESHOLD
    xs, ws = [], []
    for hip_idx in (KP.L_HIP, KP.R_HIP):
        pt = kpts[hip_idx]
        if float(pt[2]) > conf:
            xs.append(float(pt[0]))
            ws.append(float(pt[2]))
    if xs:
        return float(np.average(xs, weights=ws))
    if boxes is not None:
        box = boxes.xyxy[idx].cpu().numpy()
        return float((box[0] + box[2]) / 2)
    return None


def _label_color(label: str) -> tuple:
    if label == "POSITIVE":
        return config.POSITIVE_COLOR
    if label == "NEGATIVE":
        return config.NEGATIVE_COLOR
    return config.UNKNOWN_COLOR


def _draw_skeleton(
    frame: np.ndarray, kpts: np.ndarray, color: tuple, thickness: int = 2
) -> None:
    """Draw COCO-17 skeleton lines and keypoint dots."""
    conf = config.KEYPOINT_CONF_THRESHOLD
    for a, b in _SKELETON_PAIRS:
        pa, pb = kpts[a], kpts[b]
        if float(pa[2]) > conf and float(pb[2]) > conf:
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color, thickness, cv2.LINE_AA,
            )
    for kp in kpts:
        if float(kp[2]) > conf:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)