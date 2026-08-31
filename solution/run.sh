#!/usr/bin/env bash
# run.sh — Euro NCAP OOP | Feet-on-Dashboard Detector
# Linux / macOS one-click runner (equivalent of run.bat)
#
# Usage:
#   chmod +x run.sh   (only needed once)
#   ./run.sh
#
# Requirements:
#   - Python 3.10 or later must be installed and available on PATH
#   - Internet connection required on first run (auto-downloads yolov8n-pose.pt ~6 MB)
#   - GPU optional — CUDA used automatically if available, otherwise CPU is used

set -euo pipefail

# ── Move to the directory where this script lives ─────────────────────────────
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo "  Euro NCAP OOP | Feet-on-Dashboard Detector"
echo "================================================================"
echo "  Input   : ../input"
echo "  Output  : ../output"
echo "================================================================"
echo ""

# ── Check Python 3.10+ is available ───────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo ""
    echo "ERROR: python3 not found on PATH."
    echo "       Please install Python 3.10 or later:"
    echo "         Linux  : sudo apt install python3 python3-venv  (Debian/Ubuntu)"
    echo "                  sudo dnf install python3               (Fedora)"
    echo "         macOS  : brew install python@3.11"
    echo "                  or download from https://www.python.org"
    exit 1
fi

# Verify minimum Python version (3.10)
PY_VER=$(python3 -c "import sys; print(sys.version_info.major * 100 + sys.version_info.minor)")
if [ "$PY_VER" -lt 310 ]; then
    READABLE=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo ""
    echo "ERROR: Python 3.10 or later is required."
    echo "       Found: Python $READABLE"
    echo "       Please upgrade your Python installation."
    exit 1
fi

echo "Python version OK: $(python3 --version)"
echo ""

# ── Step 1: Create virtual environment ────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "[1/4] Removing existing .venv..."
    rm -rf .venv
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv || {
    echo ""
    echo "ERROR: Could not create virtual environment."
    echo "       On Debian/Ubuntu, run: sudo apt install python3-venv"
    exit 1
}

echo "[1/4] Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install -r requirements.txt || {
    echo ""
    echo "ERROR: Dependency installation failed."
    echo "       Check your internet connection and try again."
    exit 1
}

echo ""

# ── Step 2: Feature extraction (skip if already done) ─────────────────────────
if [ -f "features.npz" ]; then
    echo "[2/4] Skipping feature extraction  (features.npz already exists)"
else
    echo "[2/4] Extracting keypoint features from all input videos..."
    echo "      NOTE: yolov8n-pose.pt (~6 MB) will be auto-downloaded on first run."
    echo "            An internet connection is required for this step only."
    .venv/bin/python extract_features.py || {
        echo ""
        echo "ERROR: Feature extraction failed."
        exit 1
    }
fi

echo ""

# ── Step 3: LSTM training (skip if already done) ──────────────────────────────
if [ -f "oop_classifier.pt" ]; then
    echo "[3/4] Skipping training  (oop_classifier.pt already exists)"
else
    echo "[3/4] Training LSTM classifier..."
    .venv/bin/python train_classifier.py --epochs 80 --val-ratio 0 || {
        echo ""
        echo "ERROR: Training failed."
        exit 1
    }
fi

echo ""

# ── Step 4: Run detector on all videos ────────────────────────────────────────
echo "[4/4] Running detector on all input videos..."
.venv/bin/python main.py || {
    echo ""
    echo "ERROR: Detection failed."
    exit 1
}

echo ""
echo "================================================================"
echo "  Done!  Annotated videos written to ../output/"
echo "================================================================"
echo ""