@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ================================================================
echo   Euro NCAP OOP ^| Feet-on-Dashboard Detector
echo ================================================================
echo   Input   : ..\input
echo   Output  : ..\output
echo ================================================================
echo.

REM ── Check Python is available ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found on PATH.
    echo        Please install Python 3.10 or later from https://www.python.org
    echo        and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM ── Step 1: Create virtual environment (always fresh) ───────────
if exist .venv (
    echo [1/4] Removing existing .venv...
    rmdir /s /q .venv
)
echo [1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not create virtual environment.
    pause
    exit /b 1
)
echo [1/4] Installing dependencies...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.

REM ── Step 2: Feature extraction (skip if already done) ────────────
if exist features.npz (
    echo [2/4] Skipping feature extraction  ^(features.npz already exists^)
) else (
    echo [2/4] Extracting keypoint features from all input videos...
    .venv\Scripts\python extract_features.py
    if errorlevel 1 (
        echo.
        echo ERROR: Feature extraction failed.
        pause
        exit /b 1
    )
)

echo.

REM ── Step 3: LSTM training (skip if already done) ─────────────────
if exist oop_classifier.pt (
    echo [3/4] Skipping training  ^(oop_classifier.pt already exists^)
) else (
    echo [3/4] Training LSTM classifier...
    .venv\Scripts\python train_classifier.py --epochs 80 --val-ratio 0
    if errorlevel 1 (
        echo.
        echo ERROR: Training failed.
        pause
        exit /b 1
    )
)

echo.

REM ── Step 4: Run detector on all videos ───────────────────────────
echo [4/4] Running detector on all input videos...
.venv\Scripts\python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Detection failed.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Done!  Annotated videos written to ..\output\
echo ================================================================
echo.
pause
