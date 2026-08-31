# OOP Case 1 — Feet on Dashboard Detector

## Overview

This solution detects whether the front passenger has one or both feet
resting on the dashboard (Euro NCAP OOP Case 1) from IR in-cabin video
footage. It uses a hybrid approach — YOLOv8-Pose for body keypoint
extraction, a Bidirectional LSTM for activity recognition, and a
rule-based heuristic as a fallback. Every frame of every input video
is classified as POSITIVE (foot on dashboard), NEGATIVE (normal
posture), or UNKNOWN (no detection).

For full methodology, results, and findings refer to README.pdf.

---

## Folder Structure

```
submission/
├── input/                  # Input videos
│   ├── profile1/
│   ├── profile2/
│   └── profile3/
├── output/                 # Annotated output videos (auto-generated)
│   ├── profile1/
│   ├── profile2/
│   └── profile3/
├── assets/                 # Reference images
├── solution/               # All source code and models
│   ├── config.py
│   ├── detector.py
│   ├── extract_features.py
│   ├── main.py
│   ├── oop_model.py
│   ├── train_classifier.py
│   ├── video_processor.py
│   ├── requirements.txt
│   ├── run.bat
│   ├── run.sh
│   ├── features.npz
│   ├── oop_classifier.pt
│   └── yolov8n-pose.pt
├── README.md               # This file
└── README.pdf              # Full report
```

---

## Requirements

- Python 3.10 or later
- Internet connection on first run only
  (auto-downloads `yolov8n-pose.pt` ~6 MB)
- GPU not required
  (CUDA used automatically if available, otherwise CPU)

---

## How to Run

Navigate to the `solution/` folder and run the appropriate
script for your operating system.

**Windows:**
```bash
run.bat
```

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

The script will automatically:
1. Create a Python virtual environment
2. Install all dependencies
3. Extract keypoint features from input videos
   (skipped if `features.npz` already exists)
4. Train the LSTM classifier
   (skipped if `oop_classifier.pt` already exists)
5. Run the detector and write output videos to `../output/`

---

## Manual Steps

If you prefer to run each step individually from inside
the `solution/` folder:

```bash
# Step 1 — Install dependencies
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Step 2 — Extract features
python extract_features.py

# Step 3 — Train classifier
python train_classifier.py --epochs 80 --val-ratio 0

# Step 4 — Run detector
python main.py
```

---

## Custom Paths (Optional)

```bash
python main.py --input path/to/input --output path/to/output
```

---

## Output

Each output video is an annotated copy of the input with a
colour-coded banner on every frame:

-  Green — `NORMAL POSTURE` (NEGATIVE)
-  Red — `FOOT ON DASHBOARD` (POSITIVE)
-  Grey — `NO DETECTION` (UNKNOWN)

A scrolling timeline graph showing detection history is
displayed at the bottom of each frame.

---

## Full Report

See `README.pdf` for complete documentation including:
- Methodology and system architecture
- Pipeline flowchart
- Results and key findings
- Assumptions and limitations
- Future improvements
- References
