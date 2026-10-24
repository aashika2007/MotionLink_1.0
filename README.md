# MotionInk — Draw It. Bend It. Repeat It.
Track a selected car in a video, then reshape a target path with **pinch gestures** from your webcam and compare it with the recorded path.

## Install (MacBook Air M2, Python 3.12)
    cd MotionInk
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
## Launch
    python app.py      # then open http://127.0.0.1:5000
## Test (no webcam needed)
    python test_motionink.py

## Use
1. Upload MP4/MOV/AVI. Scrub to a frame where the car is clearly visible.
2. **Select car** → drag a box around it → **Track from this frame**. If tracking is lost, scrub to a frame where the car is visible, re-select, and track again (frames in between stay empty; gaps are never joined).
3. **Start camera** (allow camera access for your terminal). First start downloads the ~8 MB MediaPipe `hand_landmarker.task` model into `models/` (internet needed once).
4. Move your index finger: ring = cursor. Hover a cyan point (white ring), **pinch** thumb+index to grab (amber), move, release to drop. Undo / Reset target as fallbacks.
5. Toggle target/deviation. Amber segments on the violet path = recorded path farther than 3% of the frame diagonal from the target.
Tuning: *Pinch threshold* slider, *Edge margin* (shrinks the camera area mapped to the video so corners are reachable).

## Notes
- Uses the MediaPipe **Tasks** API (`HandLandmarker`); `mp.solutions` no longer exists in mediapipe 0.10.33, which is why legacy code fails.
- Streamlit was avoided: reruns fight with continuous webcam capture and drag editing. Flask + browser canvas keeps capture, tracking and UI on separate threads.
- Metric = mean/max distance (video pixels) from recorded points to the target polyline. It is a geometric measure only, not a driving-quality score. Editing the target never changes recorded motion.
- Tracks one selected object (CSRT); it does not auto-recognise vehicles. Video is downscaled to 960 px wide in memory (keep clips short).
## Troubleshooting
Camera error → macOS Settings › Privacy & Security › Camera. Model download fails → download `hand_landmarker.task` from the MediaPipe site into `models/`. CSRT missing → `pip install opencv-contrib-python` (and uninstall plain `opencv-python`).
