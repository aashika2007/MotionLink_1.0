# MotionInk — Draw It. Bend It. Repeat It.
Track a selected car in a video, then reshape a target path with **pinch gestures** from your webcam and compare it with the recorded path.

## Install (MacBook Air M2, Python 3.12)
    cd MotionInk
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
## Install and launch on Windows
Double-click `run.bat` (or type `run.bat` in Command Prompt, `.\run.bat` in PowerShell). It installs the requirements, starts the server and opens http://127.0.0.1:5000.

## Launch
    python app.py      # then open http://127.0.0.1:5000
## Test (no webcam needed)
    python test_motionink.py

## Quick test without a video
Click **Load demo clip**. It generates a **synthetic** top-down corner with two cars (`demo_clip.py`; not real footage, no licensing issues), loads it, and automatically tracks Car 1 (orange) from frame 0. On a laptop this takes a few seconds; then edit the cyan points in panel B.

## Use
1. Upload MP4/MOV/AVI. Scrub to a frame where the car is clearly visible.
2. **Select car** → drag a box around it → **Track from this frame**. If tracking is lost, scrub to a frame where the car is visible, re-select, and track again (frames in between stay empty; gaps are never joined).
3. **Start camera** (allow camera access for your terminal). First start downloads the ~8 MB MediaPipe `hand_landmarker.task` model into `models/` (internet needed once).
4. Move your index finger: ring = cursor. Hover a cyan point (white ring), **pinch** thumb+index to grab (amber), move, release to drop. Undo / Reset target as fallbacks.
5. Toggle target/deviation. The amber car-sized box animates along the calculated target path as you play or scrub the video. The comparison graph plots per-frame distance and cumulative mean deviation in pixels.
Tuning: *Pinch threshold* slider, *Edge margin* (shrinks the camera area mapped to the video so corners are reachable).

## Dashboard layout
- **Configuration A, recorded line:** the tracked car's path on the video, coloured slow → fast by image-space speed (px/s). Stars mark the sharpest turns.
- **Configuration B, edited line:** the editable target (cyan points) on the same frame, coloured green/yellow/red by distance from the recorded line. The amber box is where the edited line puts the car at the current frame.
- **Editing:** drag a cyan point with the mouse, or start the camera and pinch it. Undo, Reset line and Rebuild from track are in the footer.
- **Bottom row:**
  - recorded speed per frame,
  - turn rate for A (solid) and B (dashed),
  - comparison stats,
  - distance A → B per frame with a running mean,
  - hand control (camera preview, pinch threshold, edge margin).
- **Footer:** frame slider, Play, Pause, Restart, and Reset (clears tracking and target, keeps the video). Keyboard: space plays and pauses; ← and → step one frame (Shift for 10).
- **Old page:** the previous single-page UI is kept at `/static/index_original.html`.
- **Units:** all numbers are video pixels. There is no telemetry (steering, throttle, brake, g-force, tyres, lap times), so the dashboard doesn't show any.

## Notes
- Uses the MediaPipe **Tasks** API (`HandLandmarker`); `mp.solutions` no longer exists in mediapipe 0.10.33, which is why legacy code fails.
- Streamlit was avoided: reruns fight with continuous webcam capture and drag editing. Flask + browser canvas keeps capture, tracking and UI on separate threads.
- Metric = mean/max distance (video pixels) from each recorded point to the target position at the same fraction of distance travelled. Pacing by distance (not frame count) means a car that brakes or accelerates is not counted as deviating from an identical line; order still matters, so a reversed path still scores badly. It is a geometric measure only, not a driving-quality score. Editing the target never changes recorded motion.
- Tracks one selected object (CSRT); it does not auto-recognise vehicles. Video is downscaled to 960 px wide in memory (keep clips short).
## Troubleshooting
Camera error → macOS Settings › Privacy & Security › Camera. Model download fails → download `hand_landmarker.task` from the MediaPipe site into `models/`. CSRT missing → `pip install opencv-contrib-python` (and uninstall plain `opencv-python`).
