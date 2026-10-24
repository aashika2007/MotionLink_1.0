"""Video loading and selected-object tracking (CSRT, KCF fallback)."""
import cv2, numpy as np

MAX_W = 960
class VideoSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise ValueError("Cannot open video (try MP4/MOV/AVI with a standard codec).")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, f = self.cap.read()
            if not ok: break
            s = min(1.0, MAX_W / f.shape[1]); frames.append(cv2.resize(f, None, fx=s, fy=s) if s < 1 else f)
        self.cap.release()
        if not frames: raise ValueError("Video has no readable frames.")
        self.frames = frames; self.n = len(frames); self.h, self.w = frames[0].shape[:2]
        self.diag = float(np.hypot(self.w, self.h)); self.times = np.arange(self.n) / self.fps
    def jpeg(self, i):
        ok, b = cv2.imencode(".jpg", self.frames[int(np.clip(i, 0, self.n - 1))], [cv2.IMWRITE_JPEG_QUALITY, 85]); return b.tobytes()

class TrackResult:
    def __init__(self, n):
        self.centers = np.full((n, 2), np.nan); self.boxes = np.full((n, 4), np.nan)
        self.status = "idle"; self.message = "Pick a frame, drag a box around the car, press Track."; self.version = 0

def _make_tracker():
    for mod, name in ((cv2, "TrackerCSRT_create"), (getattr(cv2, "legacy", None), "TrackerCSRT_create"), (cv2, "TrackerKCF_create")):
        if mod is not None and hasattr(mod, name): return getattr(mod, name)()
    raise RuntimeError("No OpenCV tracker available; install opencv-contrib-python.")

def track(src, res, start, box, stop_flag, max_jump_frac=0.2):
    """Track forward from `start`. Frames >= start are cleared first. Stops (does NOT bridge) on loss."""
    res.centers[start:] = np.nan; res.boxes[start:] = np.nan; res.status = "tracking"
    x, y, w, h = [int(round(float(v))) for v in box]
    if w < 4 or h < 4: res.status, res.message = "error", "Selection too small."; return
    tr = _make_tracker(); tr.init(src.frames[start], (x, y, w, h))
    res.boxes[start] = (x, y, w, h); res.centers[start] = (x + w/2, y + h/2); prev = res.centers[start].copy()
    for i in range(start + 1, src.n):
        if stop_flag(): res.status, res.message = "stopped", f"Stopped at frame {i}."; res.version += 1; return
        ok, bb = tr.update(src.frames[i]); c = np.array([bb[0] + bb[2]/2, bb[1] + bb[3]/2]) if ok else None
        if not ok or np.linalg.norm(c - prev) > max_jump_frac * src.diag:
            res.status, res.message = "lost", f"Lost the car at frame {i}. Scrub to a frame where it is visible, re-select, Track again."
            res.version += 1; return
        res.boxes[i] = bb; res.centers[i] = c; prev = c
        if i % 5 == 0: res.version += 1
    res.status, res.message = "done", f"Tracked frames {start}-{src.n - 1}."; res.version += 1
