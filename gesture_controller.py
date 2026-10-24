"""Webcam + MediaPipe HandLandmarker (Tasks API), pinch detection, pinch-drag state machine."""
import os, threading, time, urllib.request
import numpy as np, cv2

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

class PinchDetector:
    """Pinch = thumb-tip/index-tip distance divided by palm size (hand-scale invariant). Hysteresis + EMA smoothing."""
    def __init__(self, enter=0.35, exit=0.55, smooth=0.5):
        self.enter, self.exit, self.alpha = enter, exit, smooth; self.reset()
    def reset(self): self.pinch = False; self.cur = None; self.ratio = None; self.hand = False
    def update(self, lm):
        """lm: list of 21 (x,y) normalised landmarks, or None. Returns dict state."""
        if lm is None:
            self.hand = False; self.pinch = False; self.cur = None; self.ratio = None
            return self.state()
        p = np.asarray(lm, float); self.hand = True
        palm = max(np.linalg.norm(p[0] - p[9]), 1e-6)
        r = np.linalg.norm(p[4] - p[8]) / palm; tip = p[8]
        a = self.alpha
        self.ratio = r if self.ratio is None else a*r + (1-a)*self.ratio
        self.cur = tip if self.cur is None else a*tip + (1-a)*self.cur
        if not self.pinch and self.ratio < self.enter: self.pinch = True
        elif self.pinch and self.ratio > self.exit: self.pinch = False
        return self.state()
    def state(self):
        return dict(hand=self.hand, pinch=self.pinch, ratio=None if self.ratio is None else float(self.ratio),
                    cursor=None if self.cur is None else (float(self.cur[0]), float(self.cur[1])))

def map_to_video(nx, ny, W, H, mirror=True, margin=0.12):
    """Webcam-normalised -> video pixels. Mirrored so moving right moves right; `margin` crops camera edges so corners are reachable."""
    if mirror: nx = 1 - nx
    u = np.clip((nx - margin) / (1 - 2*margin), 0, 1); v = np.clip((ny - margin) / (1 - 2*margin), 0, 1)
    return float(u * W), float(v * H)

class DragController:
    """none -> hover -> drag. A pinch only grabs on its rising edge, and only if near a point."""
    def __init__(self, target, grab_frac=0.05): self.t = target; self.grab_frac = grab_frac; self.mode = "none"; self.sel = None; self.hover = None; self._prev = False
    def update(self, g, W, H, mirror=True, margin=0.12):
        rad = self.grab_frac * np.hypot(W, H); cur = None
        if g["hand"] and g["cursor"] is not None: cur = map_to_video(*g["cursor"], W, H, mirror, margin)
        if cur is None:
            self.sel = None; self.hover = None; self.mode = "none"; self._prev = False; return None
        pinch = g["pinch"]
        if pinch and not self._prev:
            i = self.t.nearest(cur, rad)
            if i is not None: self.sel = i; self.t.begin_drag(i); self.mode = "drag"
            else: self.mode = "pinch_empty"
        elif pinch and self.sel is not None: self.t.move(self.sel, cur); self.mode = "drag"
        elif not pinch:
            self.sel = None; self.hover = self.t.nearest(cur, rad); self.mode = "hover" if self.hover is not None else "none"
        self._prev = pinch; return cur

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True); urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

class GestureController(threading.Thread):
    """Own thread: capture -> landmarks -> pinch -> on_state(gesture_dict). Latest annotated JPEG kept for MJPEG preview."""
    def __init__(self, on_state, cam=0, detector=None):
        super().__init__(daemon=True); self.on_state = on_state; self.cam = cam; self.det = detector or PinchDetector()
        self.stop_evt = threading.Event(); self.jpeg = None; self.error = None; self.status = "starting"; self.lock = threading.Lock()
    def stop(self): self.stop_evt.set()
    def run(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions
            ensure_model()
            lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.VIDEO, num_hands=1))
            cap = cv2.VideoCapture(self.cam)
            if not cap.isOpened(): raise RuntimeError("Camera unavailable. On macOS: System Settings > Privacy & Security > Camera, allow your terminal, then retry.")
            self.status = "running"; t0 = time.time()
            while not self.stop_evt.is_set():
                ok, f = cap.read()
                if not ok: time.sleep(0.01); continue
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int((time.time() - t0) * 1000))
                pts = [(l.x, l.y) for l in res.hand_landmarks[0]] if res.hand_landmarks else None
                g = self.det.update(pts); self.on_state(g)
                self._draw(f, pts, g)
            cap.release(); lm.close(); self.status = "stopped"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"; self.status = "error"; self.on_state(self.det.update(None))
    def _draw(self, f, pts, g):
        h, w = f.shape[:2]
        if pts:
            P = [(int(x*w), int(y*h)) for x, y in pts]; col = (0, 180, 255) if g["pinch"] else (255, 220, 0)
            for a, b in CONN: cv2.line(f, P[a], P[b], col, 2)
            cv2.line(f, P[4], P[8], (0, 200, 255), 3); cv2.circle(f, P[8], 9, col, -1)
        cv2.putText(f, ("PINCH" if g["pinch"] else "hand") if g["hand"] else "no hand", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, b = cv2.imencode(".jpg", cv2.flip(f, 1))  # mirrored preview, matches cursor mapping
        if ok:
            with self.lock: self.jpeg = b.tobytes()
