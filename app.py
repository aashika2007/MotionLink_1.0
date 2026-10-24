"""MotionInk server. Run: python app.py  ->  http://127.0.0.1:5000"""
import os, threading, time, numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from car_tracker import VideoSource, TrackResult, track
from gesture_controller import GestureController, PinchDetector, DragController
from trajectory import TargetPath, compare, segments

app = Flask(__name__, static_folder="static"); UP = os.path.join(os.path.dirname(__file__), "uploads"); os.makedirs(UP, exist_ok=True)
S = dict(src=None, res=None, stop=False, gc=None, g=dict(hand=False, pinch=False, ratio=None, cursor=None), cur=None)
target = TargetPath(); det = PinchDetector(); drag = DragController(target); cfg = dict(margin=0.12, mirror=True); lock = threading.Lock()

def on_gesture(g):
    with lock:
        S["g"] = g; src = S["src"]; S["cur"] = drag.update(g, src.w, src.h, cfg["mirror"], cfg["margin"]) if src else None

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.post("/api/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename.lower().endswith((".mp4", ".mov", ".avi")): return jsonify(error="Upload an MP4, MOV or AVI file."), 400
    S["stop"] = True; p = os.path.join(UP, "current" + os.path.splitext(f.filename)[1].lower()); f.save(p)
    try: src = VideoSource(p)
    except Exception as e: return jsonify(error=str(e)), 400
    with lock: S.update(src=src, res=TrackResult(src.n)); target.clear()
    return jsonify(ok=True)

@app.get("/api/frame/<int:i>")
def frame(i):
    if not S["src"]: return "", 404
    return Response(S["src"].jpeg(i), mimetype="image/jpeg")

@app.post("/api/track")
def start_track():
    d = request.json; src, res = S["src"], S["res"]
    if not src: return jsonify(error="Load a video first."), 400
    if res.status == "tracking": return jsonify(error="Already tracking."), 400
    S["stop"] = False; f = int(d["frame"])
    def job():
        try: track(src, res, f, d["box"], lambda: S["stop"])
        except Exception as e: res.status, res.message = "error", str(e)
        res.version += 1
        if res.status in ("done", "lost", "stopped"): target.init_from(res.centers) if not len(target.cp) else None
    threading.Thread(target=job, daemon=True).start(); return jsonify(ok=True)

@app.post("/api/stop_track")
def stop_track(): S["stop"] = True; return jsonify(ok=True)

@app.post("/api/target/<op>")
def target_op(op):
    with lock:
        if op == "undo": target.undo()
        elif op == "reset": target.reset()
        elif op == "rebuild" and S["res"]: target.init_from(S["res"].centers)
    return jsonify(ok=True)

@app.post("/api/reset")  # full demo reset: keep video, clear tracking + target
def reset():
    S["stop"] = True; time.sleep(0.05)
    with lock:
        if S["src"]: S["res"] = TrackResult(S["src"].n)
        target.clear()
    return jsonify(ok=True)

@app.post("/api/camera")
def camera():
    on = request.json.get("on"); gc = S["gc"]
    if gc and gc.is_alive(): gc.stop()
    if on:
        det.reset(); S["gc"] = GestureController(on_gesture, detector=det); S["gc"].start()
    return jsonify(ok=True)

@app.post("/api/config")
def config():
    d = request.json
    for k in ("enter", "exit"): 
        if k in d: setattr(det, k, float(d[k]))
    if "margin" in d: cfg["margin"] = float(np.clip(d["margin"], 0, 0.3))
    return jsonify(ok=True)

@app.get("/api/webcam")
def webcam():
    def gen():
        while True:
            gc = S["gc"]; b = gc.jpeg if gc else None
            if b: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

def _runs(res):
    return [[[round(float(x), 1), round(float(y), 1)] for x, y in res.centers[a:b+1]] for a, b in segments(res.centers)]

@app.get("/api/actual")
def actual():
    res = S["res"]; return jsonify(segs=_runs(res) if res else [], starts=[a for a, _ in segments(res.centers)] if res else [])

@app.get("/api/analysis")
def analysis():
    src, res = S["src"], S["res"]
    if not src: return jsonify(m=None)
    with lock: poly = target.polyline(); cmp = compare(res.centers, poly, src.diag)
    if not cmp: return jsonify(m=None)
    d = [None if np.isnan(v) else round(float(v), 1) for v in cmp["dist"]]
    return jsonify(m=dict(mean_px=cmp["mean_px"], max_px=cmp["max_px"], mean_pct=cmp["mean_pct_diag"], frac=cmp["frac_diverging"], thresh=cmp["thresh"]), dist=d)

@app.get("/api/state")
def state():
    src, res, gc = S["src"], S["res"], S["gc"]
    with lock:
        t = dict(cp=target.cp.round(1).tolist(), poly=target.polyline().round(1).tolist(), v=target.version, sel=drag.sel, hover=drag.hover, mode=drag.mode)
        g = dict(S["g"]); cur = S["cur"]
    return jsonify(video=None if not src else dict(n=src.n, fps=src.fps, w=src.w, h=src.h),
                   track=None if not res else dict(status=res.status, msg=res.message, v=res.version),
                   target=t, gesture=g, cursor=cur, cam=dict(status=gc.status if gc else "off", error=gc.error if gc else None))

if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, threaded=True)
