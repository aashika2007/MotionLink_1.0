import numpy as np, cv2, os, tempfile, time, io, threading
from trajectory import *
from gesture_controller import PinchDetector, DragController, map_to_video
from car_tracker import VideoSource, TrackResult, track

def hand(ratio, tip=(0.5, 0.5)):  # synthetic 21 landmarks: palm size 0.2, thumb-index gap = ratio*0.2
    lm = np.zeros((21, 2)); lm[0] = (0.5, 0.8); lm[9] = (0.5, 0.6); lm[8] = tip; lm[4] = (tip[0] + ratio*0.2, tip[1]); return lm

def test_traj():
    c = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]); poly = np.array([[0, 10.], [100, 10.]])
    assert np.allclose(dist_to_polyline(c, poly), 10)
    cm = c.copy(); cm[20:30] = np.nan; assert segments(cm) == [(0, 19), (30, 49)]  # gap not bridged
    m = compare(cm, poly, 141.4); assert abs(m["mean_px"] - 10) < 1e-6 and compare(np.full((5, 2), np.nan), poly, 1) is None
    # Same line, but the car brakes then speeds up: the identical path must not count as deviation.
    xs = 100 * np.r_[np.linspace(0, 0.3, 30), np.linspace(0.3, 0.4, 40), np.linspace(0.4, 1, 30)[1:]]
    slow = np.column_stack([xs, np.zeros_like(xs)]); same = np.array([[0., 0.], [100., 0.]])
    assert compare(slow, same, 141.4)["mean_px"] < 1e-6
    assert compare(c[::-1].copy(), np.array([[0., 0.], [100., 0.]]), 141.4)["mean_px"] > 40  # reversed path still fails
    t = TargetPath(); assert t.init_from(c); t.begin_drag(0); t.move(3, (5, 5)); t.undo(); assert np.allclose(t.cp, t.orig)
    t.move(1, (1, 1)); t.reset(); assert np.allclose(t.cp, t.orig)

def test_mapping():
    assert map_to_video(0.5, 0.5, 960, 540) == (480.0, 270.0)
    assert map_to_video(0.0, 0.5, 960, 540, True)[0] == 960.0 and map_to_video(0.0, 0.5, 960, 540, False)[0] == 0.0

def test_pinch_drag():
    d = PinchDetector(smooth=1.0); t = TargetPath(); t.cp = np.array([[480., 270.], [100., 100.]]); t.orig = t.cp.copy(); dc = DragController(t)
    run = lambda lm: dc.update(d.update(lm), 960, 540)
    run(hand(1.0)); assert dc.mode == "hover"
    run(hand(0.1)); assert d.pinch and dc.mode == "drag" and dc.sel == 0
    run(hand(0.3, (0.3, 0.4))); assert d.pinch  # hysteresis: 0.3 < exit 0.55 stays pinched
    exp = map_to_video(0.3, 0.4, 960, 540); assert np.allclose(t.cp[0], exp)
    run(hand(0.9, (0.3, 0.4))); assert not d.pinch and dc.sel is None; before = t.cp.copy()
    run(hand(0.9, (0.1, 0.1))); assert np.allclose(t.cp, before)  # released: no movement
    run(None); assert dc.mode == "none"
    run(hand(0.1, (0.9, 0.9))); assert dc.mode == "pinch_empty"; run(hand(0.1, (0.4, 0.4))); assert np.allclose(t.cp[0], before[0])  # empty pinch grabs nothing

def synth(path):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360)); rng = np.random.default_rng(0)
    bg = (rng.random((360, 640, 3)) * 60).astype(np.uint8)
    for i in range(90):
        f = bg.copy(); x = 60 + i * 6; y = int(180 + 80 * np.sin(i / 14)); cv2.rectangle(f, (x, y - 20), (x + 60, y + 20), (0, 0, 255), -1); cv2.rectangle(f, (x + 10, y - 8), (x + 30, y + 8), (255, 255, 0), -1); w.write(f)
    w.release()

def test_tracking():
    p = os.path.join(tempfile.mkdtemp(), "s.avi"); synth(p); s = VideoSource(p); r = TrackResult(s.n); assert s.n == 90
    track(s, r, 10, (60 + 60, 180 + 80 * np.sin(10 / 14) - 20, 60, 40), lambda: False)
    ok = ~np.isnan(r.centers[:, 0]); assert r.status == "done" and ok[10:].all() and not ok[:10].any(), r.message
    gt = np.array([[60 + i*6 + 30, 180 + 80*np.sin(i/14)] for i in range(90)]); err = np.linalg.norm(r.centers[10:] - gt[10:], axis=1)
    assert err.mean() < 8, err.mean()
    r2 = TrackResult(s.n); track(s, r2, 10, (120, 195, 60, 40), lambda: True); assert r2.status == "stopped"

def test_server():
    import app as A; c = A.app.test_client(); p = os.path.join(tempfile.mkdtemp(), "s.avi"); synth(p)
    assert c.post("/api/upload", data={"video": (open(p, "rb"), "s.avi")}).status_code == 200
    assert c.post("/api/upload", data={"video": (io.BytesIO(b"x"), "s.txt")}).status_code == 400
    assert c.get("/api/frame/5").mimetype == "image/jpeg"
    c.post("/api/track", json={"frame": 10, "box": [120, 195, 60, 40]})
    for _ in range(100):
        if c.get("/api/state").json["track"]["status"] in ("done", "lost"): break
        time.sleep(0.2)
    time.sleep(0.2); s = c.get("/api/state").json; assert s["track"]["status"] == "done" and len(s["target"]["cp"]) >= 2
    a = c.get("/api/analysis").json; assert a["m"]["mean_px"] < 15
    A.target.begin_drag(0); A.target.move(0, (10, 10)); assert c.get("/api/analysis").json["m"]["mean_px"] > a["m"]["mean_px"]
    assert c.post("/api/target/move", json={"i": 1, "xy": [300, 200], "begin": True}).status_code == 200
    assert np.allclose(A.target.cp[1], (300, 200)); c.post("/api/target/undo"); assert not np.allclose(A.target.cp[1], (300, 200))
    assert c.post("/api/target/move", json={"i": 99, "xy": [1, 1]}).status_code == 400
    c.post("/api/target/move", json={"i": 0, "xy": [-50, 9999]}); assert np.allclose(A.target.cp[0], (0, 359))  # clamped to frame
    c.post("/api/target/reset"); assert abs(c.get("/api/analysis").json["m"]["mean_px"] - a["m"]["mean_px"]) < 1e-6
    c.post("/api/reset"); s = c.get("/api/state").json; assert s["track"]["status"] == "idle" and s["target"]["cp"] == []
    assert c.get("/").status_code == 200

def test_demo_clip():
    import app as A; c = A.app.test_client()
    r = c.post("/api/demo", json={}); assert r.status_code == 200 and r.json["synthetic"] and r.json["frames"] == 150
    for _ in range(300):
        if c.get("/api/state").json["track"]["status"] in ("done", "lost", "error"): break
        time.sleep(0.2)
    s = c.get("/api/state").json; assert s["track"]["status"] == "done", s["track"]
    truth = np.array(A.S["demo_info"]["car1"]); got = A.S["res"].centers
    err = np.linalg.norm(got - truth, axis=1); assert np.nanmean(err) < 12, np.nanmean(err)  # vs known true positions
    a = c.get("/api/analysis").json; assert a["m"] is not None and len(s["target"]["cp"]) >= 2
    c.post("/api/reset")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
