"""Synthetic demo clip: a top-down left-hand corner with two cars.

Used to develop and rehearse without network access or licensing questions,
and as ground truth for the automated tracking tests. Car 2 leads on an
outside–inside–outside line; Car 1 follows slightly inside it, closes in and
has its closest approach near the apex.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

W, H = 960, 540
FPS = 30.0
N_FRAMES = 150
TRACK_HALF_WIDTH = 85.0
CAR_LEN, CAR_WID = 64.0, 28.0

# Centreline: straight east, 90° left-hand arc, straight north.
_ARC_C, _ARC_R = np.array([500.0, 200.0]), 240.0
_S_STRAIGHT1 = 600.0                          # from x = -100 to x = 500
_S_ARC = _ARC_R * math.pi / 2
_S_APEX = _S_STRAIGHT1 + _S_ARC / 2


def centreline(s):
    s = np.atleast_1d(np.asarray(s, dtype=float))
    pts = np.zeros((len(s), 2))
    tan = np.zeros((len(s), 2))
    for i, v in enumerate(s):
        if v <= _S_STRAIGHT1:
            pts[i] = (-100 + v, 440.0)
            tan[i] = (1, 0)
        elif v <= _S_STRAIGHT1 + _S_ARC:
            th = math.pi / 2 - (v - _S_STRAIGHT1) / _ARC_R
            pts[i] = _ARC_C + _ARC_R * np.array([math.cos(th), math.sin(th)])
            tan[i] = (math.sin(th), -math.cos(th))
        else:
            pts[i] = (740.0, 200.0 - (v - _S_STRAIGHT1 - _S_ARC))
            tan[i] = (0, -1)
    return pts, tan


def _ease(a, b, s, s0, s1):
    u = np.clip((s - s0) / (s1 - s0), 0, 1)
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u))


def lateral_line(s, inside_bias=0.0):
    """Outside–inside–outside racing line, + is right of travel (outside)."""
    s = np.asarray(s, dtype=float)
    lat = np.where(s < _S_APEX, _ease(55, -58, s, 420, _S_APEX), _ease(-58, 55, s, _S_APEX, 1150))
    return lat - inside_bias


def car_positions(s, inside_bias=0.0):
    c, t = centreline(s)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    lat = lateral_line(s, inside_bias)
    pos = c + n * lat[:, None]
    return pos


def schedules():
    t = np.arange(N_FRAMES, dtype=float)
    s2 = np.zeros(N_FRAMES)
    s2[0] = 330.0
    for i in range(1, N_FRAMES):
        slow = math.exp(-((s2[i - 1] - _S_APEX) / 180.0) ** 2)
        s2[i] = s2[i - 1] + 5.5 - 1.6 * slow
    closing = np.clip(1 - t / 95, 0, 1) ** 1.5
    gap = np.where(t <= 95, 74 + 96 * closing, 74 + 20 * (t - 95) / (N_FRAMES - 95))
    s1 = s2 - gap
    return s1, s2


def boundaries():
    s = np.linspace(0, _S_STRAIGHT1 + _S_ARC + 420, 120)
    c, t = centreline(s)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    inner = c - n * TRACK_HALF_WIDTH
    outer = c + n * TRACK_HALF_WIDTH
    return [np.round(inner, 1).tolist(), np.round(outer, 1).tolist()]


def _background(rng):
    img = np.zeros((H, W, 3), np.uint8)
    grass = rng.normal(0, 9, (H, W, 1)) + np.array([58, 118, 52])
    img[:] = np.clip(grass, 0, 255).astype(np.uint8)
    s = np.linspace(-50, _S_STRAIGHT1 + _S_ARC + 500, 400)
    c, t = centreline(s)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    outer = (c + n * (TRACK_HALF_WIDTH + 26)).astype(np.int32)
    inner = (c - n * (TRACK_HALF_WIDTH + 26)).astype(np.int32)
    runoff = np.vstack([outer, inner[::-1]])
    cv2.fillPoly(img, [runoff], (150, 160, 170))           # tarmac run-off / gravel
    road = np.vstack([(c + n * TRACK_HALF_WIDTH).astype(np.int32),
                      (c - n * TRACK_HALF_WIDTH).astype(np.int32)[::-1]])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [road], 255)
    asphalt = np.clip(rng.normal(0, 6, (H, W, 1)) + np.array([72, 72, 76]), 0, 255).astype(np.uint8)
    img[mask > 0] = asphalt[mask > 0]
    for sign in (1, -1):
        edge = (c + n * sign * TRACK_HALF_WIDTH).astype(np.int32)
        cv2.polylines(img, [edge], False, (235, 235, 235), 3, cv2.LINE_AA)
    # Kerbs: inside of the arc (apex) and outside of the exit.
    for lo, hi, sign in ((_S_STRAIGHT1 + 40, _S_STRAIGHT1 + _S_ARC - 40, -1),
                         (_S_STRAIGHT1 + _S_ARC - 60, _S_STRAIGHT1 + _S_ARC + 120, 1)):
        ks = np.arange(lo, hi, 14)
        kc, kt = centreline(ks)
        kn = np.column_stack([-kt[:, 1], kt[:, 0]])
        for j in range(len(ks) - 1):
            a1 = kc[j] + kn[j] * sign * (TRACK_HALF_WIDTH + 1)
            a2 = kc[j + 1] + kn[j + 1] * sign * (TRACK_HALF_WIDTH + 1)
            b1 = kc[j] + kn[j] * sign * (TRACK_HALF_WIDTH + 13)
            b2 = kc[j + 1] + kn[j + 1] * sign * (TRACK_HALF_WIDTH + 13)
            col = (40, 40, 215) if j % 2 == 0 else (240, 240, 240)
            cv2.fillPoly(img, [np.array([a1, a2, b2, b1], np.int32)], col, cv2.LINE_AA)
    # Braking boards on the approach.
    for k, (x, label) in enumerate(((330, "150"), (400, "100"), (460, "50"))):
        y = 440 - TRACK_HALF_WIDTH - 50
        cv2.rectangle(img, (x - 14, int(y) - 9), (x + 14, int(y) + 9), (245, 245, 245), -1)
        cv2.putText(img, label, (x - 12, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA)
    return img


def _draw_car(img, centre, heading, body, accent, number):
    ca, sa = math.cos(heading), math.sin(heading)

    def tf(pts):
        p = np.asarray(pts, float)
        x = p[:, 0] * ca - p[:, 1] * sa + centre[0]
        y = p[:, 0] * sa + p[:, 1] * ca + centre[1]
        return np.column_stack([x, y]).astype(np.int32)

    L, Wd = CAR_LEN / 2, CAR_WID / 2
    wheel = np.array([[-6, -4], [6, -4], [6, 4], [-6, 4]])
    for wx, wy in ((L - 12, Wd - 2), (L - 12, -Wd + 2), (-L + 10, Wd - 1), (-L + 10, -Wd + 1)):
        cv2.fillPoly(img, [tf(wheel + [wx, wy])], (25, 25, 25), cv2.LINE_AA)
    cv2.fillPoly(img, [tf([[L, -3], [L - 4, -Wd + 1], [L - 4, Wd - 1], [L, 3]])], accent, cv2.LINE_AA)
    chassis = [[L - 2, -2], [L - 2, 2], [8, 7], [-12, Wd - 4], [-L + 4, 7], [-L + 4, -7], [-12, -Wd + 4], [8, -7]]
    cv2.fillPoly(img, [tf(chassis)], body, cv2.LINE_AA)
    cv2.fillPoly(img, [tf([[-L + 1, -Wd + 2], [-L + 7, -Wd + 2], [-L + 7, Wd - 2], [-L + 1, Wd - 2]])], accent, cv2.LINE_AA)
    cv2.fillPoly(img, [tf([[4, -3], [4, 3], [-6, 4], [-6, -4]])], (20, 20, 20), cv2.LINE_AA)
    nc = tf([[-16, 0]])[0]
    cv2.circle(img, tuple(int(v) for v in nc), 5, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.putText(img, number, (int(nc[0]) - 3, int(nc[1]) + 3), cv2.FONT_HERSHEY_PLAIN, 0.6, (20, 20, 20), 1, cv2.LINE_AA)


def generate(seed: int = 7):
    """Return (frames, info) where info holds ground truth centres and boxes."""
    rng = np.random.default_rng(seed)
    bg = _background(rng)
    s1, s2 = schedules()
    p1 = car_positions(s1, inside_bias=10.0)
    p2 = car_positions(s2)
    h1 = np.gradient(p1, axis=0)
    h2 = np.gradient(p2, axis=0)
    noise_bank = [rng.normal(0, 2.5, (H, W, 3)).astype(np.float32) for _ in range(6)]
    frames = []
    for i in range(N_FRAMES):
        img = bg.copy()
        _draw_car(img, p2[i], math.atan2(h2[i, 1], h2[i, 0]), (200, 160, 20), (240, 240, 240), "2")
        _draw_car(img, p1[i], math.atan2(h1[i, 1], h1[i, 0]), (20, 125, 250), (110, 50, 20), "1")
        frames.append(np.clip(img.astype(np.float32) + noise_bank[i % 6], 0, 255).astype(np.uint8))

    def aabb(p, h):
        ang = math.atan2(h[1], h[0])
        w = abs(CAR_LEN * math.cos(ang)) + abs(CAR_WID * math.sin(ang))
        hh = abs(CAR_LEN * math.sin(ang)) + abs(CAR_WID * math.cos(ang))
        return [p[0] - w / 2 - 3, p[1] - hh / 2 - 3, w + 6, hh + 6]

    info = {
        "car1": p1.round(2).tolist(),
        "car2": p2.round(2).tolist(),
        "car1_box0": aabb(p1[0], h1[0]),
        "car2_box0": aabb(p2[0], h2[0]),
        "boxes": {"car1": [aabb(p1[i], h1[i]) for i in range(N_FRAMES)],
                  "car2": [aabb(p2[i], h2[i]) for i in range(N_FRAMES)]},
        "closest_frame": int(np.argmin(np.linalg.norm(p1 - p2, axis=1))),
    }
    return frames, info


if __name__ == "__main__":  # write an MP4 for use in other tools
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "motionink_demo.mp4"
    frames, _ = generate()
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    print("wrote", out)
