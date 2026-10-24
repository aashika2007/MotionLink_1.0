"""Target-path editing and path comparison. All coordinates are video pixels."""
import numpy as np

def catmull_rom(pts, n=10):
    p = np.asarray(pts, float)
    if len(p) < 3: return p
    P = np.vstack([p[0], p, p[-1]]); t = np.linspace(0, 1, n, endpoint=False)[:, None]; out = []
    for i in range(1, len(P) - 2):
        a, b, c, d = P[i-1], P[i], P[i+1], P[i+2]
        out.append(0.5*(2*b + (-a+c)*t + (2*a-5*b+4*c-d)*t**2 + (-a+3*b-3*c+d)*t**3))
    out.append(p[-1:]); return np.vstack(out)

def dist_to_polyline(pts, poly):
    pts = np.asarray(pts, float); poly = np.asarray(poly, float)
    if len(poly) == 1: return np.linalg.norm(pts - poly[0], axis=1)
    a, b = poly[:-1], poly[1:]; ab = b - a; L = np.maximum((ab**2).sum(1), 1e-9)
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(2) / L[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None] - proj, axis=2).min(1)

def segments(centers):
    """Contiguous runs of valid (non-NaN) frames -> [(start, end_inclusive)]. Gaps are never bridged."""
    ok = ~np.isnan(centers[:, 0]); runs = []; s = None
    for i, v in enumerate(ok):
        if v and s is None: s = i
        if not v and s is not None: runs.append((s, i - 1)); s = None
    if s is not None: runs.append((s, len(ok) - 1))
    return runs

class TargetPath:
    def __init__(self): self.cp = np.zeros((0, 2)); self.orig = self.cp.copy(); self.undo_stack = []; self.version = 0
    def init_from(self, centers, n_points=12):
        v = centers[~np.isnan(centers[:, 0])]
        if len(v) < 2: return False
        idx = np.unique(np.linspace(0, len(v) - 1, min(n_points, len(v))).round().astype(int))
        self.cp = v[idx].copy(); self.orig = self.cp.copy(); self.undo_stack = []; self.version += 1; return True
    def nearest(self, xy, radius):
        if not len(self.cp): return None
        d = np.linalg.norm(self.cp - np.asarray(xy), axis=1); i = int(d.argmin())
        return i if d[i] <= radius else None
    def begin_drag(self, i): self.undo_stack.append(self.cp.copy())
    def move(self, i, xy): self.cp[i] = xy; self.version += 1
    def undo(self):
        if self.undo_stack: self.cp = self.undo_stack.pop(); self.version += 1
    def reset(self): self.cp = self.orig.copy(); self.undo_stack = []; self.version += 1
    def clear(self): self.cp = np.zeros((0, 2)); self.orig = self.cp.copy(); self.undo_stack = []; self.version += 1
    def polyline(self): return catmull_rom(self.cp) if len(self.cp) >= 2 else self.cp

def compare(centers, poly, diag, thresh_frac=0.03):
    """Compare recorded frames with target positions at matching path progress."""
    ok = ~np.isnan(centers[:, 0])
    if ok.sum() < 2 or len(poly) < 2: return None
    # Resample the editable path by arc length, so every tracked frame has a
    # corresponding target position. This captures timing/order differences;
    # nearest-point distance alone can make a reversed or delayed path look good.
    lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-9: return None
    # Pace the target by how far the car has actually travelled (fraction of the recorded
    # path length), not by frame count: a car that brakes for a corner must not count as
    # "deviating" from an identical line. Order still matters, so a reversed path still fails.
    rec = centers[ok]
    rec_cum = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rec, axis=0), axis=1))]
    frac = rec_cum / rec_cum[-1] if rec_cum[-1] > 1e-9 else np.linspace(0., 1., len(rec))
    progress = frac * cumulative[-1]
    target = np.column_stack([np.interp(progress, cumulative, poly[:, k]) for k in (0, 1)])
    d = np.full(len(centers), np.nan); d[ok] = np.linalg.norm(centers[ok] - target, axis=1)
    th = thresh_frac * diag
    return dict(dist=d, thresh=th, mean_px=float(np.nanmean(d)), max_px=float(np.nanmax(d)),
                mean_pct_diag=float(np.nanmean(d) / diag * 100), frac_diverging=float(np.mean(d[ok] > th)),
                target=target)
