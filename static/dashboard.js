"use strict";
// MotionInk analytics dashboard. Talks to the existing Flask API (app.py).
// All positions are VIDEO PIXELS; speeds are px/s and turn rates °/s in image space.

const $ = (id) => document.getElementById(id);
const cvA = $("cvA"), cvB = $("cvB"), cxA = cvA.getContext("2d"), cxB = cvB.getContext("2d");
const COL = { magenta: "#e45bff", cyan: "#22c7e6", amber: "#ffb300", green: "#3ddc84", yellow: "#ffd23f", red: "#ff4757", text: "#e8f1f5", muted: "#8aa4b2", grid: "#1b3a4c" };
const RAMP = ["#2f7bff", "#22c7e6", "#3ddc84", "#ffd23f", "#ff4757"];

let st = null, frame = 0, playing = false, lastT = 0;
let shown = null;                 // last fully loaded frame image
let selecting = false, box = null, drag0 = null;
const D = resetDerived();
const edit = { i: -1, begun: false, inflight: false, pending: null };
let mouseHover = -1;

function resetDerived() {
  return { trackV: -1, targetV: -1, centers: [], speed: [], turn: [], calc: [], turnB: [], dist: [], mean: [],
           m: null, ghost: null, lenA: 0, lenB: 0, peaksA: [], peaksB: [], frameStart: 0, frameEnd: 0 };
}

const J = (u, b) => fetch(u, { method: b ? "POST" : "GET", headers: { "Content-Type": "application/json" },
  body: b ? JSON.stringify(b) : undefined }).then((r) => r.json());

// ------------------------------------------------------------------ maths helpers
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));
const hyp = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function smooth(arr, w = 5) {
  const h = Math.floor(w / 2);
  return arr.map((v, i) => {
    if (v == null) return null;
    let s = 0, c = 0;
    for (let j = i - h; j <= i + h; j++) if (arr[j] != null) { s += arr[j]; c++; }
    return s / c;
  });
}

function speedOf(pts, fps) {          // px/s between consecutive valid frames
  return smooth(pts.map((p, i) => (p && pts[i - 1] ? hyp(p, pts[i - 1]) * fps : null)));
}

function turnOf(pts, fps) {           // |heading change| in °/s
  const head = pts.map((p, i) => {
    const a = pts[i - 1], b = pts[i + 1];
    if (!p || !a || !b || hyp(a, b) < 1) return null;
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  });
  return smooth(head.map((h, i) => {
    if (h == null || head[i - 1] == null) return null;
    let d = h - head[i - 1];
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return Math.abs(d) * 180 / Math.PI * fps;
  }));
}

function peaks(arr, count = 3) {      // well-separated maxima, ignoring small wiggles
  const vals = arr.filter((v) => v != null);
  if (!vals.length) return [];
  const max = Math.max(...vals), sep = Math.max(8, Math.round(arr.length * 0.06)), out = [];
  const order = arr.map((v, i) => [v, i]).filter(([v]) => v != null && v > 0.35 * max).sort((a, b) => b[0] - a[0]);
  for (const [, i] of order) {
    if (out.every((j) => Math.abs(j - i) >= sep)) out.push(i);
    if (out.length >= count) break;
  }
  return out;
}

const pathLen = (pts) => pts.reduce((s, p, i) => s + (p && pts[i - 1] ? hyp(p, pts[i - 1]) : 0), 0);
const polyLen = (poly) => poly.reduce((s, p, i) => s + (i ? hyp(p, poly[i - 1]) : 0), 0);

function rampColor(t) {
  t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = RAMP[i].match(/\w\w/g).map((h) => parseInt(h, 16)), b = RAMP[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
function levelColor(d, th) { return d <= 0.5 * th ? COL.green : d <= th ? COL.yellow : COL.red; }

// ------------------------------------------------------------------ data refresh
async function refreshTrack() {
  const a = await J("/api/actual"), n = st.video.n, fps = st.video.fps;
  const c = new Array(n).fill(null);
  a.segs.forEach((seg, k) => seg.forEach((p, j) => { c[a.starts[k] + j] = p; }));
  D.centers = c;
  D.speed = speedOf(c, fps);
  D.turn = turnOf(c, fps);
  D.lenA = pathLen(c);
  D.peaksA = peaks(D.turn);
}

async function refreshAnalysis() {
  const a = await J("/api/analysis"), n = st.video.n, fps = st.video.fps;
  D.m = a.m || null;
  D.dist = a.dist || [];
  D.mean = a.mean_dist || [];
  D.ghost = a.endpoint_box ? [a.endpoint_box[2], a.endpoint_box[3]] : null;
  D.frameStart = a.frame_start || 0;
  D.frameEnd = a.frame_end || 0;
  const calc = new Array(n).fill(null);
  if (a.calculated) {
    let k = 0;
    D.centers.forEach((p, i) => { if (p && k < a.calculated.length) calc[i] = a.calculated[k++]; });
  }
  D.calc = calc;
  D.turnB = turnOf(calc, fps);
  D.peaksB = peaks(D.turnB);
  D.lenB = polyLen(st.target.poly || []);
}

async function poll() {
  st = await J("/api/state");
  const v = st.video;
  if (v) {
    let changed = false;
    if (st.track.v !== D.trackV) { D.trackV = st.track.v; await refreshTrack(); changed = true; }
    if (st.target.v !== D.targetV || changed) { D.targetV = st.target.v; await refreshAnalysis(); changed = true; }
    if (changed) drawCharts();
    const T = st.track, cls = T.status === "lost" || T.status === "error" ? "er" : T.status === "tracking" ? "wa" : "ok";
    if (!selecting) $("trs").innerHTML = `Tracking: <span class="${cls}">${T.status}</span> · ${T.msg}`;
  }
  const c = st.cam, g = st.gesture;
  $("gs").innerHTML = c.error ? `<span style="color:${COL.red}">${c.error}</span>`
    : c.status === "running" ? (g.hand ? `<span style="color:${COL.green}">Hand ✓</span> · pinch ${g.pinch ? `<b style="color:${COL.amber}">ACTIVE</b>` : "off"}` : "No hand detected")
    : c.status === "starting" ? "Starting camera, loading hand model…" : "Camera off";
  $("camLbl").textContent = c.status === "running" ? "live" : "";
  syncButtons();
  updateReadouts();
  drawStages();
}

// ------------------------------------------------------------------ frames + playback
function load(i) {
  if (!st?.video) return;
  frame = Math.max(0, Math.min(i, st.video.n - 1));
  const im = new Image();
  im.onload = () => { shown = im; drawStages(); };
  im.src = "/api/frame/" + frame;
  $("sl").value = frame;
  $("fr").textContent = `frame ${frame} / ${st.video.n - 1} · ${(frame / st.video.fps).toFixed(2)} s`;
  updateReadouts();
  drawCharts();
}

function setPlaying(p) { playing = p && !!st?.video; }
(function loop(t) {
  if (playing && st?.video && t - lastT > 1000 / st.video.fps) {
    lastT = t;
    if (frame >= st.video.n - 1) setPlaying(false); else load(frame + 1);
  }
  requestAnimationFrame(loop);
})(0);

// ------------------------------------------------------------------ drawing: stages
function sizeStage(cv) {
  const v = st?.video;
  if (v && (cv.width !== v.w || cv.height !== v.h)) { cv.width = v.w; cv.height = v.h; }
}
const kOf = (cv) => cv.width / (cv.getBoundingClientRect().width || cv.width);   // canvas px per CSS px

function line(cx, a, b, color, w, dash) {
  cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]);
  cx.strokeStyle = color; cx.lineWidth = w; cx.setLineDash(dash || []); cx.stroke(); cx.setLineDash([]);
}
function polyline(cx, pts, color, w, dash) {
  if (pts.length < 2) return;
  cx.beginPath(); pts.forEach((p, i) => (i ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])));
  cx.strokeStyle = color; cx.lineWidth = w; cx.setLineDash(dash || []); cx.stroke(); cx.setLineDash([]);
}
function star(cx, p, r, color) {
  cx.beginPath();
  for (let k = 0; k < 10; k++) {
    const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r;
    cx.lineTo(p[0] + rr * Math.cos(a), p[1] + rr * Math.sin(a));
  }
  cx.closePath(); cx.fillStyle = color; cx.fill(); cx.lineWidth = 1.5; cx.strokeStyle = "#04202a"; cx.stroke();
}
function tag(cx, text, p, color, k) {
  cx.font = `600 ${13 * k}px "Barlow Condensed", "Arial Narrow", sans-serif`;
  const w = cx.measureText(text).width + 10 * k, h = 18 * k;
  const x = Math.min(Math.max(2, p[0] - w / 2), cx.canvas.width - w - 2), y = Math.max(2, p[1] - h - 6 * k);
  cx.fillStyle = "rgba(4,16,22,0.85)"; cx.fillRect(x, y, w, h);
  cx.fillStyle = color; cx.fillRect(x, y, 3 * k, h);
  cx.fillStyle = COL.text; cx.textBaseline = "middle"; cx.fillText(text, x + 6 * k, y + h / 2 + k);
}
function background(cx, cv, dim) {
  cx.fillStyle = "#050c11"; cx.fillRect(0, 0, cv.width, cv.height);
  if (shown) cx.drawImage(shown, 0, 0, cv.width, cv.height);
  if (dim) { cx.fillStyle = `rgba(3,10,14,${dim})`; cx.fillRect(0, 0, cv.width, cv.height); }
}

function drawStageA() {
  sizeStage(cvA);
  const cx = cxA, k = kOf(cvA), c = D.centers;
  background(cx, cvA, 0.15);
  const vals = D.speed.filter((v) => v != null), lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
  for (let i = 1; i < c.length; i++) {          // halo, then speed-coloured segments (gaps never bridged)
    if (c[i] && c[i - 1]) line(cx, c[i - 1], c[i], "rgba(0,0,0,0.6)", 7 * k);
  }
  for (let i = 1; i < c.length; i++) {
    if (c[i] && c[i - 1]) line(cx, c[i - 1], c[i], rampColor(hi > lo ? ((D.speed[i] ?? lo) - lo) / (hi - lo) : 0.5), 4 * k);
  }
  D.peaksA.forEach((i) => c[i] && star(cx, c[i], 9 * k, COL.yellow));
  const p = c[frame];
  if (p && D.ghost) {
    const [w, h] = D.ghost;
    cx.strokeStyle = COL.text; cx.lineWidth = 2 * k; cx.strokeRect(p[0] - w / 2, p[1] - h / 2, w, h);
    tag(cx, `Tracked car · ${fmt(D.speed[frame])} px/s`, [p[0], p[1] - h / 2], COL.cyan, k);
  }
  if (box) { cx.setLineDash([6 * k, 4 * k]); cx.strokeStyle = COL.amber; cx.lineWidth = 2 * k; cx.strokeRect(...box); cx.setLineDash([]); }
}

function drawStageB() {
  sizeStage(cvB);
  const cx = cxB, k = kOf(cvB), t = st?.target;
  background(cx, cvB, 0.3);
  const c = D.centers;
  for (let i = 1; i < c.length; i++) if (c[i] && c[i - 1]) line(cx, c[i - 1], c[i], "rgba(232,241,245,0.35)", 1.5 * k, [3 * k, 4 * k]);
  if (!t) return;
  if ($("dv").checked && D.m) {
    for (let i = 1; i < D.calc.length; i++) {
      const a = D.calc[i - 1], b = D.calc[i];
      if (a && b) { line(cx, a, b, "rgba(0,0,0,0.6)", 8 * k); }
    }
    for (let i = 1; i < D.calc.length; i++) {
      const a = D.calc[i - 1], b = D.calc[i];
      if (a && b && D.dist[i] != null) line(cx, a, b, levelColor(D.dist[i], D.m.thresh), 5 * k);
    }
  }
  if ($("tg").checked && t.poly.length > 1) {
    polyline(cx, t.poly, COL.cyan, 2.5 * k, [9 * k, 6 * k]);
    const sel = edit.i >= 0 ? edit.i : t.sel, hov = mouseHover >= 0 ? mouseHover : t.hover;
    t.cp.forEach((p, i) => {
      const s = i === sel, h = i === hov;
      cx.beginPath(); cx.arc(p[0], p[1], (s ? 11 : h ? 9 : 6) * k, 0, 7);
      cx.fillStyle = s ? COL.amber : COL.cyan; cx.fill();
      if (s || h) { cx.strokeStyle = "#fff"; cx.lineWidth = 2 * k; cx.stroke(); }
    });
  }
  D.peaksB.forEach((i) => D.calc[i] && star(cx, D.calc[i], 9 * k, COL.yellow));
  const g = D.calc[frame];
  if (g && D.ghost) {
    const [w, h] = D.ghost;
    cx.save(); cx.setLineDash([6 * k, 4 * k]); cx.strokeStyle = COL.amber; cx.fillStyle = "rgba(255,179,0,0.16)"; cx.lineWidth = 2.5 * k;
    cx.fillRect(g[0] - w / 2, g[1] - h / 2, w, h); cx.strokeRect(g[0] - w / 2, g[1] - h / 2, w, h); cx.restore();
    tag(cx, `Edited line · ${fmt(D.dist[frame], 1)} px from A`, [g[0], g[1] - h / 2], COL.amber, k);
  }
  if (st.cursor) {
    const [x, y] = st.cursor, m = t.mode;
    cx.beginPath(); cx.arc(x, y, (m === "drag" ? 16 : 12) * k, 0, 7);
    cx.strokeStyle = m === "drag" ? COL.amber : m === "hover" ? "#fff" : COL.cyan; cx.lineWidth = (m === "drag" ? 5 : 2) * k; cx.stroke();
    const lbl = { drag: "DRAGGING", hover: "HOVER", pinch_empty: "PINCH (no point near)" }[m];
    if (lbl) { cx.fillStyle = "#fff"; cx.font = `${13 * k}px sans-serif`; cx.fillText(lbl, x + 20 * k, y - 12 * k); }
  }
}

function drawStages() {
  const has = !!st?.video;
  $("emptyA").hidden = has;
  $("emptyB").hidden = has && D.m != null;
  if (!has) return;
  drawStageA();
  drawStageB();
}

// ------------------------------------------------------------------ drawing: charts
function drawChart(cv, series, { unit = "", marker = null, emptyText = "No data yet" } = {}) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const L = 36, R = 8, T = 10, B = 20, n = st?.video?.n || 0;
  const vals = series.flatMap((s) => s.data.filter((v) => v != null && Number.isFinite(v)));
  g.font = "10px Barlow, sans-serif";
  if (!vals.length || n < 2) { g.fillStyle = COL.muted; g.textAlign = "center"; g.fillText(emptyText, W / 2, H / 2); return; }
  const max = Math.max(...vals) * 1.12 || 1;
  const X = (i) => L + (i / (n - 1)) * (W - L - R), Y = (v) => T + (1 - v / max) * (H - T - B);
  g.strokeStyle = COL.grid; g.fillStyle = COL.muted; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const y = T + (H - T - B) * k / 4;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.textAlign = "right"; g.fillText(fmt(max * (4 - k) / 4), L - 4, y + 3);
  }
  g.textAlign = "center";
  for (let k = 0; k <= 4; k++) g.fillText(Math.round((n - 1) * k / 4), X((n - 1) * k / 4), H - 5);
  g.save(); g.translate(9, (H - B) / 2 + T / 2); g.rotate(-Math.PI / 2); g.fillText(unit, 0, 0); g.restore();
  for (const s of series) {
    g.beginPath(); let on = false;
    s.data.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) { on = false; return; }
      on ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); on = true;
    });
    if (s.gradient) {                      // colour by value, same min→max ramp as panel A
      const sv = s.data.filter((v) => v != null), lo = Math.min(...sv), hi = Math.max(...sv);
      const grad = g.createLinearGradient(0, Y(lo), 0, Y(hi) - 0.01);
      RAMP.forEach((c, k) => grad.addColorStop(k / (RAMP.length - 1), c));
      g.strokeStyle = grad;
    } else g.strokeStyle = s.color;
    g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = "rgba(232,241,245,0.6)"; g.lineWidth = 1;   // playhead
  g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), H - B); g.stroke();
  if (marker && marker.i != null && marker.v != null) {
    g.fillStyle = "#fff"; g.beginPath(); g.arc(X(marker.i), Y(marker.v), 3.5, 0, 7); g.fill();
    g.fillStyle = COL.text; g.textAlign = X(marker.i) > W - 80 ? "right" : "left";
    g.font = "600 11px Barlow, sans-serif"; g.fillText(marker.label, X(marker.i) + (g.textAlign === "left" ? 6 : -6), Y(marker.v) - 6);
  }
}

function argmax(arr) { let b = -1, bv = -Infinity; arr.forEach((v, i) => { if (v != null && v > bv) { bv = v; b = i; } }); return b; }

function drawCharts() {
  const pk = argmax(D.speed);
  drawChart($("chSpeed"), [{ data: D.speed, gradient: true, width: 2.5 }], { unit: "px/s",
    marker: pk >= 0 ? { i: pk, v: D.speed[pk], label: `${fmt(D.speed[pk])} px/s peak` } : null, emptyText: "Track a car to see speed" });
  drawChart($("chTurn"), [{ data: D.turn, color: COL.magenta, width: 2 }, { data: D.turnB, color: COL.yellow, width: 1.5, dash: [5, 4] }],
    { unit: "°/s", emptyText: "Track a car to see turn rate" });
  drawChart($("chDist"), [{ data: D.dist, color: COL.cyan, width: 2 }, { data: D.mean, color: COL.amber, width: 2 }],
    { unit: "px", emptyText: "Needs a track and an edited line" });
  $("peakLbl").textContent = pk >= 0 ? `peak ${fmt(D.speed[pk])} px/s` : "";
  const tA = D.turn.filter((v) => v != null), tB = D.turnB.filter((v) => v != null);
  $("turnLbl").textContent = tA.length ? `max ${fmt(Math.max(...tA))} / ${tB.length ? fmt(Math.max(...tB)) : "—"} °/s` : "";
  $("distLbl").textContent = D.m ? `mean ${fmt(D.m.mean_px, 1)} px` : "";
}

// ------------------------------------------------------------------ readouts
function bar(id, frac, color) { const el = $(id); el.style.width = `${Math.max(0, Math.min(1, frac || 0)) * 100}%`; el.style.background = color; }

function updateReadouts() {
  const v = st?.video;
  if (!v) return;
  const spd = D.speed[frame], vals = D.speed.filter((x) => x != null), pk = vals.length ? Math.max(...vals) : null;
  const trn = D.turn[frame], tv = D.turn.filter((x) => x != null), tmax = tv.length ? Math.max(...tv) : null;
  const tracked = D.centers.filter(Boolean).length;
  $("vSpd").innerHTML = spd == null ? "—" : `<b>${fmt(spd)}</b> px/s`;
  bar("barSpd", pk ? spd / pk : 0, rampColor(pk ? (spd || 0) / pk : 0));
  $("vPeak").innerHTML = pk == null ? "—" : `<b>${fmt(pk)}</b> px/s`;
  bar("barPeak", pk ? 1 : 0, COL.red);
  $("vTurn").innerHTML = trn == null ? "—" : `<b>${fmt(trn)}</b> °/s`;
  bar("barTurn", tmax ? trn / tmax : 0, COL.magenta);
  $("vLenA").innerHTML = D.lenA ? `<b>${fmt(D.lenA)}</b> px` : "—";
  bar("barLenA", D.lenA && D.lenB ? D.lenA / Math.max(D.lenA, D.lenB) : D.lenA ? 1 : 0, COL.cyan);
  $("vTrk").innerHTML = tracked ? `<b>${tracked}</b> / ${v.n} frames` : "—";
  bar("barTrk", tracked / v.n, COL.green);

  const m = D.m, d = D.dist[frame], maxD = m ? m.max_px : null;
  $("vDist").innerHTML = d == null ? "—" : `<b>${fmt(d, 1)}</b> px`;
  bar("barDist", maxD ? d / maxD : 0, m && d != null ? levelColor(d, m.thresh) : COL.green);
  $("vMean").innerHTML = m ? `<b>${fmt(m.mean_px, 1)}</b> px` : "—";
  bar("barMean", m && maxD ? m.mean_px / maxD : 0, m ? levelColor(m.mean_px, m.thresh) : COL.green);
  $("vMax").innerHTML = m ? `<b>${fmt(m.max_px, 1)}</b> px` : "—";
  bar("barMax", m ? 1 : 0, m ? levelColor(m.max_px, m.thresh) : COL.green);
  $("vLenB").innerHTML = D.lenB && m ? `<b>${fmt(D.lenB)}</b> px` : "—";
  bar("barLenB", D.lenA && D.lenB ? D.lenB / Math.max(D.lenA, D.lenB) : 0, COL.amber);
  const npts = st.target?.cp?.length || 0;
  $("vPts").innerHTML = npts ? `<b>${npts}</b> points` : "—";
  bar("barPts", npts ? 1 : 0, COL.cyan);

  const T = st.track;
  const rows = [
    ["Mean deviation", m ? `${fmt(m.mean_px, 1)} px` : null],
    ["Mean, % of frame", m ? `${fmt(m.mean_pct, 1)}%` : null],
    ["Max deviation", m ? `${fmt(m.max_px, 1)} px` : null],
    [`Frames > ${m ? fmt(m.thresh, 0) : "—"} px`, m ? `${fmt(m.frac * 100)}%` : null],
    ["Length A / B", D.lenA && m ? `${fmt(D.lenA)} / ${fmt(D.lenB)} px` : null],
    ["Length Δ", D.lenA && m ? `${D.lenB - D.lenA >= 0 ? "+" : ""}${fmt(D.lenB - D.lenA)} px` : null],
    ["Tracking", T ? T.status : null],
    ["Tracked frames", tracked ? `${D.frameStart}–${D.frameEnd}` : null],
  ];
  $("stats").innerHTML = rows.map(([k, val], i) =>
    `<dt class="${k === "Length Δ" ? "hl" : ""}">${k}</dt><dd class="${val == null ? "na" : ""}${k === "Length Δ" ? " hl" : ""}">${val ?? "—"}</dd>`).join("");
}

// ------------------------------------------------------------------ buttons
function syncButtons() {
  const has = !!st?.video, tracking = st?.track?.status === "tracking", hasT = (st?.target?.cp?.length || 0) > 0;
  $("selbtn").disabled = !has || tracking;
  $("trk").disabled = !has || tracking || !box;
  $("stp").disabled = !tracking;
  for (const id of ["undo", "rst"]) $(id).disabled = !hasT;
  $("rebuild").disabled = !has || !D.centers.some(Boolean);
  for (const id of ["play", "pause", "restart", "all"]) $(id).disabled = !has;
  $("play").classList.toggle("on", playing);
}

$("file").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("video", f);
  $("trs").textContent = `Uploading and decoding ${f.name}…`;
  const r = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  e.target.value = "";
  if (r.error) { $("trs").innerHTML = `<span class="er">${r.error}</span>`; return; }
  Object.assign(D, resetDerived()); box = null; shown = null; setPlaying(false);
  await poll();
  $("sl").max = st.video.n - 1;
  load(0);
};

async function loadDemo() {
  $("trs").textContent = "Building the synthetic demo clip and tracking Car 1 (orange)…";
  const r = await J("/api/demo", {});
  if (r.error) { $("trs").innerHTML = `<span class="er">${r.error}</span>`; return; }
  Object.assign(D, resetDerived()); box = null; shown = null; setPlaying(false);
  await poll();
  $("sl").max = st.video.n - 1;
  load(0);
}
$("demo").onclick = loadDemo;
$("demo2").onclick = loadDemo;

$("selbtn").onclick = () => {
  selecting = true; setPlaying(false); box = null;
  $("stageA").classList.add("selecting");
  $("trs").textContent = "Drag a box tightly around the car on the left video, then press Track.";
};
$("trk").onclick = async () => {
  if (!box) return;
  const r = await J("/api/track", { frame, box });
  if (r.error) $("trs").innerHTML = `<span class="er">${r.error}</span>`;
  else box = null;
  selecting = false; $("stageA").classList.remove("selecting");
};
$("stp").onclick = () => J("/api/stop_track", {});
$("undo").onclick = () => J("/api/target/undo", {});
$("rst").onclick = () => J("/api/target/reset", {});
$("rebuild").onclick = () => J("/api/target/rebuild", {});
$("all").onclick = async () => { await J("/api/reset", {}); box = null; Object.assign(D, resetDerived()); setPlaying(false); load(0); };
$("play").onclick = () => setPlaying(true);
$("pause").onclick = () => setPlaying(false);
$("restart").onclick = () => { setPlaying(false); load(0); };
$("sl").oninput = (e) => { setPlaying(false); load(+e.target.value); };
$("tg").onchange = $("dv").onchange = () => drawStages();

let camOn = false;
$("camb").onclick = async () => {
  camOn = !camOn;
  await J("/api/camera", { on: camOn });
  $("camb").textContent = camOn ? "Stop camera" : "Start camera";
  if (camOn) { $("cam").src = "/api/webcam"; $("cam").hidden = false; $("camHint").hidden = true; }
  else { $("cam").removeAttribute("src"); $("cam").hidden = true; $("camHint").hidden = false; }
};
$("th").oninput = (e) => { $("thOut").textContent = (+e.target.value).toFixed(2); J("/api/config", { enter: +e.target.value, exit: +e.target.value + 0.2 }); };
$("mg").oninput = (e) => { $("mgOut").textContent = (+e.target.value).toFixed(2); J("/api/config", { margin: +e.target.value }); };

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select") || !st?.video) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "ArrowRight") { setPlaying(false); load(frame + (e.shiftKey ? 10 : 1)); }
  else if (e.key === "ArrowLeft") { setPlaying(false); load(frame - (e.shiftKey ? 10 : 1)); }
});

// ------------------------------------------------------------------ pointer: select (A) and edit (B)
function pos(cv, e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; }

cvA.addEventListener("pointerdown", (e) => { if (selecting) { drag0 = pos(cvA, e); cvA.setPointerCapture(e.pointerId); } });
cvA.addEventListener("pointermove", (e) => {
  if (!drag0) return;
  const p = pos(cvA, e);
  box = [Math.min(drag0[0], p[0]), Math.min(drag0[1], p[1]), Math.abs(p[0] - drag0[0]), Math.abs(p[1] - drag0[1])];
  drawStages();
});
cvA.addEventListener("pointerup", () => {
  if (!drag0) return;
  drag0 = null;
  if (box && (box[2] < 4 || box[3] < 4)) { box = null; $("trs").innerHTML = `<span class="er">That box is too small. Drag around the whole car.</span>`; }
  else if (box) $("trs").textContent = "Box set. Press 2 · Track from this frame.";
  syncButtons();
});

function nearestCp(p) {
  const cp = st?.target?.cp || [], r = 14 * kOf(cvB);
  let best = -1, bd = Infinity;
  cp.forEach((q, i) => { const d = hyp(p, q); if (d < bd) { bd = d; best = i; } });
  return bd <= r ? best : -1;
}
async function sendMove(xy) {
  if (edit.inflight) { edit.pending = xy; return; }
  edit.inflight = true;
  await J("/api/target/move", { i: edit.i, xy, begin: !edit.begun }).catch(() => {});
  edit.begun = true; edit.inflight = false;
  if (edit.pending) { const p = edit.pending; edit.pending = null; await sendMove(p); }
}
cvB.addEventListener("pointerdown", (e) => {
  if (!$("tg").checked) return;
  const i = nearestCp(pos(cvB, e));
  if (i < 0) return;
  edit.i = i; edit.begun = false; setPlaying(false);
  cvB.setPointerCapture(e.pointerId); cvB.classList.add("grabbing");
});
cvB.addEventListener("pointermove", (e) => {
  const p = pos(cvB, e);
  if (edit.i >= 0) {
    st.target.cp[edit.i] = p;            // instant feedback; server confirms on next poll
    drawStages(); sendMove(p); return;
  }
  const h = nearestCp(p);
  if (h !== mouseHover) { mouseHover = h; cvB.classList.toggle("grab", h >= 0); drawStages(); }
});
async function endEdit() {
  if (edit.i < 0) return;
  while (edit.inflight) await new Promise((r) => setTimeout(r, 15));
  edit.i = -1; cvB.classList.remove("grabbing");
}
cvB.addEventListener("pointerup", endEdit);
cvB.addEventListener("pointercancel", endEdit);

window.addEventListener("resize", () => { drawCharts(); drawStages(); });
window.MI = { get st() { return st; }, D, load, get frame() { return frame; } };
setInterval(() => poll().catch(() => {}), 50);
poll().catch(() => {});
