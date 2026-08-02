"""Sensor-realistic domain randomization by object injection.

Manufactures training positives by pasting **real** object event crops (harvested
from labeled GT windows) onto **real** background-only windows, along a
randomized 2-D trajectory.  This replaces pure transform augmentation
(flip/drop/scale in :mod:`orbitsight.augment`), which can only perturb an
existing window and can never create a new (object x background x motion)
combination — exactly the combinations the dim DVX/Thuraya3 + star-field Stars3
regimes are starved of.

Design constraints that shape this file
--------------------------------------
1. **Real crops, not synthetic blobs.**  We cut object events (`label>0`, or
   events inside the GT box) from real GT windows so the injected object keeps
   in-distribution event micro-statistics.  Fully synthetic events would teach
   the detector artifacts.
2. **Trajectory-consistent across the whole context span.**  The temporal-context
   CenterNet (`--context N`) and the track-before-detect linker read +/-N
   neighboring windows.  An object injected into a single window would *teleport*
   between context frames and actively poison those models, so we emit the crop
   at the trajectory position of **every** window in the span, not just the
   center one.  The GT box is the object's position at the center window.
3. **Normalized coordinates.**  The training voxelizer runs with width=height=1
   and x,y in [0,1] (sensor-agnostic).  The background events handed in are
   already normalized; per-sensor pixel geometry enters only through the crop
   library and the `sensor` argument, and object/hot/clutter events are
   normalized before they are appended.

Randomization axes (each maps to a real domain-shift lever):
  velocity / acceleration   -> Thuraya3 track linking      (grounded in measured px/window)
  event-rate (magnitude)    -> dim-object recall           (subsample the crop toward 2-5 ev/window)
  polarity imbalance        -> ON/OFF-biased objects
  hot pixels                -> sensor artifact / hard negative
  correlated bg bursts      -> clutter hard negative (star-field FPs)
  telescope (platform) jitter -> coherent scene motion -> star-rejection cue
  temporal gaps             -> missing-window coasting regime (Kalman coast)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Crop library
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    """One real object appearance, centered at the origin in pixels."""

    dx: np.ndarray      # x offset from box center, pixels
    dy: np.ndarray      # y offset from box center, pixels
    rel: np.ndarray     # event time within source window, [0, 1]
    pol: np.ndarray     # polarity in {0, 1}, int8
    w_px: float         # source GT box width  (px)
    h_px: float         # source GT box height (px)


class CropLibrary:
    """Per-sensor bank of real object crops + measured motion statistics.

    Built once from a loaded ``WindowSet`` (its events + item list), so no extra
    on-disk artifact is needed — the training process already holds the events.
    """

    def __init__(self):
        self.by_sensor: dict[str, list[Crop]] = {}
        self.vel_px90: dict[str, float] = {}     # 90th-pct object speed (px/window)
        self.med_events: dict[str, int] = {}     # median object event count

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_windowset(cls, events, items, sensors, *, window_us,
                       min_events=6, margin_px=2.0, max_per_sensor=6000):
        """Harvest crops from every positive window in a loaded set.

        ``items`` are ``(seq, lo, hi, ws, we, box)`` tuples (box normalized or
        None); ``events[seq]`` is an :class:`orbitsight.data.Events`.
        """
        lib = cls()
        speeds: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        # per-sequence ordered GT centers -> consecutive-window displacement
        gt_track: dict[str, list[tuple[int, float, float]]] = {}

        for seq, lo, hi, ws, we, box in items:
            if box is None:
                continue
            sn = sensors[seq]
            ev = events[seq]
            cx, cy, bw, bh = box
            cx_px, cy_px = cx * sn.width, cy * sn.height
            hw = bw * sn.width * 0.5 + margin_px
            hh = bh * sn.height * 0.5 + margin_px
            x = ev.x[lo:hi]
            y = ev.y[lo:hi]
            m = (np.abs(x - cx_px) <= hw) & (np.abs(y - cy_px) <= hh)
            # purify with per-event labels when they carry object marks
            lab = ev.label[lo:hi]
            if lab.any():
                m &= lab > 0
            if int(m.sum()) < min_events:
                continue
            t = ev.t[lo:hi]
            dur = max(int(we - ws), 1)
            crop = Crop(
                dx=(x[m] - cx_px).astype(np.float32),
                dy=(y[m] - cy_px).astype(np.float32),
                rel=np.clip((t[m] - ws).astype(np.float64) / dur, 0.0, 1.0),
                pol=(ev.pol[lo:hi][m] > 0).astype(np.int8),
                w_px=float(bw * sn.width),
                h_px=float(bh * sn.height),
            )
            lib.by_sensor.setdefault(sn.name, []).append(crop)
            counts.setdefault(sn.name, []).append(int(m.sum()))
            gt_track.setdefault(seq, []).append((int(ws), cx_px, cy_px))

        # measured object speed (px per 40 ms window) from consecutive GT windows
        for seq, pts in gt_track.items():
            sn = sensors[seq]
            pts.sort(key=lambda p: p[0])
            for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
                gap = round((t1 - t0) / window_us)
                if gap >= 1:
                    speeds.setdefault(sn.name, []).append(
                        float(np.hypot(x1 - x0, y1 - y0) / gap))

        for name, arr in speeds.items():
            lib.vel_px90[name] = float(np.percentile(arr, 90)) if arr else 0.0
        for name, arr in counts.items():
            lib.med_events[name] = int(np.median(arr)) if arr else 0
        # cap library size per sensor (keeps a diverse but bounded bank)
        for name, crops in lib.by_sensor.items():
            if len(crops) > max_per_sensor:
                idx = np.linspace(0, len(crops) - 1, max_per_sensor).astype(int)
                lib.by_sensor[name] = [crops[i] for i in idx]
        return lib

    # -- sampling ----------------------------------------------------------- #
    def sample(self, sensor_name: str, rng: np.random.Generator) -> Optional[Crop]:
        crops = self.by_sensor.get(sensor_name)
        if not crops:
            return None
        return crops[rng.integers(len(crops))]

    def summary(self) -> str:
        parts = []
        for name in sorted(self.by_sensor):
            parts.append(
                f"{name}: {len(self.by_sensor[name])} crops, "
                f"med_ev={self.med_events.get(name, 0)}, "
                f"v90={self.vel_px90.get(name, 0.0):.1f}px/win")
        return " | ".join(parts) if parts else "(empty)"


# --------------------------------------------------------------------------- #
#  Injection config
# --------------------------------------------------------------------------- #
@dataclass
class InjectCfg:
    p_inject: float = 0.5        # P(convert a background window -> synthetic positive)
    v_px_max: float = 0.0        # max speed px/window; <=0 -> use measured v90
    v_px_floor: float = 2.0      # lower bound used if a sensor has no measured motion
    a_px_max: float = 1.0        # max |acceleration| px/window^2
    mag_lo: float = 0.15         # event-rate scale (fraction of crop kept), low
    mag_hi: float = 1.30         # ... high
    p_pol_imbalance: float = 0.4  # P(bias object polarity toward one sign)
    pol_bias_max: float = 0.85   # max fraction forced to the dominant polarity
    p_hot: float = 0.5           # P(inject hot pixels)
    hot_max: int = 6
    hot_rate_lo: int = 4
    hot_rate_hi: int = 30
    p_clutter: float = 0.5       # P(inject correlated background bursts)
    clutter_max: int = 3
    clutter_ev_lo: int = 8
    clutter_ev_hi: int = 40
    clutter_sigma_px: float = 3.0
    jitter_px: float = 1.0       # telescope platform jitter amplitude (px, per-window RW)
    p_gap: float = 0.35          # P(object absent in a non-center window) -> temporal gap
    size_jitter: float = 0.20    # +/- fractional crop/box size randomization
    margin_frac: float = 0.06    # keep trajectory this far (frac of frame) from edges
    min_center_events: int = 0   # if >0: relabel as BACKGROUND (no box) any synthetic
                                 # whose center window has < this many events -> never
                                 # train a positive heatmap for an effectively-absent obj
    v_px_min_frac: float = 0.0   # speed floor as a fraction of v_cap (avoid near-static
                                 # synthetics that look like hot pixels / stars)

    @classmethod
    def conservative(cls, p_inject=0.15):
        """Regularize (not replace) the real prior: fewer synthetics, realistic
        magnitude/motion, mild nuisances, acceleration OFF, min-event guard. Use this
        after the broad default was falsified (Thuraya3 injection A/B, -0.17)."""
        return cls(p_inject=p_inject, a_px_max=0.0, mag_lo=0.40, mag_hi=1.10,
                   p_gap=0.10, p_hot=0.10, p_clutter=0.15, p_pol_imbalance=0.15,
                   min_center_events=3, v_px_min_frac=0.35)


# --------------------------------------------------------------------------- #
#  Injection
# --------------------------------------------------------------------------- #
def inject_window(xn, yn, pol, t, sensor, ws, we, ws0, we0, window_us,
                  lib: CropLibrary, rng: np.random.Generator,
                  cfg: InjectCfg = InjectCfg()):
    """Paste a moving object + sensor nuisances onto a background window.

    Parameters mirror the training call site: ``xn, yn`` are **normalized**
    background event coords in [0, 1]; ``pol`` polarity; ``t`` timestamps (us);
    ``[ws, we]`` the full (context-widened) span; ``[ws0, we0]`` the center
    window whose GT box is produced.  Returns ``(xn, yn, pol, t, box)`` with the
    same normalization and ``box`` = normalized ``(cx, cy, w, h)`` or None.
    """
    crop = lib.sample(sensor.name, rng)
    W, H = float(sensor.width), float(sensor.height)
    xn = np.asarray(xn, dtype=np.float64)
    yn = np.asarray(yn, dtype=np.float64)
    pol = np.asarray(pol, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    if crop is None:
        return xn, yn, pol, t, None

    wu = int(window_us)
    starts = np.arange(int(ws), int(we), wu, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([int(ws0)], dtype=np.int64)
    n = starts.size
    center_mid = 0.5 * (ws0 + we0)
    # index of the window that owns the center window (never gapped, defines box)
    c_idx = int(np.clip(np.searchsorted(starts, ws0, side="right") - 1, 0, n - 1))

    # --- telescope (platform) jitter: coherent per-window random walk -------- #
    if cfg.jitter_px > 0:
        jx = np.cumsum(rng.normal(0, cfg.jitter_px, n))
        jy = np.cumsum(rng.normal(0, cfg.jitter_px, n))
        jx -= jx.mean(); jy -= jy.mean()
    else:
        jx = np.zeros(n); jy = np.zeros(n)
    # shift the real background too — the whole focal plane jitters
    if cfg.jitter_px > 0 and xn.size:
        ki = np.clip(((t - ws) // wu).astype(np.int64), 0, n - 1)
        xn = xn + jx[ki] / W
        yn = yn + jy[ki] / H

    # --- trajectory (grounded velocity, physical acceleration) --------------- #
    v_cap = cfg.v_px_max if cfg.v_px_max > 0 else max(
        lib.vel_px90.get(sensor.name, 0.0), cfg.v_px_floor)
    ang = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(cfg.v_px_min_frac * v_cap, v_cap)   # floor avoids near-static
    vx, vy = speed * np.cos(ang), speed * np.sin(ang)
    ax = rng.uniform(-cfg.a_px_max, cfg.a_px_max)
    ay = rng.uniform(-cfg.a_px_max, cfg.a_px_max)
    sf = 1.0 + cfg.size_jitter * rng.uniform(-1, 1)      # size scale
    w_px = max(crop.w_px * sf, 2.0)
    h_px = max(crop.h_px * sf, 2.0)

    # place the trajectory so the object stays on-frame across the whole span
    mgx, mgy = cfg.margin_frac * W, cfg.margin_frac * H
    tau = (starts + wu / 2 - center_mid) / wu             # window offset, in windows
    ox = vx * tau + 0.5 * ax * tau ** 2                   # per-window x offset from p0
    oy = vy * tau + 0.5 * ay * tau ** 2
    p0x = rng.uniform(mgx - ox.min(), W - mgx - ox.max()) if ox.max() > ox.min() \
        else rng.uniform(mgx, W - mgx)
    p0y = rng.uniform(mgy - oy.min(), H - mgy - oy.max()) if oy.max() > oy.min() \
        else rng.uniform(mgy, H - mgy)
    cxk = p0x + ox + jx                                   # object center per window (px)
    cyk = p0y + oy + jy

    # --- emit the crop at each window (magnitude scaling + temporal gaps) ----- #
    ox_list, oy_list, op_list, ot_list = [], [], [], []
    n_crop = crop.dx.size
    n_center = 0                                           # events at the center window
    pol_bias = None
    if rng.random() < cfg.p_pol_imbalance:
        pol_bias = (int(rng.integers(2)),
                    float(rng.uniform(0.5, cfg.pol_bias_max)))
    for k in range(n):
        if k != c_idx and rng.random() < cfg.p_gap:
            continue                                      # temporal gap
        mag = rng.uniform(cfg.mag_lo, cfg.mag_hi)
        keep = rng.random(n_crop) < mag
        if not keep.any():
            if k == c_idx:                                # center must emit >=1
                keep[rng.integers(n_crop)] = True
            else:
                continue
        if k == c_idx:
            n_center = int(keep.sum())
        ex = cxk[k] + crop.dx[keep] * sf
        ey = cyk[k] + crop.dy[keep] * sf
        et = starts[k] + crop.rel[keep] * wu
        ep = crop.pol[keep].astype(np.int64)
        if pol_bias is not None:
            sign, frac = pol_bias
            flip = rng.random(ep.size) < frac
            ep[flip] = sign
        ox_list.append(ex); oy_list.append(ey)
        op_list.append(ep); ot_list.append(et)

    # min-event guard: if the object is effectively absent at the center window, keep the
    # window as BACKGROUND rather than training a positive heatmap for near-zero evidence.
    if cfg.min_center_events > 0 and n_center < cfg.min_center_events:
        return xn, yn, pol, t, None

    add_x = [xn * W]; add_y = [yn * H]
    add_p = [pol]; add_t = [t]
    if ox_list:
        add_x.append(np.concatenate(ox_list))
        add_y.append(np.concatenate(oy_list))
        add_p.append(np.concatenate(op_list))
        add_t.append(np.concatenate(ot_list))

    # --- hot pixels (sensor artifact / hard negative) ------------------------ #
    if rng.random() < cfg.p_hot:
        for _ in range(int(rng.integers(1, cfg.hot_max + 1))):
            hx = rng.uniform(0, W); hy = rng.uniform(0, H)
            r = int(rng.integers(cfg.hot_rate_lo, cfg.hot_rate_hi + 1))
            add_x.append(np.full(r, hx)); add_y.append(np.full(r, hy))
            add_p.append(rng.integers(0, 2, r))
            add_t.append(rng.uniform(ws, we, r))

    # --- correlated background bursts (clutter hard negative) ---------------- #
    if rng.random() < cfg.p_clutter:
        for _ in range(int(rng.integers(1, cfg.clutter_max + 1))):
            bx = rng.uniform(0, W); by = rng.uniform(0, H)
            m = int(rng.integers(cfg.clutter_ev_lo, cfg.clutter_ev_hi + 1))
            add_x.append(rng.normal(bx, cfg.clutter_sigma_px, m))
            add_y.append(rng.normal(by, cfg.clutter_sigma_px, m))
            add_p.append(rng.integers(0, 2, m))
            add_t.append(rng.uniform(ws, we, m))

    px = np.concatenate(add_x); py = np.concatenate(add_y)
    pp = np.concatenate(add_p).astype(np.int64)
    pt = np.concatenate(add_t).astype(np.float64)
    # back to normalized coords; drop anything the jitter/motion pushed off-frame
    nx = px / W; ny = py / H
    inb = (nx >= 0) & (nx < 1) & (ny >= 0) & (ny < 1)
    nx, ny, pp, pt = nx[inb], ny[inb], pp[inb], pt[inb]

    # --- GT box at the center window ----------------------------------------- #
    bx = float(np.clip((p0x + ox[c_idx] + jx[c_idx]) / W, 0.0, 1.0))
    by = float(np.clip((p0y + oy[c_idx] + jy[c_idx]) / H, 0.0, 1.0))
    box = (bx, by,
           float(np.clip(w_px / W, 1e-3, 1.0)),
           float(np.clip(h_px / H, 1e-3, 1.0)))
    return nx, ny, pp, pt, box


# --------------------------------------------------------------------------- #
#  CLI: measure real object statistics (the "ground the ranges first" step)
# --------------------------------------------------------------------------- #
def _main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from orbitsight import data as D  # noqa: E402
    from orbitsight.config import (DEFAULT_CONFIG, TRAIN_SEQUENCES,  # noqa: E402
                                   sensor_for_sequence)

    ap = argparse.ArgumentParser(
        description="Measure real object velocity/event-rate stats for injection.")
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Training_sets")
    args = ap.parse_args()

    events, sensors, items = {}, {}, []
    for seq in TRAIN_SEQUENCES:
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        sn = sensor_for_sequence(seq)
        events[seq] = ev
        sensors[seq] = sn
        gt = {}
        gp = os.path.join(args.data_dir, seq + D.GT_SUFFIX)
        if os.path.exists(gp):
            for ws, we, cx, cy, w, h in D.load_gt_boxes(gp):
                gt[ws] = (cx / sn.width, cy / sn.height, w / sn.width, h / sn.height)
        for win in D.make_window_grid(ev.t, DEFAULT_CONFIG.window_us):
            items.append((seq, win.lo, win.hi, win.start_us, win.end_us,
                          gt.get(win.start_us)))

    lib = CropLibrary.from_windowset(
        events, items, sensors, window_us=DEFAULT_CONFIG.window_us)
    print("[inject] measured crop library:")
    print("  " + lib.summary())


if __name__ == "__main__":
    _main()
