"""Temporal accumulation upstream of the detection threshold — "synthetic
tracking" / shift-and-stack for dim moving objects (Tech Report §5.4: integrate
evidence over time so sparse, dim-object windows still register).

A faint RSO emits only 2-4 events per 40 ms window — too few for a per-window
detector.  But it persists on a smooth motion track.  If we shift every event
back along a hypothesized constant velocity to a common reference time, the
object's events from many windows collapse onto one spot (they "stack"), while
background activity smears out.  Searching a small velocity grid and looking for
the stacked peak recovers objects no single window could surface.

This runs BEFORE any per-event classification, so it can find objects that never
produce a candidate downstream — a different axis from the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, Sensor
from . import data as D


@dataclass
class StackDet:
    win_idx: int        # window this box belongs to
    ws: int
    we: int
    cx: float
    cy: float
    vx: float           # px / window
    vy: float
    peak: float         # stacked peak count (evidence)
    snr: float          # peak / background level


def hot_pixel_mask(events: "D.Events", sensor: Sensor,
                   win_frac: float = 0.15) -> np.ndarray:
    """Boolean keep-mask removing static hot sources (hot pixels, bright fixed
    stars) that otherwise dominate the v=0 stack.

    A static source occupies the *same* pixel across a large fraction of time
    windows; a moving RSO touches each pixel only briefly.  So we flag pixels
    active in more than ``win_frac`` of all 40 ms windows and drop their events.
    """
    W, H = sensor.width, sensor.height
    xi = np.clip(events.x.astype(np.int64), 0, W - 1)
    yi = np.clip(events.y.astype(np.int64), 0, H - 1)
    t = events.t
    n_win = max(int((t[-1] - t[0]) // 40_000) + 1, 1)
    win_id = ((t - t[0]) // 40_000).astype(np.int64)
    pix = xi * H + yi
    # count DISTINCT windows per pixel via unique (pixel, window) pairs
    key = pix * n_win + win_id
    uniq = np.unique(key)
    upix = uniq // n_win
    active_windows = np.bincount(upix, minlength=W * H)
    hot = active_windows > (win_frac * n_win)
    return ~hot[pix]


def _velocity_grid(vmax: float, step: float) -> np.ndarray:
    v = np.arange(-vmax, vmax + 1e-9, step)
    VX, VY = np.meshgrid(v, v)
    return np.column_stack([VX.ravel(), VY.ravel()])


def stack_block(x, y, dtw, W, H, cell, vels):
    """Shift-and-stack one block of events over a velocity grid.

    A real object is an *outlier in velocity space*: its true velocity produces
    a peak far above the peak ANY wrong velocity produces (those see only
    background, which doesn't cohere).  A background-only block has similar peaks
    at all velocities.  So we return the best peak AND the distribution of
    per-velocity peaks, and the caller thresholds on best / typical-peak.

    Returns: (vx, vy, cx, cy, best_peak, typ_peak)
        typ_peak = median peak across the velocity grid (background reference).
    """
    nx = int(np.ceil(W / cell))
    ny = int(np.ceil(H / cell))
    best = None
    best_peak = -1.0
    peaks = np.empty(len(vels), dtype=np.float64)
    for i, (vx, vy) in enumerate(vels):
        xs = x - vx * dtw
        ys = y - vy * dtw
        xi = np.floor(xs / cell).astype(np.int64)
        yi = np.floor(ys / cell).astype(np.int64)
        m = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        if not m.any():
            peaks[i] = 0.0
            continue
        counts = np.bincount(xi[m] * ny + yi[m], minlength=nx * ny)
        pk = int(counts.argmax())
        c = float(counts[pk])
        peaks[i] = c
        if c > best_peak:
            best_peak = c
            cxi, cyi = divmod(pk, ny)
            best = (float(vx), float(vy), (cxi + 0.5) * cell, (cyi + 0.5) * cell)
    if best is None:
        return None
    typ = float(np.median(peaks)) + 1e-6
    return (*best, best_peak, typ)


def detect_sequence(events: "D.Events", sensor: Sensor, cfg: Config,
                    keep_mask: np.ndarray | None = None,
                    block: int = 11, stride: int = 4,
                    vmax: float = 8.0, vstep: float = 1.0,
                    cell: float = 6.0, min_ratio: float = 3.0,
                    min_peak: int = 12) -> list[StackDet]:
    """Run shift-and-stack over a whole sequence; emit per-window detections
    for the dominant stacked track in each block.

    A block fires only when its best-velocity peak both (a) clears an absolute
    floor ``min_peak`` and (b) exceeds ``min_ratio`` x the typical per-velocity
    peak (i.e. the stack is a genuine velocity-space outlier, not background).
    Pass ``keep_mask`` (denoise survivors) to stack on cleaned events.
    """
    ts = events.t
    W, H = sensor.width, sensor.height
    wins = D.make_window_grid(ts, cfg.window_us)
    vels = _velocity_grid(vmax, vstep)
    win_us = cfg.window_us
    out: list[StackDet] = []

    for b0 in range(0, len(wins) - 1, stride):
        b1 = min(b0 + block, len(wins))
        wlo, whi = wins[b0], wins[b1 - 1]
        lo = wlo.lo
        hi = whi.hi
        if hi - lo < min_peak:
            continue
        ref_idx = (b0 + b1) // 2
        t_ref = wins[ref_idx].start_us
        sl = slice(lo, hi)
        if keep_mask is not None:
            idx = np.nonzero(keep_mask[sl])[0] + lo
            if idx.size < min_peak:
                continue
            x = events.x[idx].astype(np.float64)
            y = events.y[idx].astype(np.float64)
            dtw = (ts[idx].astype(np.float64) - t_ref) / win_us
        else:
            x = events.x[sl].astype(np.float64)
            y = events.y[sl].astype(np.float64)
            dtw = (ts[sl].astype(np.float64) - t_ref) / win_us
        res = stack_block(x, y, dtw, W, H, cell, vels)
        if res is None:
            continue
        vx, vy, cx, cy, peak, typ = res
        snr = peak / typ
        if peak < min_peak or snr < min_ratio:
            continue
        # propagate the stacked detection to every window in the block
        for j in range(b0, b1):
            wj = wins[j]
            dj = j - ref_idx
            px = cx + vx * dj
            py = cy + vy * dj
            if not (0 <= px < W and 0 <= py < H):
                continue
            out.append(StackDet(wj.index, wj.start_us, wj.end_us,
                                px, py, vx, vy, peak, snr))
    return out


def stack_sequence(events: "D.Events", sensor: Sensor, cfg: Config) -> list:
    """Convenience wrapper: hot-pixel removal + shift-and-stack + collapse to
    one box per window, using the stack-* parameters from the Config."""
    keep = hot_pixel_mask(events, sensor, win_frac=cfg.stack_hot_frac)
    sd = detect_sequence(events, sensor, cfg, keep_mask=keep,
                         block=cfg.stack_block, stride=cfg.stack_stride,
                         vmax=cfg.stack_vmax, vstep=cfg.stack_vstep,
                         cell=cfg.stack_cell, min_ratio=cfg.stack_min_ratio,
                         min_peak=cfg.stack_min_peak)
    return to_detections(sd, sensor, cfg)


def merge_fill(main: list, stack: list, max_conf: float = 0.5) -> list:
    """Merge stacked detections into the main per-window detections, filling
    only windows the main pipeline left empty (recovers dim-object recall
    without overriding the higher-precision classifier boxes)."""
    by_ws = {d.ws: d for d in main}
    for d in stack:
        if d.ws not in by_ws:
            d.conf = min(d.conf, max_conf)
            by_ws[d.ws] = d
    return [by_ws[k] for k in sorted(by_ws)]


def to_detections(stack_dets: list[StackDet], sensor: Sensor, cfg: Config):
    """Collapse per-window stacked detections (overlapping blocks produce
    duplicates) into one box per window, keeping the highest-SNR hit, sized at
    the per-sensor typical GT box."""
    typ = cfg.box_size_px.get(sensor.name, (cfg.min_box_px * 3, cfg.min_box_px * 3))
    best: dict[int, StackDet] = {}
    for d in stack_dets:
        cur = best.get(d.win_idx)
        if cur is None or d.snr > cur.snr:
            best[d.win_idx] = d
    dets = []
    W, H = sensor.width, sensor.height
    peaks = np.array([d.peak for d in best.values()]) if best else np.array([1.0])
    pmax = float(peaks.max()) + 1e-9
    for wi in sorted(best):
        d = best[wi]
        conf = min(1.0, 0.3 + 0.7 * d.peak / pmax)
        dets.append(D.Detection(
            d.ws, d.we,
            int(round(np.clip(d.cx, 0, W - 1))), int(round(np.clip(d.cy, 0, H - 1))),
            int(round(typ[0])), int(round(typ[1])), float(conf)))
    return dets
