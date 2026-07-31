"""Motion-guided track-before-detect utilities for faint single-object streams.

The ordinary CenterNet path thresholds each 40 ms heatmap independently.  That
throws away weak peaks before the temporal tracker can use them.  Thuraya3 is
the regime where this is most costly: the target is coherent over time but is
often represented by only a few events in any one window.

This module links *sub-threshold* heatmap peaks with a bounded beam search.  A
hypothesis is rewarded for detector evidence and penalized for position and
velocity innovations.  Static tracks are rejected, which prevents a persistent
hot pixel from winning merely because it is easy to link.  The implementation
is NumPy-only so the temporal logic can be tested without importing PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .data import Detection


@dataclass(frozen=True)
class Peak:
    """One local heatmap maximum in one challenge window."""

    win_idx: int
    ws: int
    we: int
    cx: float
    cy: float
    w: float
    h: float
    score: float


@dataclass(frozen=True)
class BeamConfig:
    """Hyper-parameters for the single-target motion beam."""

    peak_floor: float = 0.03
    seed_thresh: float = 0.10
    gate_px: float = 10.0
    first_link_scale: float = 1.6
    gate_growth_px: float = 2.0
    max_gap: int = 8
    velocity_alpha: float = 0.65
    accel_scale_px: float = 3.0
    emission_weight: float = 1.0
    match_bonus: float = 0.35
    position_weight: float = 1.25
    acceleration_weight: float = 0.20
    miss_penalty: float = 0.30
    min_track_len: int = 8
    min_displacement_px: float = 8.0
    max_rms_residual_px: float = 8.0
    beam_size: int = 256
    velocity_bin_px: float = 0.5


@dataclass(frozen=True)
class TrackResult:
    """The best coherent path and diagnostics used by confidence calibration."""

    peaks: tuple[Peak, ...]
    residuals: tuple[float, ...]
    objective: float
    displacement_px: float
    rms_residual_px: float


@dataclass(frozen=True)
class _State:
    objective: float
    peaks: tuple[Peak, ...]
    residuals: tuple[float, ...]
    vx: float
    vy: float
    misses: int

    @property
    def last(self) -> Peak:
        return self.peaks[-1]


def _emission(score: float, cfg: BeamConfig) -> float:
    # Evidence is measured relative to the permissive peak floor.  A peak at
    # the floor is neutral; stronger peaks add evidence without requiring the
    # usual independent-window acceptance threshold.
    return cfg.emission_weight * math.log(
        max(float(score), cfg.peak_floor) / cfg.peak_floor
    )


def _extend(st: _State, p: Peak, cfg: BeamConfig) -> _State | None:
    dt = p.win_idx - st.last.win_idx
    if dt <= 0 or dt > cfg.max_gap + 1:
        return None

    if len(st.peaks) == 1:
        pred_x, pred_y = st.last.cx, st.last.cy
        gate = cfg.gate_px * cfg.first_link_scale * dt
    else:
        pred_x = st.last.cx + st.vx * dt
        pred_y = st.last.cy + st.vy * dt
        gate = cfg.gate_px + cfg.gate_growth_px * max(dt - 1, 0)

    residual = float(math.hypot(p.cx - pred_x, p.cy - pred_y))
    if residual > gate:
        return None

    obs_vx = (p.cx - st.last.cx) / dt
    obs_vy = (p.cy - st.last.cy) / dt
    if len(st.peaks) == 1:
        nvx, nvy = obs_vx, obs_vy
        accel = 0.0
    else:
        accel = math.hypot(obs_vx - st.vx, obs_vy - st.vy)
        a = cfg.velocity_alpha
        nvx = a * st.vx + (1.0 - a) * obs_vx
        nvy = a * st.vy + (1.0 - a) * obs_vy

    pos_cost = cfg.position_weight * (residual / max(gate, 1e-6)) ** 2
    acc_cost = cfg.acceleration_weight * (
        accel / max(cfg.accel_scale_px, 1e-6)
    ) ** 2
    gap_cost = cfg.miss_penalty * max(dt - 1, 0)
    objective = (
        st.objective
        + _emission(p.score, cfg)
        + cfg.match_bonus
        - pos_cost
        - acc_cost
        - gap_cost
    )
    return _State(
        objective=objective,
        peaks=st.peaks + (p,),
        residuals=st.residuals + (residual,),
        vx=nvx,
        vy=nvy,
        misses=0,
    )


def _prune(states: Sequence[_State], cfg: BeamConfig) -> list[_State]:
    """Diversity-aware beam pruning.

    Hypotheses ending at nearly the same position and velocity are equivalent
    for future expansion.  Keeping only the best one in each coarse state bin
    prevents copies of one path from crowding all other candidates out.
    """

    bins: dict[tuple[int, int, int, int, int], _State] = {}
    vb = max(cfg.velocity_bin_px, 1e-6)
    for st in states:
        key = (
            int(round(st.last.cx / 2.0)),
            int(round(st.last.cy / 2.0)),
            int(round(st.vx / vb)),
            int(round(st.vy / vb)),
            st.misses,
        )
        old = bins.get(key)
        if old is None or st.objective > old.objective:
            bins[key] = st
    return sorted(bins.values(), key=lambda s: s.objective, reverse=True)[
        : cfg.beam_size
    ]


def link_peaks(
    per_window: Sequence[Sequence[Peak]], cfg: BeamConfig = BeamConfig()
) -> TrackResult | None:
    """Return the strongest moving path through local heatmap peaks.

    Peaks may be below the detector's normal output threshold.  Births require
    ``seed_thresh`` evidence, while an established path may continue through
    weaker peaks down to ``peak_floor``.  This asymmetry is the core
    track-before-detect behavior.
    """

    active: list[_State] = []
    finished: list[_State] = []

    for win_peaks in per_window:
        candidates = [
            p for p in win_peaks if float(p.score) >= cfg.peak_floor
        ]
        nxt: list[_State] = []

        for st in active:
            matched = False
            for p in candidates:
                ext = _extend(st, p, cfg)
                if ext is not None:
                    nxt.append(ext)
                    matched = True

            if st.misses < cfg.max_gap:
                # Keep a predict-only hypothesis alive.  It can attach to a
                # later weak peak, but the accumulated miss cost prevents long
                # evidence-free paths from dominating.
                nxt.append(
                    _State(
                        objective=st.objective - cfg.miss_penalty,
                        peaks=st.peaks,
                        residuals=st.residuals,
                        vx=st.vx,
                        vy=st.vy,
                        misses=st.misses + 1,
                    )
                )
            elif not matched:
                finished.append(st)

        for p in candidates:
            if p.score >= cfg.seed_thresh:
                nxt.append(
                    _State(
                        objective=_emission(p.score, cfg),
                        peaks=(p,),
                        residuals=(0.0,),
                        vx=0.0,
                        vy=0.0,
                        misses=0,
                    )
                )
        active = _prune(nxt, cfg)

    finished.extend(active)
    valid: list[tuple[_State, float, float]] = []
    for st in finished:
        if len(st.peaks) < cfg.min_track_len:
            continue
        first, last = st.peaks[0], st.peaks[-1]
        disp = float(math.hypot(last.cx - first.cx, last.cy - first.cy))
        rms = float(np.sqrt(np.mean(np.square(st.residuals[1:])))) if len(
            st.residuals
        ) > 1 else 0.0
        if disp < cfg.min_displacement_px or rms > cfg.max_rms_residual_px:
            continue
        valid.append((st, disp, rms))

    if not valid:
        return None
    # The objective already grows with coherent evidence and duration.  A tiny
    # length tie-break favors a long faint path over a short bright fragment.
    best, disp, rms = max(
        valid, key=lambda x: x[0].objective + 1e-3 * len(x[0].peaks)
    )
    return TrackResult(
        peaks=best.peaks,
        residuals=best.residuals,
        objective=best.objective,
        displacement_px=disp,
        rms_residual_px=rms,
    )


def track_to_detections(
    track: TrackResult,
    *,
    sensor_width: int,
    sensor_height: int,
    box_px: tuple[float, float] | None = None,
    fill_decay: float = 0.88,
) -> list[Detection]:
    """Emit observed and interpolated boxes across the selected track span."""

    obs = {p.win_idx: p for p in track.peaks}
    idx = sorted(obs)
    median_w = float(np.median([p.w for p in track.peaks]))
    median_h = float(np.median([p.h for p in track.peaks]))
    if box_px is not None:
        median_w, median_h = map(float, box_px)
    residual = {p.win_idx: r for p, r in zip(track.peaks, track.residuals)}
    maturity = 1.0 - math.exp(-len(track.peaks) / 6.0)
    out: list[Detection] = []

    for wi in range(idx[0], idx[-1] + 1):
        if wi in obs:
            p = obs[wi]
            support = math.exp(
                -0.5 * (residual[wi] / max(track.rms_residual_px + 1.0, 1.0)) ** 2
            )
            conf = math.sqrt(max(p.score, 1e-6) * max(support, 1e-6)) * maturity
            cx, cy = p.cx, p.cy
            bw, bh = (median_w, median_h) if box_px is not None else (p.w, p.h)
            ws, we = p.ws, p.we
        else:
            left_i = max(i for i in idx if i < wi)
            right_i = min(i for i in idx if i > wi)
            left, right = obs[left_i], obs[right_i]
            frac = (wi - left_i) / (right_i - left_i)
            cx = left.cx + frac * (right.cx - left.cx)
            cy = left.cy + frac * (right.cy - left.cy)
            bw, bh = median_w, median_h
            win_us = left.we - left.ws
            ws = left.ws + (wi - left_i) * win_us
            we = ws + win_us
            gap = min(wi - left_i, right_i - wi)
            conf = (
                math.sqrt(max(min(left.score, right.score), 1e-6))
                * maturity
                * 0.65
                * fill_decay**gap
            )

        out.append(
            Detection(
                int(ws),
                int(we),
                int(round(np.clip(cx, 0, sensor_width - 1))),
                int(round(np.clip(cy, 0, sensor_height - 1))),
                max(int(round(np.clip(bw, 1, sensor_width))), 1),
                max(int(round(np.clip(bh, 1, sensor_height))), 1),
                float(np.clip(conf, 1e-4, 1.0)),
            )
        )
    return out
