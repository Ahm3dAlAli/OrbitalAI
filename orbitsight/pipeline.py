"""End-to-end per-sequence orchestration of Stages 0-4.

Two entry points:
  * :func:`extract_training_samples` — for offline classifier training, returns
    per-event features + labels (RSO vs background) from labeled sequences.
  * :func:`run_sequence` — full inference: events -> list[Detection], the
    per-window boxes written to ``<sequence>_bb_windows_40ms.txt``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import data as D
from .config import Config, Sensor, sensor_for_sequence
from .features import build_cloud
from .model import CoherenceClassifier
from .detect import window_candidates, track_candidates, tracks_to_detections


@dataclass
class SeqResult:
    seq: str
    sensor: str
    n_events: int
    n_windows: int
    n_detections: int
    seconds: float
    ms_per_window: float


# --------------------------------------------------------------------------- #
#  Shared per-window driver
# --------------------------------------------------------------------------- #
def _iter_window_features(events: np.ndarray, sensor: Sensor, cfg: Config):
    """Yield (win, core_feats, core_idx_global, keep_local) per window.

    core_feats   : features for denoise-surviving core events
    core_idx_global: their indices into the full ``events`` array
    """
    ts = events.t
    wins = D.make_window_grid(ts, cfg.window_us)
    x_all = events.x
    y_all = events.y
    pol_all = events.pol

    for win in wins:
        lo_h, hi_h = D.slice_with_halo(ts, win, cfg.halo_us)
        if hi_h - lo_h == 0:
            yield win, np.zeros((0, 12), np.float32), np.zeros(0, np.int64), None
            continue
        cloud = build_cloud(x_all[lo_h:hi_h], y_all[lo_h:hi_h],
                            ts[lo_h:hi_h], pol_all[lo_h:hi_h], sensor, cfg)
        # core event positions within the haloed slice
        c_lo = win.lo - lo_h
        c_hi = win.hi - lo_h
        if c_hi <= c_lo:
            yield win, np.zeros((0, 12), np.float32), np.zeros(0, np.int64), None
            continue
        keep = cloud.denoise_keep(c_lo, c_hi)                     # Stage 1
        kept_local = np.nonzero(keep)[0]                          # within core
        if kept_local.size == 0:
            yield win, np.zeros((0, 12), np.float32), np.zeros(0, np.int64), None
            continue
        # Latency guard: in dense (bright) windows, subsample survivors before
        # the O(N·k) feature loop.  Sparse (dim-object) windows are untouched,
        # so dim objects keep all their few events.  Deterministic per window.
        if kept_local.size > cfg.feat_query_cap:
            rng = np.random.default_rng(win.index)
            kept_local = np.sort(rng.choice(kept_local, cfg.feat_query_cap,
                                            replace=False))
        # featurize only denoise survivors (indices into the haloed cloud)
        query_idx = c_lo + kept_local
        core_feats = cloud.features(query_idx)                    # Stage 2
        core_idx_global = win.lo + kept_local
        yield win, core_feats, core_idx_global, kept_local


# --------------------------------------------------------------------------- #
#  Offline: build training samples
# --------------------------------------------------------------------------- #
def extract_training_samples(events: np.ndarray, sensor: Sensor, cfg: Config,
                             rng: np.random.Generator,
                             max_rso_per_window: int = 300,
                             max_rso_per_seq: int = 60_000):
    """Return (X, y) for classifier training.

    For efficiency, query events are *selected* per window (denoise-surviving
    RSO events, capped, plus a class-balanced background subsample) BEFORE
    coherence features are computed — so we never featurize noise we will throw
    away.  This keeps even the dense, 50%-RSO EVK4 sequence tractable.
    """
    ts = events.t
    x_all = events.x
    y_all = events.y
    pol_all = events.pol
    label_all = events.label

    wins = D.make_window_grid(ts, cfg.window_us)
    Xs, ys = [], []
    rso_budget = max_rso_per_seq

    for win in wins:
        lo_h, hi_h = D.slice_with_halo(ts, win, cfg.halo_us)
        if hi_h - lo_h == 0 or win.hi <= win.lo:
            continue
        cloud = build_cloud(x_all[lo_h:hi_h], y_all[lo_h:hi_h],
                            ts[lo_h:hi_h], pol_all[lo_h:hi_h], sensor, cfg)
        c_lo, c_hi = win.lo - lo_h, win.hi - lo_h
        keep = cloud.denoise_keep(c_lo, c_hi)
        kept_local = np.nonzero(keep)[0]            # within core
        if kept_local.size == 0:
            continue
        g_idx = win.lo + kept_local                 # global event indices
        lab = (label_all[g_idx] == 1).astype(np.int8)
        pos = np.nonzero(lab == 1)[0]
        neg = np.nonzero(lab == 0)[0]

        # cap RSO queries per window and per sequence
        if pos.size > max_rso_per_window:
            pos = rng.choice(pos, max_rso_per_window, replace=False)
        if pos.size > rso_budget:
            pos = rng.choice(pos, rso_budget, replace=False)
        rso_budget -= pos.size

        keep_neg = max(pos.size * cfg.bg_per_rso, 8)
        sel_neg = neg if neg.size <= keep_neg else rng.choice(neg, keep_neg, replace=False)

        sel = np.concatenate([pos, sel_neg]) if pos.size or sel_neg.size else np.empty(0, int)
        if sel.size == 0:
            continue
        query_idx = c_lo + kept_local[sel]          # into haloed cloud
        feats = cloud.features(query_idx)
        Xs.append(feats)
        ys.append(lab[sel])

    if not Xs:
        return np.zeros((0, 12), np.float32), np.zeros(0, np.int8)
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


# --------------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------------- #
def run_sequence(events: "D.Events", seq_name: str, clf: CoherenceClassifier,
                 cfg: Config) -> tuple[list[D.Detection], SeqResult]:
    sensor = sensor_for_sequence(seq_name)
    x_all = events.x
    y_all = events.y
    n_win = 0
    t0 = time.perf_counter()

    # Pass 1 — per-window candidate proposals (classify + cluster).
    per_window: list[list] = []
    for win, feats, idx_global, _ in _iter_window_features(events, sensor, cfg):
        n_win += 1
        if feats.shape[0] > 0:
            scores = clf.predict_proba(feats)                    # Stage 2 head
            cands = window_candidates(win.index, win.start_us, win.end_us,
                                      x_all[idx_global], y_all[idx_global],
                                      scores, sensor, cfg)        # Stage 3a
        else:
            cands = []
        per_window.append(cands)

    # Pass 2 — link candidates into tracks and emit one box per window.
    tracks = track_candidates(per_window, sensor, cfg)            # Stage 3b
    dets = tracks_to_detections(tracks, sensor, cfg,
                                n_windows=len(per_window))       # Stage 3c/4

    secs = time.perf_counter() - t0
    res = SeqResult(seq_name, sensor.name, int(events.n), n_win,
                    len(dets), secs, 1000.0 * secs / max(n_win, 1))
    return dets, res
