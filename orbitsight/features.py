"""Stages 0-2: per-sensor normalization, O(N) background-activity denoising,
and learned-classifier feature extraction over local (x, y, t) coherence.

Design notes
------------
* All spatial work is done in *normalized* coordinates (pixel / sensor.diag)
  so a single parameter set transfers across DAVIS / DVX / EVK4 (NFR-3).
* Hypothesis H1 (Tech Report §3.1): RSO events lie on locally linear,
  temporally coherent streaks in (x, y, t); background activity is isotropic.
  The discriminative features therefore measure *local coherence*, not
  appearance: PCA linearity, neighbor density, flow consistency, polarity.
* Everything is computed per 40 ms window (plus a temporal halo) from a single
  KD-tree on a few-thousand-event point cloud.  Two radius queries give us
  (a) the background-activity denoise decision and (b) the coherence features.
  -> O(N) overall, CPU-real-time, no GPU primitives.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import Config, Sensor

# Number of features must match cfg.feature_names.
_N_FEATURES = 12


# --------------------------------------------------------------------------- #
#  Stage 0 — normalization
# --------------------------------------------------------------------------- #
def normalize_xy(x: np.ndarray, y: np.ndarray, sensor: Sensor):
    """Map pixel coordinates to a resolution-agnostic frame (divide by the
    sensor diagonal).  Returns float arrays roughly in [0, ~1]."""
    d = sensor.diag
    return x.astype(np.float64) / d, y.astype(np.float64) / d


def scaled_time(t: np.ndarray, cfg: Config) -> np.ndarray:
    """Scale microsecond timestamps so a ``feat_time_us`` gap maps to
    ``time_scale`` in normalized space (comparable to the spatial axes)."""
    t0 = float(t[0]) if t.size else 0.0
    return (t.astype(np.float64) - t0) / float(cfg.feat_time_us) * cfg.time_scale


# --------------------------------------------------------------------------- #
#  Combined Stage 1 + Stage 2 over one window's (haloed) point cloud
# --------------------------------------------------------------------------- #
class WindowCloud:
    """Holds the KD-tree and normalized coordinates for one haloed window so
    denoising and feature extraction reuse a single spatial index."""

    __slots__ = ("xn", "yn", "tn", "pol", "pts", "tree", "cfg", "diag")

    def __init__(self, xn, yn, tn, pol, cfg: Config, diag: float):
        self.xn, self.yn, self.tn, self.pol = xn, yn, tn, pol
        self.cfg = cfg
        self.diag = diag
        self.pts = np.column_stack([xn, yn, tn])
        self.tree = cKDTree(self.pts) if self.pts.shape[0] else None

    # -- Stage 1: adaptive background-activity denoise ------------------- #
    def denoise_keep(self, lo: int, hi: int) -> np.ndarray:
        """Boolean keep-mask for events ``[lo:hi]``: survive iff the event has
        >= ``denoise_min_support`` neighbors inside a small space-time ball.

        This is the geometric equivalent of a timestamp-map BA filter: isolated
        (incoherent) noise events have no nearby support and are dropped, while
        events on a moving-object streak are densely supported.
        """
        n = hi - lo
        if n == 0 or self.tree is None:
            return np.zeros(n, dtype=bool)
        # denoise ball: a few px spatially (normalized units), within the
        # temporal halo encoded by tn.  Kept gentle so that DIM objects
        # (only 1-3 events per window) survive while truly isolated noise is
        # dropped (H2: BA noise is incoherent; sparse signal must not be lost).
        r = self.cfg.denoise_radius_px / self.diag
        counts = self.tree.query_ball_point(self.pts[lo:hi], r=r, workers=-1,
                                             return_length=True)
        # counts include the event itself; require >= denoise_min_support
        # supporting neighbors (so total > min_support).
        return counts > self.cfg.denoise_min_support

    # -- Stage 2: coherence features (vectorized, fixed-k) -------------- #
    def features(self, query_idx: np.ndarray) -> np.ndarray:
        """Coherence features for events at local indices ``query_idx``.

        Uses fixed-k nearest neighbors so the whole computation is batched
        NumPy (covariance, eigenvalues, flow regression, polarity) — orders of
        magnitude faster than a per-event Python loop, which matters for the
        dense EVK4 stream and the <40 ms/window budget.
        """
        cfg = self.cfg
        n = int(query_idx.size)
        out = np.zeros((n, _N_FEATURES), dtype=np.float32)
        if n == 0 or self.tree is None:
            return out
        npts = self.pts.shape[0]
        k = min(cfg.feat_max_neighbors, npts)      # neighbors incl. self
        if k < 2:
            return out
        q = self.pts[query_idx]
        dist, idx = self.tree.query(q, k=k, workers=-1)   # (n,k)
        if k == 1:
            dist = dist[:, None]; idx = idx[:, None]

        P = self.pts[idx]                          # (n,k,3)
        within = dist <= cfg.feat_radius_norm      # neighbors inside the ball
        m = within.sum(axis=1).astype(np.float32)  # (n,)  count incl. self
        out[:, 0] = m
        # density: count / volume of the ball reaching the farthest in-ball nb
        rad = np.where(within, dist, 0.0).max(axis=1) + 1e-6
        out[:, 1] = m / ((4.0 / 3.0) * np.pi * rad ** 3 + 1e-12)

        # mask neighbors beyond the ball by collapsing them onto the centroid
        # (zero weight) — use a weight mask for the moments.
        w = within.astype(np.float64)              # (n,k)
        wsum = w.sum(axis=1, keepdims=True) + 1e-9
        c = (P * w[:, :, None]).sum(axis=1, keepdims=True) / wsum[:, :, None]
        Q = (P - c) * w[:, :, None]                # centered, masked  (n,k,3)

        # batched covariance and eigenvalues
        cov = np.einsum("nki,nkj->nij", Q, Q) / wsum[:, :, None]   # (n,3,3)
        evals = np.linalg.eigvalsh(cov)            # ascending (n,3)
        l3 = np.clip(evals[:, 0], 0, None)
        l2 = np.clip(evals[:, 1], 0, None)
        l1 = np.clip(evals[:, 2], 0, None)
        ssum = l1 + l2 + l3 + 1e-12
        good = m >= cfg.feat_min_neighbors
        out[:, 2] = np.where(good, l1 / (l2 + l3 + 1e-9), 0)
        out[:, 3] = np.where(good, l2 / (l3 + 1e-9), 0)
        out[:, 4] = np.where(good, l1 / ssum, 0)
        out[:, 11] = np.where(good, 1.0 - l3 / (l1 + 1e-9), 0)

        qx, qy, qt = Q[:, :, 0], Q[:, :, 1], Q[:, :, 2]
        out[:, 7] = np.sqrt((qt ** 2).sum(axis=1) / wsum[:, 0])           # time spread
        out[:, 8] = np.sqrt((qx ** 2 + qy ** 2).sum(axis=1) / wsum[:, 0])  # space spread

        # flow: least-squares velocity (slope of position vs time) + residual
        denom = (qt ** 2).sum(axis=1) + 1e-12
        vx = (qt * qx).sum(axis=1) / denom
        vy = (qt * qy).sum(axis=1) / denom
        out[:, 6] = np.hypot(vx, vy)
        res = np.sqrt((((qx - vx[:, None] * qt) ** 2 +
                        (qy - vy[:, None] * qt) ** 2) * w).sum(axis=1) / wsum[:, 0])
        out[:, 5] = np.exp(-res / (out[:, 8] + 1e-9))

        # polarity structure over in-ball neighbors
        pol = self.pol[idx].astype(np.float64)     # (n,k)
        pm = (pol * w).sum(axis=1) / wsum[:, 0]
        out[:, 9] = pm
        p1 = np.clip(pm, 1e-6, 1 - 1e-6)
        out[:, 10] = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))

        out[~good, 2:] = 0.0
        return np.nan_to_num(out, copy=False)


def build_cloud(x, y, t, pol, sensor: Sensor, cfg: Config) -> WindowCloud:
    """Construct a :class:`WindowCloud` from raw (pixel) event columns."""
    xn, yn = normalize_xy(x, y, sensor)
    tn = scaled_time(t, cfg)
    return WindowCloud(xn, yn, tn, pol, cfg, sensor.diag)
