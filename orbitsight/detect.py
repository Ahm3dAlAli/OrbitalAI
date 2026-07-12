"""Stage 3 + Stage 4: turn per-event RSO scores into one bounding box per
40 ms window via global, motion-gated trajectory tracking.

Rationale (Tech Report §5.4): the GT has exactly one box per window and the
object moves smoothly, while background activity and hot-pixel clutter are
either incoherent or *static*.  So instead of picking a box independently per
window (which locks onto whatever clutter is densest), we:

  1. propose candidate clusters per window (cheap, permissive),
  2. link candidates across windows into tracks with a constant-velocity
     model + gating (integrates evidence over time -> dim objects register),
  3. keep tracks that are long enough and *moving* (rejects static clutter),
  4. emit one box per window from the surviving track(s), interpolating gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .config import Config, Sensor
from .data import Detection


# --------------------------------------------------------------------------- #
#  Clustering (scipy-only; avoids sklearn's segfaulting OpenMP kernel)
# --------------------------------------------------------------------------- #
def _cluster(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = pts.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    labels = np.full(n, -1, dtype=np.int64)
    nl = 0
    for root in np.unique(roots):
        members = np.nonzero(roots == root)[0]
        if members.size >= min_samples:
            labels[members] = nl
            nl += 1
    return labels


# --------------------------------------------------------------------------- #
#  Candidate proposals (one or more per window)
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    win_idx: int
    ws: int
    we: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    n: int
    score: float


def _gate(scores: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean keep-mask for a window's event scores.

    With ``use_percentile_gate``, keep the top ``(100-keep_percentile)%`` events
    in this window subject to an absolute ``score_floor`` — a single parameter
    that adapts its operating point to the per-window event rate, so dim windows
    (few events) still surface their best events instead of being silenced by a
    global absolute cut.  The floor keeps pure-noise windows from contributing.
    """
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    # dense windows keep the strict absolute cut (do not flood bright sensors);
    # only sparse, dim-object windows get the relaxed percentile + floor gate.
    if not cfg.use_percentile_gate or scores.size > cfg.sparse_window_max:
        return scores >= cfg.score_threshold
    thr = max(np.percentile(scores, cfg.keep_percentile), cfg.score_floor)
    return scores >= thr


def window_candidates(win_idx: int, ws: int, we: int,
                      x: np.ndarray, y: np.ndarray, scores: np.ndarray,
                      sensor: Sensor, cfg: Config) -> list[Candidate]:
    """Permissive per-window proposals: cluster gated events and return up to
    ``max_candidates`` clusters as Candidates (extent + evidence).
    """
    keep = _gate(scores, cfg)
    if keep.sum() < cfg.min_cluster_events:
        return []
    xs, ys, ss = x[keep], y[keep], scores[keep]
    diag = sensor.diag
    labels = _cluster(np.column_stack([xs / diag, ys / diag]),
                      cfg.cluster_eps_norm, cfg.cluster_min_samples)

    cands = []
    for lab in set(labels.tolist()):
        if lab == -1:
            continue
        m = labels == lab
        n = int(m.sum())
        if n < cfg.min_cluster_events:
            continue
        bx, by, bs = xs[m], ys[m], ss[m]
        # score-weighted centroid: tighter, more accurate center than the
        # extent midpoint (which a single stray event can drag off).
        wsum = float(bs.sum()) + 1e-9
        cx = float((bx * bs).sum() / wsum)
        cy = float((by * bs).sum() / wsum)
        cands.append(Candidate(
            win_idx, ws, we, cx, cy,
            float(bx.min()), float(by.min()), float(bx.max()), float(by.max()),
            n, float(bs.mean())))
    # strongest first; cap the number considered by the tracker
    cands.sort(key=lambda c: c.n * c.score, reverse=True)
    return cands[: cfg.max_candidates]


# --------------------------------------------------------------------------- #
#  Tracking
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    members: list = field(default_factory=list)   # list[Candidate]
    cx: float = 0.0
    cy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    last_win: int = -1
    miss: int = 0

    def predict(self, win_idx: int):
        dw = win_idx - self.last_win
        return self.cx + self.vx * dw, self.cy + self.vy * dw

    def add(self, c: Candidate):
        if self.members:
            dw = max(c.win_idx - self.last_win, 1)
            nvx = (c.cx - self.cx) / dw
            nvy = (c.cy - self.cy) / dw
            # exponential smoothing of velocity
            self.vx = 0.6 * self.vx + 0.4 * nvx
            self.vy = 0.6 * self.vy + 0.4 * nvy
        self.cx, self.cy = c.cx, c.cy
        self.last_win = c.win_idx
        self.members.append(c)
        self.miss = 0

    # -- track-level summaries ------------------------------------------- #
    @property
    def length(self) -> int:
        return len(self.members)

    @property
    def span(self) -> int:
        return self.members[-1].win_idx - self.members[0].win_idx + 1 if self.members else 0

    def net_displacement(self) -> float:
        if len(self.members) < 2:
            return 0.0
        a, b = self.members[0], self.members[-1]
        return float(np.hypot(b.cx - a.cx, b.cy - a.cy))

    def evidence(self) -> float:
        return float(sum(c.n * c.score for c in self.members))

    def motion_residual(self) -> float:
        """RMS deviation of the track's centers from a straight constant-
        velocity path, in pixels.  Smooth real motion -> small residual;
        random-walk noise -> large residual."""
        if len(self.members) < 3:
            return 0.0
        wi = np.array([c.win_idx for c in self.members], dtype=np.float64)
        cx = np.array([c.cx for c in self.members], dtype=np.float64)
        cy = np.array([c.cy for c in self.members], dtype=np.float64)
        A = np.column_stack([wi, np.ones_like(wi)])
        rx = cx - A @ np.linalg.lstsq(A, cx, rcond=None)[0]
        ry = cy - A @ np.linalg.lstsq(A, cy, rcond=None)[0]
        return float(np.sqrt((rx ** 2 + ry ** 2).mean()))

    def fit_cv(self):
        """Robust constant-velocity fit cx(k)=cx0+vx·(k-k0) (pixels, window-time
        axis) via 2 IRLS passes so one bad candidate can't tilt the line.
        Returns (cx0, cy0, vx, vy, k0) for extrapolation."""
        wi = np.array([c.win_idx for c in self.members], dtype=np.float64)
        cx = np.array([c.cx for c in self.members], dtype=np.float64)
        cy = np.array([c.cy for c in self.members], dtype=np.float64)
        k0 = wi.mean()
        A = np.column_stack([np.ones_like(wi), wi - k0])
        w = np.ones_like(wi)
        bx = by = None
        for _ in range(2):
            W = np.sqrt(w)[:, None]
            bx = np.linalg.lstsq(A * W, cx * np.sqrt(w), rcond=None)[0]
            by = np.linalg.lstsq(A * W, cy * np.sqrt(w), rcond=None)[0]
            r = np.hypot(cx - A @ bx, cy - A @ by)
            mad = np.median(np.abs(r)) + 1e-9
            w = 1.0 / (1.0 + (r / (3.0 * mad)) ** 2)
        return float(bx[0]), float(by[0]), float(bx[1]), float(by[1]), float(k0)


def track_candidates(per_window: list[list[Candidate]], sensor: Sensor,
                     cfg: Config) -> list[Track]:
    """Greedy nearest-neighbor tracker with a constant-velocity gate."""
    gate = cfg.track_gate_norm * sensor.diag       # gate in pixels
    active: list[Track] = []
    closed: list[Track] = []

    for cands in per_window:
        if not cands:
            # age active tracks
            for tr in active:
                tr.miss += 1
            active = [t for t in active if t.miss <= cfg.track_max_gap or closed.append(t)]
            continue
        win_idx = cands[0].win_idx
        used = [False] * len(cands)
        # match existing tracks to nearest candidate within gate
        for tr in active:
            px, py = tr.predict(win_idx)
            best_j, best_d = -1, gate * (1 + tr.miss)
            for j, c in enumerate(cands):
                if used[j]:
                    continue
                d = np.hypot(c.cx - px, c.cy - py)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                tr.add(cands[best_j])
                used[best_j] = True
            else:
                tr.miss += 1
        # retire stale tracks
        still = []
        for tr in active:
            (still if tr.miss <= cfg.track_max_gap else closed).append(tr)
        active = still
        # spawn new tracks from unused candidates
        for j, c in enumerate(cands):
            if not used[j]:
                tr = Track()
                tr.add(c)
                active.append(tr)

    closed.extend(active)
    return closed


def _keep_track(tr: Track, cfg: Config) -> bool:
    if tr.length < cfg.min_track_len:
        return False
    # must move (rejects static hot-pixel clutter) and move *smoothly*
    # (rejects random-walk noise that happens to drift) — unless the evidence
    # is overwhelming.
    moving = tr.net_displacement() >= cfg.min_track_disp_px
    smooth = tr.motion_residual() <= cfg.max_track_residual_px
    strong = tr.evidence() >= cfg.strong_track_evidence
    return (moving and smooth) or strong


def tracks_to_detections(tracks: list[Track], sensor: Sensor, cfg: Config,
                         n_windows: int | None = None) -> list[Detection]:
    """Convert surviving tracks into one box per window.

    Per track: emit observed boxes, interpolate interior gaps, and — the recall
    lever — extrapolate a fitted constant-velocity model up to ``track_extend``
    windows past each end (and across gaps) with decaying confidence, recovering
    sub-threshold windows where a *confirmed* object is still present.  Box sizes
    get the per-sensor IoU calibration.  Conflicts resolve to the stronger track.
    """
    kept = [t for t in tracks if _keep_track(t, cfg)]
    kept.sort(key=lambda t: t.evidence(), reverse=True)
    # track budget scales with sequence length (long sequences host many objects
    # over time; a small fixed cap silently throttles their recall).
    n_win = n_windows or 0
    budget = int(cfg.max_tracks_per_seq + cfg.tracks_per_kwin * n_win / 1000.0)
    if kept:
        best_ev = kept[0].evidence()
        kept = [t for t in kept[: budget]
                if t.evidence() >= cfg.track_evidence_frac * best_ev]

    win_max = (n_windows - 1) if n_windows else (1 << 30)
    sw, sh = cfg.box_size_scale.get(sensor.name, (1.0, 1.0))
    typ = cfg.box_size_px.get(sensor.name)         # (w, h) typical GT box, or None

    def _size(ext_w, ext_h):
        """Final emitted box size in pixels.

        If a per-sensor typical GT size is known, blend the measured extent
        toward it (dim windows with tiny extent then emit a sensibly-sized box
        that clears IoU>=0.5).  Otherwise fall back to extent+margin with the
        multiplicative scale correction.  The two are mutually exclusive so size
        is never double-corrected.
        """
        ew = max(ext_w + 2 * cfg.box_margin_px, cfg.min_box_px)
        eh = max(ext_h + 2 * cfg.box_margin_px, cfg.min_box_px)
        if typ is None:
            return ew * sw, eh * sh
        # sensor-adaptive blend toward the typical GT size (see config)
        if isinstance(cfg.box_size_blend, dict):
            b = cfg.box_size_blend.get(sensor.name, cfg.box_size_blend_default)
        else:
            b = cfg.box_size_blend
        return (1 - b) * ew + b * typ[0], (1 - b) * eh + b * typ[1]

    by_win: dict[int, Detection] = {}
    for rank, tr in enumerate(kept):
        members = {c.win_idx: c for c in tr.members}
        obs = sorted(members)
        w0, w1 = obs[0], obs[-1]
        med_w = float(np.median([c.x2 - c.x1 for c in tr.members])) + 2 * cfg.box_margin_px
        med_h = float(np.median([c.y2 - c.y1 for c in tr.members])) + 2 * cfg.box_margin_px
        max_ev = max(c.n * c.score for c in tr.members) + 1e-9
        mean_sc = float(np.mean([c.score for c in tr.members]))
        win_us = tr.members[0].we - tr.members[0].ws
        ref = tr.members[0]                       # anchors window_idx -> ws
        cv = tr.fit_cv() if tr.length >= cfg.extend_min_core else None

        lo, hi = w0, w1
        if cv is not None:
            lo = max(0, w0 - cfg.track_extend)
            hi = min(win_max, w1 + cfg.track_extend)

        for wi in range(lo, hi + 1):
            if wi in members:
                c = members[wi]
                w, h = _size(c.x2 - c.x1, c.y2 - c.y1)
                conf = float(min(1.0, (c.n * c.score / max_ev) *
                                 (1 - np.exp(-tr.length / 6.0))))
                ws, we, cx, cy = c.ws, c.we, c.cx, c.cy
            elif w0 < wi < w1:
                # interior gap: linear interpolation between observed neighbors
                prev = max(k for k in obs if k < wi)
                nxt = min(k for k in obs if k > wi)
                a, b = members[prev], members[nxt]
                f = (wi - prev) / (nxt - prev)
                cx, cy = a.cx + f * (b.cx - a.cx), a.cy + f * (b.cy - a.cy)
                w, h = _size(med_w - 2 * cfg.box_margin_px, med_h - 2 * cfg.box_margin_px)
                ws = ref.ws + (wi - ref.win_idx) * win_us
                we = ws + win_us
                conf = 0.25
            else:
                # extrapolation beyond the observed ends (constant velocity)
                cx0, cy0, vx, vy, k0 = cv
                cx, cy = cx0 + vx * (wi - k0), cy0 + vy * (wi - k0)
                if not (0 <= cx <= sensor.width - 1 and 0 <= cy <= sensor.height - 1):
                    continue                      # center left the focal plane
                w, h = _size(med_w - 2 * cfg.box_margin_px, med_h - 2 * cfg.box_margin_px)
                ws = ref.ws + (wi - ref.win_idx) * win_us
                we = ws + win_us
                dist = (w0 - wi) if wi < w0 else (wi - w1)
                conf = float(mean_sc * (cfg.extend_conf_decay ** dist))

            if rank > 0 and wi in by_win:
                continue                          # stronger track already owns it
            by_win[wi] = _finalize(cx, cy, w, h, conf, ws, we, sensor)
    return [by_win[k] for k in sorted(by_win)]


def _finalize(cx, cy, w, h, conf, ws, we, sensor: Sensor) -> Detection:
    W, H = sensor.width, sensor.height
    return Detection(
        int(ws), int(we),
        int(round(np.clip(cx, 0, W - 1))), int(round(np.clip(cy, 0, H - 1))),
        int(round(np.clip(w, 1, W))), int(round(np.clip(h, 1, H))),
        float(np.clip(conf, 0, 1)))
