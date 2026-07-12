"""Event-level data augmentation for training the deep detectors.

Operates in NORMALIZED coordinates (x, y in [0, 1]; box = cx, cy, w, h in [0, 1])
so it is sensor-agnostic, and is applied *before* voxelization.  The headline
augmentation is **event-drop** (random subsampling), which simulates dimmer
objects / lower event-rate regimes — directly the H2 domain-shift strategy the
Technical Report prescribes for the dim-EVK4 / sparse-DVX test gap.

All ops keep the GT box consistent with the transformed events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugCfg:
    flip: float = 0.5            # P(horizontal flip), P(vertical flip)
    drop_p: float = 0.6         # P(event-drop)
    drop_min: float = 0.3       # min keep-fraction when dropping (dim aug)
    translate: float = 0.5      # P(translate)
    translate_max: float = 0.12  # max shift (fraction of frame)
    scale: float = 0.4          # P(scale)
    scale_lo: float = 0.75
    scale_hi: float = 1.3
    noise: float = 0.4          # P(noise injection)
    noise_frac: float = 0.6     # injected count as fraction of current events
    polflip: float = 0.2        # P(polarity flip)
    time_jitter: float = 0.3    # P(timestamp jitter)
    jitter_frac: float = 0.05   # jitter as fraction of window duration


def augment(xn, yn, pol, t, box, ws, we, rng: np.random.Generator,
            cfg: AugCfg = AugCfg()):
    """Augment one window.  ``box`` is (cx, cy, w, h) normalized or None.
    Returns transformed (xn, yn, pol, t, box)."""
    xn = xn.astype(np.float64).copy()
    yn = yn.astype(np.float64).copy()
    pol = pol.astype(np.int64).copy()
    t = t.astype(np.float64).copy()
    has = box is not None
    if has:
        cx, cy, bw, bh = box
    else:
        cx = cy = bw = bh = 0.0

    # --- horizontal / vertical flips ------------------------------------- #
    if rng.random() < cfg.flip:
        xn = 1.0 - xn
        if has:
            cx = 1.0 - cx
    if rng.random() < cfg.flip:
        yn = 1.0 - yn
        if has:
            cy = 1.0 - cy

    # --- event-drop (dim augmentation) ----------------------------------- #
    if rng.random() < cfg.drop_p and xn.size > 8:
        frac = rng.uniform(cfg.drop_min, 1.0)
        keep = rng.random(xn.size) < frac
        xn, yn, pol, t = xn[keep], yn[keep], pol[keep], t[keep]

    # --- scale about box center (or frame center) ------------------------ #
    if rng.random() < cfg.scale:
        s = rng.uniform(cfg.scale_lo, cfg.scale_hi)
        ox, oy = (cx, cy) if has else (0.5, 0.5)
        xn = ox + (xn - ox) * s
        yn = oy + (yn - oy) * s
        if has:
            bw *= s
            bh *= s

    # --- translate ------------------------------------------------------- #
    if rng.random() < cfg.translate:
        dx = rng.uniform(-cfg.translate_max, cfg.translate_max)
        dy = rng.uniform(-cfg.translate_max, cfg.translate_max)
        xn = xn + dx
        yn = yn + dy
        if has:
            cx += dx
            cy += dy

    # keep events inside the frame
    m = (xn >= 0) & (xn < 1) & (yn >= 0) & (yn < 1)
    xn, yn, pol, t = xn[m], yn[m], pol[m], t[m]

    # --- background-noise injection -------------------------------------- #
    if rng.random() < cfg.noise and t.size:
        k = int(rng.uniform(0, cfg.noise_frac) * max(xn.size, 50))
        if k > 0:
            nx = rng.uniform(0, 1, k)
            ny = rng.uniform(0, 1, k)
            npl = rng.integers(0, 2, k)
            tlo, thi = (t.min(), t.max()) if t.size else (ws, we)
            nt = rng.uniform(tlo, thi + 1, k)
            xn = np.concatenate([xn, nx]); yn = np.concatenate([yn, ny])
            pol = np.concatenate([pol, npl]); t = np.concatenate([t, nt])

    # --- polarity flip / time jitter ------------------------------------ #
    if rng.random() < cfg.polflip:
        pol = 1 - pol
    if rng.random() < cfg.time_jitter and t.size:
        dur = max(int(we - ws), 1)
        t = t + rng.normal(0, cfg.jitter_frac * dur, t.size)

    # box: drop if it left the frame, else clip center
    if has:
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            box = None
        else:
            box = (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)),
                   float(np.clip(bw, 1e-3, 1.0)), float(np.clip(bh, 1e-3, 1.0)))
    return xn, yn, pol, t, box
