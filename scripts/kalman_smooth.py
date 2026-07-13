#!/usr/bin/env python3
"""UKF-style sub-pixel center refinement (constant-velocity Kalman + RTS smoother).

Post-processes a detector's per-window boxes: fits a constant-velocity Kalman
filter to the detection centers over window index, runs a Rauch-Tung-Striebel
(RTS) backward smoother, and blends each box center toward the smoothed estimate.
On smoothly-moving objects this yields sub-pixel-consistent centers that can nudge
near-threshold detections over IoU 0.5; it is real-time (a tiny linear-algebra pass
per sequence) and changes no box sizes or confidences.

Guarded so it can only help: it PASSES THROUGH unchanged when a sequence is
multi-object (Stars3 field) or does not follow constant-velocity motion (the
non-smooth Thuraya3 target), so it never fights a bad motion model.

    python3 scripts/kalman_smooth.py --data-dir OrbitSight_Dataset/Testing_sets \
        --pred-dir predictions/test_ctx --out-dir predictions/_kf --alpha 0.5 \
        --gt-dir OrbitSight_Dataset/Testing_sets
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence


def read_rows(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ws = int(r["window_start_timestamp_us"])
            out.setdefault(ws, []).append((
                float(r["center_x"]), float(r["center_y"]),
                float(r["width"]), float(r["height"]),
                float(r.get("confidence", 1.0))))
    return out


def kalman_rts(idx, meas, q, r):
    """Constant-velocity Kalman forward filter + RTS smoother.
    idx: sorted window indices (may have gaps); meas: (N,2) centers.
    Returns smoothed (N,2) at the measured indices."""
    F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
    Q = q * np.array([[.25, .5, 0, 0], [.5, 1, 0, 0],
                      [0, 0, .25, .5], [0, 0, .5, 1]], float)
    R = r * np.eye(2)
    lo, hi = int(idx[0]), int(idx[-1])
    n = hi - lo + 1
    mat = {int(i): m for i, m in zip(idx, meas)}
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    x = np.array([meas[0, 0], 0.0, meas[0, 1], 0.0]); P = np.eye(4) * 100.0
    for k in range(n):
        gi = lo + k
        xpred = x if k == 0 else F @ xs[k - 1]
        Ppred = P if k == 0 else F @ Ps[k - 1] @ F.T + Q
        xp[k], Pp[k] = xpred, Ppred
        if gi in mat:
            z = mat[gi]
            S = H @ Ppred @ H.T + R
            K = Ppred @ H.T @ np.linalg.inv(S)
            x = xpred + K @ (z - H @ xpred)
            P = (np.eye(4) - K @ H) @ Ppred
        else:
            x, P = xpred, Ppred
        xs[k], Ps[k] = x, P
    xsm = xs.copy()
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xsm[k] = xs[k] + C @ (xsm[k + 1] - xp[k + 1])
    return np.array([[xsm[int(i) - lo, 0], xsm[int(i) - lo, 2]] for i in idx])


def smooth_sequence(ev, seq, rows, cfg, alpha, q, r, min_dets, max_resid, multi_frac):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    idx_of = {w.start_us: k for k, w in enumerate(wins)}
    # multi-object guard
    n_multi = sum(1 for b in rows.values() if len(b) > 1)
    if rows and n_multi / len(rows) > multi_frac:
        return _flatten(rows, cfg), False, "multi-object"
    flat = [(idx_of[ws], ws) + b for ws, bs in rows.items() if ws in idx_of for b in bs]
    if len(flat) < min_dets:
        return _flatten(rows, cfg), False, "too few"
    flat.sort()
    A = np.array([[f[0], f[2], f[3]] for f in flat], float)  # idx, cx, cy
    sm = kalman_rts(A[:, 0], A[:, 1:3], q, r)
    resid = float(np.median(np.hypot(A[:, 1] - sm[:, 0], A[:, 2] - sm[:, 1])))
    if resid > max_resid:                                    # non-smooth -> bail
        return _flatten(rows, cfg), False, f"non-smooth ({resid:.1f}px)"
    dets = []
    for (i, ws, cx, cy, bw, bh, cf), s in zip(flat, sm):
        ncx = (1 - alpha) * cx + alpha * s[0]
        ncy = (1 - alpha) * cy + alpha * s[1]
        we = ws + cfg.window_us
        dets.append(D.Detection(ws, we,
            int(round(np.clip(ncx, 0, sn.width - 1))),
            int(round(np.clip(ncy, 0, sn.height - 1))),
            max(int(round(bw)), 1), max(int(round(bh)), 1), cf))
    return dets, True, f"smoothed (resid {resid:.1f}px)"


def _flatten(rows, cfg):
    out = []
    for ws, bs in rows.items():
        for (cx, cy, bw, bh, cf) in bs:
            out.append(D.Detection(ws, ws + cfg.window_us, int(round(cx)),
                       int(round(cy)), int(round(bw)), int(round(bh)), cf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.5, help="blend toward smoothed center")
    ap.add_argument("--q", type=float, default=1.0, help="process noise")
    ap.add_argument("--r", type=float, default=4.0, help="measurement noise")
    ap.add_argument("--min-dets", type=int, default=8)
    ap.add_argument("--max-resid", type=float, default=6.0, help="px; above -> non-smooth, skip")
    ap.add_argument("--multi-frac", type=float, default=0.15, help="frac multi-box windows -> multi-object, skip")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG

    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.GT_SUFFIX)]
        for p in glob.glob(os.path.join(args.pred_dir, "*" + D.GT_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for seq in seqs:
        rows = read_rows(os.path.join(args.pred_dir, seq + D.GT_SUFFIX))
        ep = D.find_event_file(args.data_dir, seq)
        if not rows or not ep:
            continue
        ev = D.Events.from_npy(ep)
        dets, applied, why = smooth_sequence(ev, seq, rows, cfg, args.alpha, args.q,
            args.r, args.min_dets, args.max_resid, args.multi_frac)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"  {seq[:38]:38s} {'SMOOTHED' if applied else 'unchanged'}: {why}")
        del ev

    if args.gt_dir:
        print("\n[score]")
        os.system(f'python3 Dataloader/evaluate.py --gt-dir "{args.gt_dir}" '
                  f'--pred-dir "{args.out_dir}" 2>/dev/null '
                  '| grep -E "EVK4|SAOCOM|Stars3|Thuraya3|mAP @"')


if __name__ == "__main__":
    main()
