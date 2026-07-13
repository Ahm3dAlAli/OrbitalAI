#!/usr/bin/env python3
"""Coasting Kalman tracker — recall recovery through short sub-threshold gaps.

Thuraya3's bottleneck is recall (0.65): the object is present but the detector
drops below threshold for a few windows and the track is lost. This runs a
constant-velocity Kalman filter + RTS smoother over the detection centers and
COASTS through short gaps: at each empty window BETWEEN nearby detections it emits
a box at the filter's predicted position (reduced-but-nonzero confidence),
converting sub-threshold false-negatives into true positives.

Unlike trajectory_fill (which fit ONE global polynomial and rejected Thuraya3 for a
47 px global residual), this filter is LOCAL — the CV state adapts each step, so it
coasts correctly even when the full track is not a single polynomial (the detections
follow the local motion to ~1 px). Coasting is BOUNDED to gaps <= max_coast so it
never extrapolates into the long absences where the object truly leaves the frame.

    python3 scripts/kalman_coast.py --data-dir OrbitSight_Dataset/Testing_sets \
        --pred-dir predictions/test_ctx --out-dir predictions/_coast \
        --sequences DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 \
        --max-coast 4 --gt-dir OrbitSight_Dataset/Testing_sets
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


def kalman_full(idx, meas, q, r):
    """CV Kalman forward + RTS smoother; returns smoothed (x,y) at EVERY grid
    index in [idx[0], idx[-1]] (gaps included, via predict-only steps)."""
    F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
    Q = q * np.array([[.25, .5, 0, 0], [.5, 1, 0, 0],
                      [0, 0, .25, .5], [0, 0, .5, 1]], float)
    R = r * np.eye(2)
    lo, hi = int(idx[0]), int(idx[-1]); n = hi - lo + 1
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
            S = H @ Ppred @ H.T + R
            K = Ppred @ H.T @ np.linalg.inv(S)
            x = xpred + K @ (mat[gi] - H @ xpred)
            P = (np.eye(4) - K @ H) @ Ppred
        else:
            x, P = xpred, Ppred
        xs[k], Ps[k] = x, P
    xsm = xs.copy()
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xsm[k] = xs[k] + C @ (xsm[k + 1] - xp[k + 1])
    return lo, xsm[:, [0, 2]]


def coast_sequence(ev, seq, rows, cfg, q, r, min_dets, max_coast, extend,
                   decay, multi_frac):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    idx_of = {w.start_us: k for k, w in enumerate(wins)}
    win_of = {k: w for k, w in enumerate(wins)}
    n_multi = sum(1 for b in rows.values() if len(b) > 1)
    if rows and n_multi / len(rows) > multi_frac:
        return _flat(rows, cfg), 0, "multi-object (skip)"
    # one measurement per window: the highest-confidence detection
    det = {}
    for ws, bs in rows.items():
        if ws in idx_of:
            b = max(bs, key=lambda z: z[4])
            det[idx_of[ws]] = b
    if len(det) < min_dets:
        return _flat(rows, cfg), 0, "too few"
    di = np.array(sorted(det)); meas = np.array([[det[i][0], det[i][1]] for i in di], float)
    lo, sm = kalman_full(di, meas, q, r)
    bw = float(np.median([det[i][2] for i in di]))
    bh = float(np.median([det[i][3] for i in di]))
    base = float(np.median([det[i][4] for i in di]))
    det_set = set(di.tolist())

    # windows to emit: all real detections + coasted gap/extension windows
    emit = set(det_set)
    for a, b in zip(di[:-1], di[1:]):
        if b - a <= max_coast + 1:                       # bridge short interior gaps
            emit.update(range(a + 1, b))
    emit.update(range(max(int(di[0]) - extend, 0), int(di[0])))
    emit.update(range(int(di[-1]) + 1, min(int(di[-1]) + extend, len(wins) - 1) + 1))

    dets = []; n_coast = 0
    for i in sorted(emit):
        w = win_of.get(i)
        if w is None:
            continue
        if i in det_set:
            cx, cy, ow, oh, cf = det[i]
        else:
            pos = sm[i - lo] if 0 <= i - lo < len(sm) else None
            if pos is None:
                continue
            gap = int(np.min(np.abs(di - i)))
            cx, cy, ow, oh, cf = pos[0], pos[1], bw, bh, base * (decay ** gap)
            n_coast += 1
        dets.append(D.Detection(w.start_us, w.end_us,
            int(round(np.clip(cx, 0, sn.width - 1))),
            int(round(np.clip(cy, 0, sn.height - 1))),
            max(int(round(ow)), 1), max(int(round(oh)), 1),
            float(np.clip(cf, 1e-4, 1.0))))
    return dets, n_coast, f"coasted +{n_coast} windows"


def _flat(rows, cfg):
    return [D.Detection(ws, ws + cfg.window_us, int(round(b[0])), int(round(b[1])),
            int(round(b[2])), int(round(b[3])), b[4])
            for ws, bs in rows.items() for b in bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=4.0)
    ap.add_argument("--min-dets", type=int, default=8)
    ap.add_argument("--max-coast", type=int, default=4, help="bridge interior gaps up to this many windows")
    ap.add_argument("--extend", type=int, default=2, help="coast this many windows past each end")
    ap.add_argument("--decay", type=float, default=0.9, help="conf decay per coasted window")
    ap.add_argument("--multi-frac", type=float, default=0.15)
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
        dets, nc, why = coast_sequence(ev, seq, rows, cfg, args.q, args.r,
            args.min_dets, args.max_coast, args.extend, args.decay, args.multi_frac)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"  {seq[:38]:38s} in={sum(len(v) for v in rows.values()):5d} "
              f"out={len(dets):5d}  {why}")
        del ev

    if args.gt_dir:
        print("\n[score]")
        os.system(f'python3 Dataloader/evaluate.py --gt-dir "{args.gt_dir}" '
                  f'--pred-dir "{args.out_dir}" 2>/dev/null '
                  '| grep -E "Stars3|Thuraya3|SAOCOM|EVK4|mAP @"')


if __name__ == "__main__":
    main()
