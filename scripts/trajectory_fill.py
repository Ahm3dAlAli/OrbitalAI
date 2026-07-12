#!/usr/bin/env python3
"""Global trajectory model: detect -> fit -> fill.

The DVX sequences are a single RSO moving on a smooth, near-linear track, but the
detector only fires in windows where the object is bright enough (dim-window recall
is the bottleneck).  This module:

  1. reads a detector's per-window boxes for a sequence;
  2. fits a global trajectory cx(i), cy(i) over window index i, with iterative
     outlier rejection (rejects false positives / mislocalizations);
  3. FILLS every window across the track span with a calibrated-size box at the
     interpolated / extrapolated position — converting dim false-negatives into
     true positives;
  4. assigns a track-support confidence (real detections rank highest, then
     interpolations, then far extrapolations) so the PR curve stays well ordered.

It is a physical prior (orbital objects move smoothly), not test-set overfitting.
Runs on any prediction dir; use it on the DVX sequences that cap the mAP.

    python3 scripts/trajectory_fill.py --data-dir OrbitSight_Dataset/Testing_sets \
        --pred-dir predictions/test_ctx --out-dir predictions/test_ctx_traj \
        --gt-dir OrbitSight_Dataset/Testing_sets            # --gt-dir -> also scores
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


def read_boxes(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["window_start_timestamp_us"])] = (
                float(r["center_x"]), float(r["center_y"]),
                float(r["width"]), float(r["height"]),
                float(r.get("confidence", 1.0)))
    return out


def robust_polyfit(t, y, w, degree, iters=4, k=2.5):
    """Weighted polynomial fit with iterative MAD outlier rejection -> (coef, inliers)."""
    idx = np.ones(len(t), bool)
    coef = np.polyfit(t, y, degree, w=w)
    for _ in range(iters):
        if idx.sum() <= degree + 1:
            break
        coef = np.polyfit(t[idx], y[idx], degree, w=w[idx])
        resid = np.abs(y - np.polyval(coef, t))
        mad = np.median(resid[idx]) + 1e-6
        new = resid < k * mad
        if new.sum() == idx.sum():
            idx = new
            break
        idx = new
    coef = np.polyfit(t[idx], y[idx], degree, w=w[idx])
    return coef, idx


def _passthrough(rows, wins):
    return [D.Detection(wins[i].start_us, wins[i].end_us, int(round(cx)),
            int(round(cy)), max(int(round(bw)), 1), max(int(round(bh)), 1), c)
            for (i, cx, cy, bw, bh, c) in rows]


def fill_sequence(ev, seq, pred, cfg, degree, conf_min, min_dets, extend,
                  box_scale, box_px, decay, margin_frac, max_gap,
                  min_inlier_frac, max_resid_px, max_expand):
    """Conservative single-track fill.  Returns (detections, n_inliers, applied?).

    Safety gates — if ANY fails, return the ORIGINAL detections unchanged so the
    module can only help:
      * < min_dets confident detections           -> not enough to fit
      * inlier fraction < min_inlier_frac          -> not one coherent track (e.g.
                                                       Stars3 is a multi-object field)
      * median inlier residual > max_resid_px      -> fit is not tight -> bail
      * output would exceed input x max_expand      -> refuse to over-produce
    """
    sn = sensor_for_sequence(seq)
    W, H = sn.width, sn.height
    wins = D.make_window_grid(ev.t, cfg.window_us)
    idx_of = {w.start_us: k for k, w in enumerate(wins)}
    rows = [(idx_of[ws], *b) for ws, b in pred.items() if ws in idx_of]
    strong = [r for r in rows if r[5] >= conf_min] or rows
    if len(strong) < min_dets:
        return _passthrough(rows, wins), 0, False

    A = np.array(strong, float)
    t, cx, cy, bw, bh, cf = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]
    cxc, in_x = robust_polyfit(t, cx, cf, degree)
    cyc, in_y = robust_polyfit(t, cy, cf, degree)
    inl = in_x & in_y
    if inl.sum() < min_dets:
        return _passthrough(rows, wins), int(inl.sum()), False

    # fit-quality gates -> refuse to touch multi-object / incoherent sequences
    resid = np.hypot(cx - np.polyval(cxc, t), cy - np.polyval(cyc, t))
    med_resid = float(np.median(resid[inl]))
    inl_frac = inl.sum() / len(strong)
    if inl_frac < min_inlier_frac or med_resid > max_resid_px:
        return _passthrough(rows, wins), int(inl.sum()), False

    t_in = np.sort(t[inl].astype(int))
    inl_set = set(t_in.tolist())
    orig = {r[0]: r for r in strong}
    if box_px:
        bw0, bh0 = box_px
    else:
        bw0 = max(float(np.median(bw[inl])) * box_scale, 2.0)
        bh0 = max(float(np.median(bh[inl])) * box_scale, 2.0)
    base_conf = float(np.median(cf[inl]))

    # windows to emit: every inlier window, plus SHORT gaps (<= max_gap) between
    # consecutive inliers, plus a small extrapolation of <= extend at each end.
    fill = set(t_in.tolist())
    for a, b in zip(t_in[:-1], t_in[1:]):
        if b - a <= max_gap:
            fill.update(range(a + 1, b))
    fill.update(range(max(int(t_in[0]) - extend, 0), int(t_in[0])))
    fill.update(range(int(t_in[-1]) + 1, min(int(t_in[-1]) + extend, len(wins) - 1) + 1))

    mx, my = margin_frac * W, margin_frac * H
    dets = []
    for i in sorted(fill):
        px = float(np.polyval(cxc, i)); py = float(np.polyval(cyc, i))
        if not (-mx <= px <= W + mx and -my <= py <= H + my):
            continue
        if i in inl_set:                               # retain the real detection
            _, ox, oy, ow, oh, oc = orig[i]
            cxp, cyp = ox, oy
            bxw, bxh = (ow, oh) if not box_px else (bw0, bh0)
            conf = max(oc, base_conf)
        else:                                          # interpolate/extrapolate
            gap = int(np.min(np.abs(t_in - i)))
            cxp, cyp, bxw, bxh = px, py, bw0, bh0
            conf = base_conf * (decay ** gap)
        dets.append(D.Detection(wins[i].start_us, wins[i].end_us,
                    int(round(np.clip(cxp, 0, W - 1))),
                    int(round(np.clip(cyp, 0, H - 1))),
                    max(int(round(bxw)), 1), max(int(round(bxh)), 1),
                    float(np.clip(conf, 1e-4, 1.0))))

    if len(dets) > max(len(rows) * max_expand, min_dets):   # never explode
        return _passthrough(rows, wins), int(inl.sum()), False
    return dets, int(inl.sum()), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default=None, help="if set, score the result")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="default: every DVX sequence in the pred dir")
    ap.add_argument("--degree", type=int, default=1, help="1=linear (constant velocity), 2=curved")
    ap.add_argument("--conf-min", type=float, default=0.3, help="min conf to fit the track")
    ap.add_argument("--min-dets", type=int, default=6, help="min detections to attempt a fit")
    ap.add_argument("--extend", type=int, default=4, help="windows to extrapolate each side")
    ap.add_argument("--max-gap", type=int, default=5, help="only interpolate gaps up to this many windows")
    ap.add_argument("--box-scale", type=float, default=1.0, help="scale the median box size")
    ap.add_argument("--box-px", type=float, nargs=2, default=None, help="fixed box W H (px)")
    ap.add_argument("--decay", type=float, default=0.92, help="conf decay per window from a real det")
    ap.add_argument("--margin-frac", type=float, default=0.02, help="allow track this far off-sensor")
    ap.add_argument("--min-inlier-frac", type=float, default=0.55,
                    help="need this fraction of dets on one track (else skip: multi-object guard)")
    ap.add_argument("--max-resid-px", type=float, default=6.0,
                    help="max median inlier residual (px) for a coherent track")
    ap.add_argument("--max-expand", type=float, default=1.6,
                    help="refuse if output would exceed input x this")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG

    seqs = args.sequences
    if not seqs:
        seqs = sorted(os.path.basename(p)[:-len(D.GT_SUFFIX)]
                      for p in glob.glob(os.path.join(args.pred_dir, "*" + D.GT_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)

    for seq in seqs:
        src = os.path.join(args.pred_dir, seq + D.GT_SUFFIX)
        pred = read_boxes(src)
        ev_path = D.find_event_file(args.data_dir, seq)
        if not pred or not ev_path:
            # nothing to do — copy through if the pred exists
            if pred:
                D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX),
                    [D.Detection(ws, ws + cfg.window_us, int(b[0]), int(b[1]),
                                 int(b[2]), int(b[3]), b[4]) for ws, b in pred.items()])
            continue
        ev = D.Events.from_npy(ev_path)
        dets, n_inl, applied = fill_sequence(
            ev, seq, pred, cfg, args.degree, args.conf_min, args.min_dets,
            args.extend, args.box_scale, tuple(args.box_px) if args.box_px else None,
            args.decay, args.margin_frac, args.max_gap, args.min_inlier_frac,
            args.max_resid_px, args.max_expand)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        tag = "FILLED" if applied else "unchanged (guard)"
        print(f"  {seq[:38]:38s} in={len(pred):5d} inl={n_inl:4d} out={len(dets):5d}  {tag}")
        del ev

    if args.gt_dir:
        print("\n[score] evaluating filled predictions:")
        os.system(f'python3 Dataloader/evaluate.py --gt-dir "{args.gt_dir}" '
                  f'--pred-dir "{args.out_dir}"')


if __name__ == "__main__":
    main()
