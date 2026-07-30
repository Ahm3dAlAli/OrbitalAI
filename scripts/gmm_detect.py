#!/usr/bin/env python3
"""Event-based GMM weak-target detector (Dong et al., CyberC 2023) for OrbitSight.

The paper detects weak space targets by (1) Background-Activity filtering then (2)
fitting a Gaussian Mixture Model to the per-window event cloud — each real target is a
dense Gaussian blob, distinguishable from diffuse noise. Their BAF == our Stage-1
denoise; the GMM is a *different* (unsupervised, density-based) detector than our learned
CenterNet, so it has different failure modes.

Two modes:

  standalone  : fit a GMM per 40 ms window, emit the strongest component(s) as boxes.
                A/B its AP against the CenterNet.

  --gate CN   : keep the CenterNet's (well-calibrated) boxes but RE-RANK them by whether
                a GMM cluster supports each detection — a phantom the CenterNet hallucinates
                with no real event cluster gets demoted. Attacks the precision floor
                (Thuraya3 precision ~0.43) using a detector with independent errors.

    # standalone:
    python3 scripts/gmm_detect.py --data-dir OrbitSight_Dataset/Testing_sets \
        --out-dir /tmp/gmm --sequences DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 \
        --gt-dir OrbitSight_Dataset/Testing_sets
    # agreement gate over existing CenterNet preds:
    python3 scripts/gmm_detect.py --data-dir OrbitSight_Dataset/Testing_sets \
        --gate /tmp/th_raw --out-dir /tmp/gmm_gate --beta 0.5 \
        --sequences DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 \
        --gt-dir OrbitSight_Dataset/Testing_sets
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid sklearn OpenMP clash

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.features import build_cloud
from sklearn.mixture import GaussianMixture


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


def window_components(xs, ys, k, reg):
    """Fit a GMM to (x,y) pixels; return components (cx, cy, weight, n, density)."""
    n = xs.size
    if n < 3:
        return []
    pts = np.stack([xs, ys], 1).astype(np.float64)
    kk = int(min(k, max(1, n // 3)))
    try:
        gm = GaussianMixture(kk, covariance_type="full", reg_covar=reg,
                             init_params="kmeans", max_iter=60, n_init=1)
        gm.fit(pts)
        lab = gm.predict(pts)
    except Exception:
        return []
    comps = []
    for j in range(kk):
        m = lab == j
        nj = int(m.sum())
        if nj < 2:
            continue
        cov = gm.covariances_[j]
        area = float(np.sqrt(max(np.linalg.det(cov), 1.0)))   # spread of the blob
        dens = nj / (area + 1e-6)                             # events per unit area
        comps.append((float(gm.means_[j, 0]), float(gm.means_[j, 1]),
                      float(gm.weights_[j]), nj, dens))
    return comps


_RNG = np.random.default_rng(0)


def window_xy(ev, w, sensor, cfg, do_denoise, max_events):
    """Return (x,y) pixels for the window, event-capped for speed, BAF-denoised."""
    lo, hi = w.lo, w.hi
    n = hi - lo
    if n > max_events:                          # cap: bounds KD-tree + GMM cost
        sel = np.sort(_RNG.choice(n, max_events, replace=False)) + lo
        xs, ys, ts, ps = ev.x[sel], ev.y[sel], ev.t[sel], ev.pol[sel]
    else:
        xs, ys, ts, ps = ev.x[lo:hi], ev.y[lo:hi], ev.t[lo:hi], ev.pol[lo:hi]
    if not do_denoise or xs.size < 3:
        return xs, ys
    cloud = build_cloud(xs, ys, ts, ps, sensor, cfg)
    keep = cloud.denoise_keep(0, xs.size)       # BAF: >= support neighbors in (x,y,t) ball
    return xs[keep], ys[keep]


def run_sequence(ev, seq, cfg, args, cn_rows=None):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    box = args.box_px
    out = []
    for w in wins:
        # gate mode: only fit a GMM where there's a CenterNet box to verify (huge speedup)
        cn_here = cn_rows.get(w.start_us, []) if cn_rows is not None else None
        if cn_rows is not None and not cn_here:
            continue
        xs, ys = window_xy(ev, w, sn, cfg, args.denoise, args.max_events)
        comps = window_components(xs, ys, args.k, args.reg)
        if cn_rows is None:
            # ---- standalone: emit the strongest component(s) as boxes ------- #
            comps.sort(key=lambda c: c[2] * c[4], reverse=True)   # weight x density
            for (cx, cy, wt, nj, dens) in comps[:args.topk]:
                conf = float(np.clip(wt * np.log1p(nj) / 3.0, 1e-3, 1.0))
                out.append(D.Detection(w.start_us, w.end_us,
                    int(round(np.clip(cx, 0, sn.width - 1))),
                    int(round(np.clip(cy, 0, sn.height - 1))),
                    box, box, conf))
        else:
            # ---- gate mode: re-rank the CenterNet boxes by GMM support ------- #
            for (cx, cy, bw, bh, cf) in cn_here:
                near = 0.0
                for (gx, gy, wt, nj, dens) in comps:
                    if abs(gx - cx) <= args.gate_px and abs(gy - cy) <= args.gate_px:
                        near = max(near, wt)                      # supported by a cluster
                supp = near / (near + 0.15)                        # 0..1 support score
                nc = float(np.clip(cf ** (1 - args.beta) * (supp + 0.05) ** args.beta,
                                   1e-4, 1.0))
                out.append(D.Detection(w.start_us, w.end_us, int(cx), int(cy),
                    int(bw), int(bh), nc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--gate", default=None,
                    help="CenterNet pred dir -> re-rank those boxes by GMM support "
                         "(instead of standalone GMM detection)")
    ap.add_argument("--k", type=int, default=3, help="max GMM components per window")
    ap.add_argument("--reg", type=float, default=2.0, help="covariance regularization (px^2)")
    ap.add_argument("--topk", type=int, default=1, help="standalone: components kept/window")
    ap.add_argument("--box-px", type=int, default=11, help="standalone: emitted box size (px)")
    ap.add_argument("--gate-px", type=float, default=12.0, help="gate: cluster-to-box match radius")
    ap.add_argument("--beta", type=float, default=0.5, help="gate: 0=no change, 1=rank by support")
    ap.add_argument("--denoise", action="store_true", help="apply BAF denoise before GMM")
    ap.add_argument("--max-events", type=int, default=400,
                    help="cap events/window fed to the GMM (subsample; bounds cost). "
                         "A target blob needs few points, so this is near-lossless.")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG

    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for seq in seqs:
        ep = D.find_event_file(args.data_dir, seq)
        if not ep:
            continue
        cn_rows = read_rows(os.path.join(args.gate, seq + D.GT_SUFFIX)) if args.gate else None
        ev = D.Events.from_npy(ep)
        dets = run_sequence(ev, seq, cfg, args, cn_rows)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        mode = "gate" if args.gate else "standalone"
        print(f"  [{mode}] {seq[:38]:38s} -> {len(dets)} boxes")
        del ev

    if args.gt_dir:
        os.system(f'python3 scripts/evaluate_wrapper.py --dataset '
                  f'"{os.path.dirname(args.gt_dir.rstrip("/"))}" --pred-dir "{args.out_dir}" '
                  f'--excel-out "{args.out_dir}/m.xlsx" 2>/dev/null '
                  f'| grep -E "Thuraya3|Stars3|SAOCOM|EVK4_mag7.3|mAP @"')


if __name__ == "__main__":
    main()
