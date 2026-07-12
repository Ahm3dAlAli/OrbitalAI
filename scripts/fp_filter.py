#!/usr/bin/env python3
"""Coherence / concentration false-positive re-ranker (Stars3 lever).

Multi-object sequences (Stars3 star field) have many false positives: the detector
fires where there is no real event concentration (~half the boxes are FP, precision
~0.49). AP depends only on the *ranking* of predictions by confidence, so if we
re-rank real detections above clutter we lift AP WITHOUT changing any box.

Signal (paper's H1 — coherence is the signal): a real object produces a local PEAK
of events at the box, whereas star clutter / noise is spatially flat. For each
predicted box we measure event **concentration** = density inside the box vs density
in the surrounding ring, then fold it into the confidence NON-monotonically:

    new_conf = orig_conf ** (1-beta)  *  concentration_norm ** beta

Boxes with a genuine peak keep/raise their rank; flat/empty boxes sink. This is a
real-time post-processing step (a few event-count queries per box).

    python3 scripts/fp_filter.py --data-dir OrbitSight_Dataset/Testing_sets \
        --pred-dir predictions/router_ctta --out-dir predictions/_fp \
        --sequences DVX_Filtered_Stars3_2025-01-20-20-22-53 --beta 0.5 \
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
    """window_start -> list of (cx, cy, w, h, conf)."""
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


def concentration(xw, yw, cx, cy, w, h):
    """Event density in the box vs the surrounding ring (peak-vs-flat).  >1 = peak."""
    hw, hh = w / 2.0, h / 2.0
    inb = (np.abs(xw - cx) <= hw) & (np.abs(yw - cy) <= hh)
    n_in = int(inb.sum())
    outb = (np.abs(xw - cx) <= 2 * hw) & (np.abs(yw - cy) <= 2 * hh)
    n_out = int(outb.sum())
    n_ring = n_out - n_in
    area_in = max(w * h, 1.0)
    area_ring = max(3.0 * w * h, 1.0)
    d_in = (n_in + 0.5) / area_in
    d_ring = (n_ring + 0.5) / area_ring
    return d_in / d_ring, n_in


def rescore_sequence(ev, seq, rows, cfg, beta, drop_empty):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    idx = {w.start_us: w for w in wins}
    out = []
    concs = []
    for ws, boxes in rows.items():
        w = idx.get(ws)
        if w is None:                       # keep unscored windows as-is
            for (cx, cy, bw, bh, c) in boxes:
                out.append((ws, ws + cfg.window_us, cx, cy, bw, bh, c))
            continue
        xw, yw = ev.x[w.lo:w.hi], ev.y[w.lo:w.hi]
        for (cx, cy, bw, bh, c) in boxes:
            conc, n_in = concentration(xw, yw, cx, cy, bw, bh)
            concs.append(conc)
            if drop_empty and n_in == 0:
                continue                     # pure-empty box -> definite FP
            cn = conc / (conc + 1.0)         # -> (0,1); peak~1, flat~0.5, empty~small
            new_c = (c ** (1 - beta)) * (cn ** beta)
            out.append((w.start_us, w.end_us, cx, cy, bw, bh,
                        float(np.clip(new_c, 1e-4, 1.0))))
    return out, (float(np.median(concs)) if concs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="0 = no change, 1 = confidence fully replaced by concentration")
    ap.add_argument("--drop-empty", action="store_true",
                    help="also drop boxes with zero events inside (definite FPs)")
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
        dets, med = rescore_sequence(ev, seq, rows, cfg, args.beta, args.drop_empty)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX),
            [D.Detection(*d) for d in dets])
        print(f"  {seq[:38]:38s} boxes {sum(len(v) for v in rows.values()):5d} "
              f"-> {len(dets):5d}  median_conc={med:.2f}")
        del ev

    if args.gt_dir:
        os.system(f'python3 Dataloader/evaluate.py --gt-dir "{args.gt_dir}" '
                  f'--pred-dir "{args.out_dir}" 2>/dev/null | grep -E "Stars3|Thuraya3|mAP @"')


if __name__ == "__main__":
    main()
