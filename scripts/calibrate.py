#!/usr/bin/env python3
"""Offline per-sensor box-size IoU calibration (recall fix #3).

Boxes can be systematically too large/small even when their *center* is correct,
which loses true positives at the IoU>=0.5 gate.  We learn a per-sensor
multiplicative (sw, sh) correction so emitted box sizes better match GT.

Chicken-and-egg note: match (pred, gt) pairs by **center distance**, not by
IoU>=0.5 — if boxes are badly sized, none clear IoU 0.5 and an IoU-gated fit
would have no samples.  Localization is correct even when size is wrong, so
center-distance matching yields the pairs needed to learn the correction.

Writes models/box_scales.json -> loaded automatically by scripts/infer.py.

Usage:
    python3 scripts/calibrate.py \
        --gt-dir OrbitSight_Dataset/Training_sets \
        --pred-dir predictions/training \
        --out models/box_scales.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import sensor_for_sequence

GT_SUFFIX = "_bb_windows_40ms.txt"


def _load_pred(path):
    import csv
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((int(r["window_start_timestamp_us"]),
                         int(r["window_end_timestamp_us"]),
                         float(r["center_x"]), float(r["center_y"]),
                         float(r["width"]), float(r["height"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="models/box_scales.json")
    ap.add_argument("--max-center-dist", type=float, default=20.0,
                    help="px center-distance to accept a (pred,gt) match")
    args = ap.parse_args()

    # accumulate gt/pred size ratios + raw GT sizes per sensor
    ratios: dict[str, list] = {}
    gt_sizes: dict[str, list] = {}
    for gt_path in sorted(glob.glob(os.path.join(args.gt_dir, "*" + GT_SUFFIX))):
        seq = os.path.basename(gt_path)[: -len(GT_SUFFIX)]
        sensor = sensor_for_sequence(seq).name
        gt = D.load_gt_boxes(gt_path)
        for _, _, _, _, gw, gh in gt:
            gt_sizes.setdefault(sensor, []).append((gw, gh))
        pred_path = os.path.join(args.pred_dir, seq + GT_SUFFIX)
        if not os.path.exists(pred_path):
            continue
        pred = _load_pred(pred_path)
        # index GT by window-overlap; match each pred to nearest-center GT in time
        for ws, we, pcx, pcy, pw, ph in pred:
            best = None
            best_d = args.max_center_dist
            for gws, gwe, gcx, gcy, gw, gh in gt:
                if ws < gwe and we > gws:                  # time overlap
                    d = float(np.hypot(pcx - gcx, pcy - gcy))
                    if d < best_d:
                        best_d, best = d, (gw, gh, pw, ph)
            if best and best[2] > 0 and best[3] > 0:
                ratios.setdefault(sensor, []).append((best[0] / best[2],
                                                       best[1] / best[3]))

    out = {}
    sensors = set(gt_sizes) | set(ratios)
    for sensor in sorted(sensors):
        entry = {}
        if sensor in ratios:
            arr = np.array(ratios[sensor])
            entry["scale"] = [round(float(np.median(arr[:, 0])), 4),
                              round(float(np.median(arr[:, 1])), 4)]
        if sensor in gt_sizes:
            sz = np.array(gt_sizes[sensor])
            entry["size"] = [round(float(np.median(sz[:, 0])), 2),
                             round(float(np.median(sz[:, 1])), 2)]
        out[sensor] = entry
        print(f"{sensor:6s}  scale={entry.get('scale')}  "
              f"typical_size={entry.get('size')}  (GT n={len(gt_sizes.get(sensor, []))})")
    if not out:
        print("[calibrate] no data found")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] wrote {args.out}")


if __name__ == "__main__":
    main()
