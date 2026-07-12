#!/usr/bin/env python3
"""OrbitSight inference: run the full pipeline over a directory of event
recordings and write one prediction file per sequence.

Prediction files are written under ``--out-dir`` with TWO names for maximum
compatibility:
  * ``<sequence>_bb_windows_40ms.txt`` — matches the frozen evaluate.py loader
  * ``<sequence>_pred.txt``            — matches the README submission spec

Usage:
    python3 scripts/infer.py \
        --data-dir OrbitSight_Dataset/Testing_sets \
        --model models/coherence_lgbm.joblib \
        --out-dir predictions/testing
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG
from orbitsight.model import CoherenceClassifier
from orbitsight.pipeline import run_sequence


def discover_sequences(data_dir: str):
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*" + D.EV_SUFFIX))):
        out.append(os.path.basename(p)[: -len(D.EV_SUFFIX)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="models/coherence_lgbm.joblib")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--single-name", action="store_true",
                    help="only write <seq>_bb_windows_40ms.txt")
    ap.add_argument("--stack", action="store_true",
                    help="add synthetic-tracking (shift-and-stack) detections "
                         "to recover dim objects the classifier misses")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    clf = CoherenceClassifier.load(args.model, cfg)
    mode = "LightGBM" if clf.model is not None else "heuristic-fallback"
    print(f"[INFO] classifier: {mode}")

    # load per-sensor box-size calibration if present (scripts/calibrate.py)
    scale_path = os.path.join(os.path.dirname(args.model) or ".", "box_scales.json")
    if os.path.exists(scale_path):
        import json
        blob = json.load(open(scale_path))
        for sensor, entry in blob.items():
            if isinstance(entry, dict):                      # rich format
                if "scale" in entry:
                    cfg.box_size_scale[sensor] = tuple(entry["scale"])
                if "size" in entry:
                    cfg.box_size_px[sensor] = tuple(entry["size"])
            else:                                            # legacy [sw, sh]
                cfg.box_size_scale[sensor] = tuple(entry)
        print(f"[INFO] box scale={cfg.box_size_scale} typical={cfg.box_size_px}")

    seqs = args.sequences or discover_sequences(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    tot_ms = []
    for i, seq in enumerate(seqs, 1):
        path = D.find_event_file(args.data_dir, seq)
        if not path:
            print(f"[skip] missing {seq}")
            continue
        ev = D.Events.from_npy(path)
        dets, res = run_sequence(ev, seq, clf, cfg)
        if args.stack:
            from orbitsight.accumulate import stack_sequence, merge_fill
            from orbitsight.config import sensor_for_sequence
            stack = stack_sequence(ev, sensor_for_sequence(seq), cfg)
            n0 = len(dets)
            dets = merge_fill(dets, stack)
            print(f"        [stack] +{len(dets) - n0} dim-object windows recovered")
        del ev
        gt = os.path.join(args.out_dir, seq + D.GT_SUFFIX)
        D.write_predictions(gt, dets)
        if not args.single_name:
            D.write_predictions(os.path.join(args.out_dir, seq + "_pred.txt"), dets)
        tot_ms.append(res.ms_per_window)
        print(f"[{i:2d}/{len(seqs)}] {seq[:40]:40s} {res.sensor:5s} "
              f"win={res.n_windows:5d} det={res.n_detections:5d} "
              f"{res.ms_per_window:6.2f} ms/win  ({res.seconds:5.1f}s)")

    if tot_ms:
        import numpy as np
        print(f"\n[INFO] mean {np.mean(tot_ms):.2f} ms/window "
              f"(max {np.max(tot_ms):.2f}) — budget < 40 ms")


if __name__ == "__main__":
    main()
