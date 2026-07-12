#!/usr/bin/env python3
"""Merge shift-and-stack detections INTO existing predictions (DVX recall).

The CenterNet localizes well but misses the dimmest windows entirely; the
synthetic-tracking stacker (velocity-space outlier) is built exactly for those
2-4-event objects.  This fills windows the base predictions left empty with
stack detections, boosting recall without overriding the CenterNet's boxes.

    python3 scripts/stack_merge.py \
        --data-dir OrbitSight_Dataset/Testing_sets \
        --base-dir predictions/test_g192 --out-dir predictions/test_g192_stack \
        --sequences DVX_Filtered_Stars3_2025-01-20-20-22-53 \
                    DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.accumulate import stack_sequence, merge_fill


def load_pred(path):
    dets = []
    if not os.path.exists(path):
        return dets
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            dets.append(D.Detection(
                int(r["window_start_timestamp_us"]), int(r["window_end_timestamp_us"]),
                int(float(r["center_x"])), int(float(r["center_y"])),
                int(float(r["width"])), int(float(r["height"])),
                float(r.get("confidence", 1.0))))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-dir", required=True, help="predictions to augment")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--scales", default="models/box_scales.json")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    if os.path.exists(args.scales):
        for s, e in json.load(open(args.scales)).items():
            if isinstance(e, dict) and "size" in e:
                cfg.box_size_px[s] = tuple(e["size"])

    os.makedirs(args.out_dir, exist_ok=True)
    # copy over everything from base first (so non-DVX sequences pass through)
    for p in os.listdir(args.base_dir):
        if p.endswith(D.GT_SUFFIX):
            shutil.copy(os.path.join(args.base_dir, p), os.path.join(args.out_dir, p))

    for seq in args.sequences:
        path = D.find_event_file(args.data_dir, seq)
        if not path:
            print(f"[skip] {seq}"); continue
        ev = D.Events.from_npy(path)
        sn = sensor_for_sequence(seq)
        base = load_pred(os.path.join(args.base_dir, seq + D.GT_SUFFIX))
        stack = stack_sequence(ev, sn, cfg)
        merged = merge_fill(base, stack)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), merged)
        print(f"{seq[:40]:40s} base={len(base):5d} +stack_filled={len(merged)-len(base):5d} "
              f"-> {len(merged)}")
        del ev
    print(f"[stack-merge] -> {args.out_dir}")


if __name__ == "__main__":
    main()
