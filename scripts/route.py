#!/usr/bin/env python3
"""Per-sensor detector router.

Assembles one prediction directory by choosing, for each sequence, the
prediction file from the detector that wins that sensor — then the frozen
evaluator scores a single consistent set.

Two ways to specify the policy:

  # explicit per-sensor -> directory map (preferred):
  python3 scripts/route.py --out-dir predictions/router \
      --map EVK4=predictions/aug DVX=predictions/aug DAVIS=predictions/noaug

  # legacy two-dir form (CenterNet on some sensors, classical on the rest):
  python3 scripts/route.py --cnet-dir predictions/testing_cnet \
      --classical-dir predictions/testing --out-dir predictions/router \
      --cnet-sensors EVK4 DAVIS
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import sensor_for_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--map", nargs="+", default=None,
                    help="SENSOR=dir entries, e.g. EVK4=preds/aug DAVIS=preds/noaug")
    # legacy form
    ap.add_argument("--cnet-dir", default=None)
    ap.add_argument("--classical-dir", default=None)
    ap.add_argument("--cnet-sensors", nargs="+", default=["EVK4", "DAVIS"])
    args = ap.parse_args()

    if args.map:
        policy = {}
        for kv in args.map:
            s, d = kv.split("=", 1)
            policy[s.upper()] = d
        all_dirs = list(policy.values())
    else:
        assert args.cnet_dir and args.classical_dir, "need --map or the legacy dirs"
        cnet = set(s.upper() for s in args.cnet_sensors)
        policy = None
        all_dirs = [args.cnet_dir, args.classical_dir]

    os.makedirs(args.out_dir, exist_ok=True)
    seqs = set()
    for d in all_dirs:
        for p in glob.glob(os.path.join(d, "*" + D.GT_SUFFIX)):
            seqs.add(os.path.basename(p)[: -len(D.GT_SUFFIX)])

    for seq in sorted(seqs):
        sensor = sensor_for_sequence(seq).name
        if policy is not None:
            src_dir = policy.get(sensor)
        else:
            src_dir = args.cnet_dir if sensor in cnet else args.classical_dir
        if not src_dir:
            continue
        src = os.path.join(src_dir, seq + D.GT_SUFFIX)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, seq + D.GT_SUFFIX))
            print(f"  {seq[:44]:44s} [{sensor:5s}] <- {src_dir}")
    print(f"[route] -> {args.out_dir}")


if __name__ == "__main__":
    main()
