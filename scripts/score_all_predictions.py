#!/usr/bin/env python3
"""Score EVERY prediction directory against the test GT and rank by mAP.

Shows the full over-time progression of the project (every experiment we ran)
and identifies the winning dir.  Writes docs/results_over_time.md + .json.

    python3 scripts/score_all_predictions.py
    python3 scripts/score_all_predictions.py --gt-dir OrbitSight_Dataset/Testing_sets
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import eval_harness as H


def mAP_of(gt_dir, pred_dir):
    """Overall mAP (mean of per-sequence AP) for one prediction dir."""
    res = H.evaluate_dir(gt_dir, pred_dir, 0.5, "")
    aps = [r["ap"] for r in res if r.get("ap") is not None and not np.isnan(r["ap"])]
    if not aps:
        return None, 0
    return float(np.mean(aps)), len(aps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="OrbitSight_Dataset/Testing_sets")
    ap.add_argument("--pred-root", default="predictions")
    ap.add_argument("--md-out", default="docs/results_over_time.md")
    ap.add_argument("--json-out", default="docs/results_over_time.json")
    args = ap.parse_args()

    if not os.path.isdir(args.gt_dir):
        sys.exit(f"[ERR] no GT dir {args.gt_dir} — sync the test set first")

    dirs = sorted(d for d in glob.glob(os.path.join(args.pred_root, "*"))
                  if os.path.isdir(d))
    rows = []
    for d in dirs:
        # must contain at least one prediction file matching the GT suffix
        if not glob.glob(os.path.join(d, "*" + H.GT_SUFFIX
                         if hasattr(H, "GT_SUFFIX") else "*_bb_windows_40ms.txt")):
            continue
        m, n = mAP_of(args.gt_dir, d)
        if m is None:
            continue
        rows.append({"dir": os.path.basename(d), "mAP": round(m, 4), "sequences": n})
        print(f"  {os.path.basename(d):28s} mAP={m:.4f}  ({n} seq)")

    rows.sort(key=lambda r: r["mAP"], reverse=True)
    if not rows:
        sys.exit("[ERR] no scorable prediction dirs found")

    best = rows[0]
    print("\n" + "=" * 52)
    print(f"  WINNER: {best['dir']}  —  mAP {best['mAP']:.4f}")
    print("=" * 52)

    os.makedirs(os.path.dirname(os.path.abspath(args.md_out)) or ".", exist_ok=True)
    with open(args.md_out, "w") as f:
        f.write("# Results over time — every prediction dir, ranked\n\n")
        f.write(f"Scored against `{args.gt_dir}` with the frozen evaluator "
                "(mAP = mean of per-sequence AP @ IoU 0.5).\n\n")
        f.write("| Rank | Prediction dir | mAP @ IoU 0.5 | seqs |\n")
        f.write("|---:|---|---:|---:|\n")
        for i, r in enumerate(rows, 1):
            star = " ⭐" if i == 1 else ""
            f.write(f"| {i} | `{r['dir']}`{star} | **{r['mAP']:.4f}** | {r['sequences']} |\n")
    with open(args.json_out, "w") as f:
        json.dump({"gt_dir": args.gt_dir, "ranked": rows, "winner": best}, f, indent=2)
    print(f"\n[out] {args.md_out}\n[out] {args.json_out}")
    print(f"\nRegenerate all assets from the winner:")
    print(f"  PRED_DIR=predictions/{best['dir']} bash scripts/build_all.sh")


if __name__ == "__main__":
    main()
