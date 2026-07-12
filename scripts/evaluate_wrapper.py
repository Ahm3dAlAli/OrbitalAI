#!/usr/bin/env python3
"""Drive the frozen OrbitSight evaluator over a dataset + prediction folder and
write ``Evaluation_Metrics.xlsx`` (PRD FR-8).

Auto-detects ``Training_sets`` / ``Testing_sets`` under ``--dataset`` (falling
back to a flat layout) and matches predictions by filename.  Reuses the
vendored, unmodified evaluator in ``orbitsight/eval_harness.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import eval_harness as H


def _has_gt(d):
    return os.path.isdir(d) and any(f.endswith(H.GT_SUFFIX if hasattr(H, "GT_SUFFIX")
                                    else "_bb_windows_40ms.txt") for f in os.listdir(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--excel-out", default="Evaluation_Metrics.xlsx")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    train_gt = os.path.join(args.dataset, "Training_sets")
    test_gt = os.path.join(args.dataset, "Testing_sets")

    results = []
    if _has_gt(train_gt):
        print(f"[eval] training GT: {train_gt}")
        results += H.evaluate_dir(train_gt, args.pred_dir, args.iou, "Training")
    if _has_gt(test_gt):
        print(f"[eval] testing  GT: {test_gt}")
        results += H.evaluate_dir(test_gt, args.pred_dir, args.iou, "Testing")
    if not results and _has_gt(args.dataset):
        print(f"[eval] flat GT: {args.dataset}")
        results += H.evaluate_dir(args.dataset, args.pred_dir, args.iou, "")

    if not results:
        print("[eval] no GT found — cannot score.")
        return 1

    H.print_results(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.excel_out)), exist_ok=True)
    H.write_excel(results, args.excel_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
