#!/usr/bin/env python3
"""Oracle separability check (dev diagnostic).

Question: at the *true* GT location, is the dim object separable from
background at all?  This decides whether higher mAP is physically attainable
or whether the information simply is not in the data.

For each GT window we place the GT box (oracle) and count, inside it:
  signal = events labeled RSO (label==1)
  noise  = events labeled background (label==0)
plus the same in a margin ring around the box.  We summarize the per-window
signal/noise counts and how often signal is cleanly dominant.

Verdict heuristic:
  * median signal >> median in-box noise, signal>=~10  -> separable, accumulate
  * signal ~ noise and both tiny (~2 vs ~2)            -> not separable here
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import sensor_for_sequence


def box_mask(x, y, cx, cy, w, h, pad=0.0):
    hw, hh = w / 2 + pad, h / 2 + pad
    return (np.abs(x - cx) <= hw) & (np.abs(y - cy) <= hh)


def analyze(seq, data_dir):
    ev = D.Events.from_npy(D.find_event_file(data_dir, seq))
    gt = D.load_gt_boxes(os.path.join(data_dir, seq + D.GT_SUFFIX))
    ts, lab = ev.t, ev.label
    sig_in, noise_in, noise_ring, tot_win = [], [], [], []
    sig_anywhere = []
    for ws, we, cx, cy, w, h in gt:
        lo = np.searchsorted(ts, ws); hi = np.searchsorted(ts, we)
        if hi <= lo:
            continue
        sx, sy, sl = ev.x[lo:hi], ev.y[lo:hi], lab[lo:hi]
        inbox = box_mask(sx, sy, cx, cy, w, h)
        ring = box_mask(sx, sy, cx, cy, w, h, pad=max(w, h)) & ~inbox
        sig_in.append(int(((sl == 1) & inbox).sum()))
        noise_in.append(int(((sl == 0) & inbox).sum()))
        noise_ring.append(int(((sl == 0) & ring).sum()))
        tot_win.append(hi - lo)
        sig_anywhere.append(int((sl == 1).sum()))
    return (np.array(sig_in), np.array(noise_in), np.array(noise_ring),
            np.array(tot_win), np.array(sig_anywhere))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Testing_sets")
    ap.add_argument("--sequences", nargs="+", required=True)
    args = ap.parse_args()

    for seq in args.sequences:
        sig, noi, ring, tot, sig_any = analyze(seq, args.data_dir)
        sensor = sensor_for_sequence(seq).name
        n = len(sig)
        if n == 0:
            print(f"{seq}: no GT windows"); continue
        # in-box SNR per window (signal / (signal+noise))
        snr = sig / np.maximum(sig + noi, 1)
        clean = np.mean(sig > noi)                 # windows where signal dominates box
        sep5 = np.mean((sig >= 5) & (sig > noi))   # comfortably separable
        print(f"\n===== {seq}  [{sensor}]  ({n} GT windows) =====")
        print(f"  events/window (total)        median={int(np.median(tot))}")
        print(f"  SIGNAL events in true box    median={int(np.median(sig))}  "
              f"mean={sig.mean():.1f}  p25={int(np.percentile(sig,25))} "
              f"p75={int(np.percentile(sig,75))}")
        print(f"  NOISE  events in true box    median={int(np.median(noi))}  "
              f"mean={noi.mean():.1f}")
        print(f"  NOISE  events in ring        median={int(np.median(ring))}  "
              f"mean={ring.mean():.1f}")
        print(f"  in-box signal fraction       median={np.median(snr):.2f}  "
              f"mean={snr.mean():.2f}")
        print(f"  windows signal>noise         {clean:.1%}")
        print(f"  windows signal>=5 & >noise   {sep5:.1%}")
        # verdict
        ms, mn = np.median(sig), np.median(noi)
        if ms >= 10 and ms > 2 * max(mn, 1):
            v = "SEPARABLE — info present; temporal accumulation can help"
        elif ms <= 4 and mn >= ms:
            v = "BURIED — ~signal≈noise at true location; likely NOT separable"
        else:
            v = "MARGINAL — partial signal; accumulation needed, gains capped"
        print(f"  VERDICT: {v}")


if __name__ == "__main__":
    main()
