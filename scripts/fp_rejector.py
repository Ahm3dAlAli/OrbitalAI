#!/usr/bin/env python3
"""Independent-representation false-positive rejector (Thuraya3 precision lever).

The GHOST probe (paper 5.4) showed the CenterNet ENCODER features cannot separate
phantom detections from real RSOs (FP-vs-TP AUROC at chance). This attacks the same
false positives with a *different* representation: the classical Stage-2 coherence
features (orbitsight/features.py), which carry a different inductive bias. For each
predicted box we aggregate the coherence features of the events inside it, plus box-
level statistics (event count, peak/ring concentration, size, original confidence),
train a small LightGBM TP-vs-FP classifier on the TRAINING sequences, and re-rank the
test confidences by the classifier's P(TP):

    new_conf = conf ** (1 - beta) * P(TP) ** beta

AP is confidence-ranked, so pushing phantoms down the ranking lifts AP without dropping
recall. EITHER outcome is informative: a gain is a new precision lever; no gain shows
the floor is representation-INDEPENDENT (inseparable in both the encoder and the
coherence space) — the strongest form of the intrinsic-floor claim.

    python3 scripts/fp_rejector.py \
      --train-data-dir OrbitSight_Dataset/Training_sets --train-gt-dir OrbitSight_Dataset/Training_sets \
      --train-pred-dir /tmp/full_r3 \
      --data-dir OrbitSight_Dataset/Testing_sets --pred-dir /tmp/full_r3 --out-dir /tmp/fp_rej \
      --sequences DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 --beta 0.5 \
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
from orbitsight.features import build_cloud, _N_FEATURES

# per-detection feature vector: [log1p(n_in), concentration, w, h, conf] + mean coherence
_N_SCALAR = 5
_DIM = _N_SCALAR + _N_FEATURES


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


def gt_by_window(path):
    """window_start -> list of GT (cx, cy, w, h)."""
    out = {}
    if not os.path.exists(path):
        return out
    for (ws, we, cx, cy, w, h) in D.load_gt_boxes(path):
        out.setdefault(int(ws), []).append((cx, cy, w, h))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def box_features(xs, ys, ts, ps, cx, cy, bw, bh, conf, cfg, sensor):
    """Aggregate classical coherence features over the events inside the box, plus
    box-level statistics. Returns a fixed-length vector of size _DIM."""
    hw, hh = bw / 2.0, bh / 2.0
    inb = (np.abs(xs - cx) <= hw) & (np.abs(ys - cy) <= hh)
    n_in = int(inb.sum())
    outb = (np.abs(xs - cx) <= 2 * hw) & (np.abs(ys - cy) <= 2 * hh)
    n_ring = int(outb.sum()) - n_in
    d_in = (n_in + 0.5) / max(bw * bh, 1.0)
    d_ring = (n_ring + 0.5) / max(3.0 * bw * bh, 1.0)
    conc = d_in / d_ring
    feat_mean = np.zeros(_N_FEATURES, dtype=np.float64)
    if n_in >= 2:
        cloud = build_cloud(xs, ys, ts, ps, sensor, cfg)
        qi = np.where(inb)[0]
        F = cloud.features(qi)                       # (n_in, _N_FEATURES)
        feat_mean = F.mean(axis=0)
    return np.concatenate([[np.log1p(n_in), conc, bw, bh, conf], feat_mean])


def collect(seqs, data_dir, pred_dir, gt_dir, cfg, label=True):
    """Build (X, y, meta) over all detections in the given sequences."""
    X, y = [], []
    for seq in seqs:
        rows = read_rows(os.path.join(pred_dir, seq + D.GT_SUFFIX))
        ep = D.find_event_file(data_dir, seq)
        if not rows or not ep:
            continue
        gt = gt_by_window(os.path.join(gt_dir, seq + D.GT_SUFFIX)) if gt_dir else {}
        ev = D.Events.from_npy(ep)
        sensor = sensor_for_sequence(seq)
        wins = {w.start_us: w for w in D.make_window_grid(ev.t, cfg.window_us)}
        for ws, boxes in rows.items():
            w = wins.get(ws)
            if w is None:
                continue
            xs, ys = ev.x[w.lo:w.hi], ev.y[w.lo:w.hi]
            ts, ps = ev.t[w.lo:w.hi], ev.pol[w.lo:w.hi]
            gtb = gt.get(ws, [])
            for (cx, cy, bw, bh, c) in boxes:
                X.append(box_features(xs, ys, ts, ps, cx, cy, bw, bh, c, cfg, sensor))
                if label:
                    tp = any(iou((cx, cy, bw, bh), g) >= 0.5 for g in gtb)
                    y.append(1 if tp else 0)
        del ev
    return (np.asarray(X, np.float32),
            np.asarray(y, np.int32) if label else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data-dir", required=True)
    ap.add_argument("--train-pred-dir", required=True)
    ap.add_argument("--train-gt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None, help="test seqs to re-rank")
    ap.add_argument("--gt-dir", default=None, help="test GT dir -> score after")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="0 = no change, 1 = confidence fully replaced by P(TP)")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG

    import lightgbm as lgb

    # ---- build training set from the TRAINING-sequence detections -------- #
    train_seqs = sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.train_data_dir, "*" + D.EV_SUFFIX)))
    print(f"[fp-rej] building training set over {len(train_seqs)} training sequences...")
    Xtr, ytr = collect(train_seqs, args.train_data_dir, args.train_pred_dir,
                       args.train_gt_dir, cfg, label=True)
    if Xtr.size == 0 or ytr.sum() == 0 or ytr.sum() == len(ytr):
        print(f"[fp-rej] ERROR: degenerate training set (n={len(ytr)}, pos={int(ytr.sum())})"); return
    print(f"[fp-rej] train detections={len(ytr)}  TP={int(ytr.sum())} "
          f"FP={int(len(ytr) - ytr.sum())} (precision {ytr.mean():.3f})")

    # small, shallow model — a phantom rejector must GENERALIZE, not memorize.
    clf = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, max_depth=4,
                             learning_rate=0.05, min_child_samples=40,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             verbose=-1)
    clf.fit(Xtr, ytr)
    tr_auc = _auroc(ytr, clf.predict_proba(Xtr)[:, 1])
    print(f"[fp-rej] train FP-vs-TP AUROC={tr_auc:.3f} (in-sample; watch for overfit)")

    # ---- apply to the test sequences ------------------------------------- #
    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.GT_SUFFIX)]
        for p in glob.glob(os.path.join(args.pred_dir, "*" + D.GT_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for seq in seqs:
        rows = read_rows(os.path.join(args.pred_dir, seq + D.GT_SUFFIX))
        ep = D.find_event_file(args.data_dir, seq)
        if not rows or not ep:
            continue
        ev = D.Events.from_npy(ep)
        sensor = sensor_for_sequence(seq)
        wins = {w.start_us: w for w in D.make_window_grid(ev.t, cfg.window_us)}
        # test AUROC if GT available (diagnostic)
        gt = gt_by_window(os.path.join(args.gt_dir, seq + D.GT_SUFFIX)) if args.gt_dir else {}
        out, ys, ps_diag = [], [], []
        for ws, boxes in rows.items():
            w = wins.get(ws)
            if w is None:
                for (cx, cy, bw, bh, c) in boxes:
                    out.append((ws, ws + cfg.window_us, cx, cy, bw, bh, c))
                continue
            xs, yy = ev.x[w.lo:w.hi], ev.y[w.lo:w.hi]
            ts, pp = ev.t[w.lo:w.hi], ev.pol[w.lo:w.hi]
            gtb = gt.get(ws, [])
            feats = [box_features(xs, yy, ts, pp, cx, cy, bw, bh, c, cfg, sensor)
                     for (cx, cy, bw, bh, c) in boxes]
            ptp = clf.predict_proba(np.asarray(feats, np.float32))[:, 1] if feats else []
            for (cx, cy, bw, bh, c), p in zip(boxes, ptp):
                nc = float(np.clip((c ** (1 - args.beta)) * (p ** args.beta), 1e-4, 1.0))
                out.append((w.start_us, w.end_us, cx, cy, bw, bh, nc))
                if gtb or args.gt_dir:
                    ys.append(1 if any(iou((cx, cy, bw, bh), g) >= 0.5 for g in gtb) else 0)
                    ps_diag.append(p)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX),
                            [D.Detection(*d) for d in out])
        auc = _auroc(np.asarray(ys), np.asarray(ps_diag)) if len(set(ys)) == 2 else float("nan")
        print(f"  {seq[:38]:38s} boxes {sum(len(v) for v in rows.values()):5d}  "
              f"test FP-vs-TP AUROC={auc:.3f}")
        del ev

    if args.gt_dir:
        os.system(f'python3 scripts/evaluate_wrapper.py --dataset '
                  f'"{os.path.dirname(args.gt_dir)}" --pred-dir "{args.out_dir}" '
                  f'--excel-out "{args.out_dir}/m.xlsx" 2>/dev/null '
                  f'| grep -E "Thuraya3|Stars3|SAOCOM|EVK4_mag7.3|mAP @"')


def _auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    main()
