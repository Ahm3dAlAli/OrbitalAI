#!/usr/bin/env python3
"""Offline training of the OrbitSight coherence classifier (Stage 2).

Builds per-event coherence features from the labeled training sequences
(all RSO events + a class-balanced background subsample), trains a LightGBM
binary classifier, and saves weights + a model-structure JSON.

Usage:
    python3 scripts/train.py \
        --data-dir OrbitSight_Dataset/Training_sets \
        --out models/coherence_lgbm.joblib
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, TRAIN_SEQUENCES, sensor_for_sequence
from orbitsight.pipeline import extract_training_samples
from orbitsight.model import CoherenceClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Training_sets")
    ap.add_argument("--out", default="models/coherence_lgbm.joblib")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="subset of sequence names (default: all training)")
    ap.add_argument("--holdout", default=None,
                    help="sensor or sequence token to EXCLUDE (leave-one-out)")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    rng = np.random.default_rng(cfg.random_seed)
    seqs = args.sequences or TRAIN_SEQUENCES
    if args.holdout:
        seqs = [s for s in seqs if args.holdout.upper() not in s.upper()]

    Xs, ys, groups = [], [], []
    t_start = time.perf_counter()
    for i, seq in enumerate(seqs, 1):
        path = D.find_event_file(args.data_dir, seq)
        if not path:
            print(f"[skip] missing {seq}")
            continue
        ev = D.Events.from_npy(path)
        if ev.n > cfg.max_events_per_seq:
            ev = ev.head(cfg.max_events_per_seq)
        sensor = sensor_for_sequence(seq)
        t0 = time.perf_counter()
        X, y = extract_training_samples(ev, sensor, cfg, rng,
                                        max_rso_per_seq=cfg.max_rso_per_seq)
        dt = time.perf_counter() - t0
        pos = int((y == 1).sum())
        print(f"[{i:2d}/{len(seqs)}] {seq[:42]:42s} {sensor.name:5s} "
              f"X={X.shape[0]:7d} pos={pos:6d}  ({dt:5.1f}s)")
        if X.shape[0]:
            Xs.append(X); ys.append(y); groups.append(np.full(len(y), i))

    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    print(f"\nTotal samples: {X.shape[0]:,}  positives: {int((y==1).sum()):,}  "
          f"({(y==1).mean():.3%})  features: {X.shape[1]}")

    clf = CoherenceClassifier(cfg)
    t0 = time.perf_counter()
    clf.fit(X, y)
    print(f"Trained LightGBM in {time.perf_counter()-t0:.1f}s")

    # quick train-set diagnostic
    p = clf.predict_proba(X)
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        print(f"Train AUC={auc:.4f}  AP={ap:.4f}")
    except Exception:
        pass
    # feature importances
    if clf.model is not None:
        imp = clf.model.booster_.feature_importance(importance_type="gain")
        names = clf.feature_names          # the model's selected feature subset
        order = np.argsort(imp)[::-1]
        print("Top features by gain (brightness-invariant subset):")
        for j in order[:8]:
            print(f"    {names[j]:18s} {imp[j]:.0f}")

    clf.save(args.out)
    print(f"\nSaved model -> {args.out}")
    print(f"Total wall time: {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
