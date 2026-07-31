#!/usr/bin/env python3
"""Motion-guided CenterNet inference for the faint Thuraya3 regime.

Unlike ordinary inference, this script keeps several local heatmap maxima down
to a permissive floor and links them *before* thresholding.  A motion beam
integrates weak evidence across windows, rejects static hot pixels, and emits
the strongest coherent single-object path.  It is deliberately a specialist:
route only single-object faint sequences such as Thuraya3 through it.

Example
-------
python3 scripts/infer_track_before_detect.py \
  --data-dir OrbitSight_Dataset/Testing_sets \
  --models models/g192_ctx_v2.pt \
  --out-dir predictions/thuraya_tbd --tta --box-px 10 12 \
  --sequences DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.evt_centernet import decode_peaks
from orbitsight.evt_model import voxelize
from orbitsight.track_before_detect import (
    BeamConfig,
    Peak,
    link_peaks,
    track_to_detections,
)
from scripts.infer_ensemble import _predict, load


def _discover(data_dir: str) -> list[str]:
    return sorted(
        os.path.basename(p)[: -len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(data_dir, "*" + D.EV_SUFFIX))
    )


@torch.no_grad()
def collect_peaks(
    ev,
    seq,
    models,
    cfgs,
    cfg,
    *,
    topk,
    tta,
    device,
    batch,
    peak_floor,
):
    """Run an ensemble and retain low-confidence local maxima in every window."""

    sensor = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    common_hm = max(c["grid"] // c["hm_div"] for c in cfgs)
    per_window: list[list[Peak]] = [[] for _ in wins]
    wbuf = []

    def make_voxel(w, grid, tbins, context, time_surface):
        lo, hi, ws, we = w.lo, w.hi, w.start_us, w.end_us
        if context > 0:
            ws -= context * cfg.window_us
            we += context * cfg.window_us
            lo = int(np.searchsorted(ev.t, ws, side="left"))
            hi = int(np.searchsorted(ev.t, we, side="left"))
        return voxelize(
            ev.x[lo:hi],
            ev.y[lo:hi],
            ev.pol[lo:hi],
            ev.t[lo:hi],
            ws,
            we,
            sensor.width,
            sensor.height,
            grid,
            tbins,
            time_surface=time_surface,
        )

    def flush():
        if not wbuf:
            return
        cache = {}
        for c in cfgs:
            key = (
                c["grid"],
                c["tbins"],
                c.get("context", 0),
                c.get("time_surface", False),
            )
            if key not in cache:
                arr = np.stack([make_voxel(w, *key) for w in wbuf])
                cache[key] = torch.from_numpy(arr).float().to(device)

        hm_sum = wh_sum = off_sum = None
        for model, c in zip(models, cfgs):
            key = (
                c["grid"],
                c["tbins"],
                c.get("context", 0),
                c.get("time_surface", False),
            )
            hm, wh, off = _predict(model, cache[key], tta)
            if hm.shape[-1] != common_hm:
                hm = F.interpolate(
                    hm, size=(common_hm, common_hm), mode="bilinear", align_corners=False
                )
                wh = F.interpolate(
                    wh, size=(common_hm, common_hm), mode="bilinear", align_corners=False
                )
                off = F.interpolate(
                    off, size=(common_hm, common_hm), mode="bilinear", align_corners=False
                )
            hm_sum = hm if hm_sum is None else hm_sum + hm
            wh_sum = wh if wh_sum is None else wh_sum + wh
            off_sum = off if off_sum is None else off_sum + off

        n_models = len(models)
        prob = (hm_sum / n_models).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(prob / (1.0 - prob))
        decoded = decode_peaks(
            logits.cpu(),
            (wh_sum / n_models).cpu(),
            (off_sum / n_models).cpu(),
            topk=topk,
        )
        for win, peaks in zip(wbuf, decoded):
            keep = []
            for score, cx, cy, bw, bh in peaks:
                if score < peak_floor:
                    continue
                keep.append(
                    Peak(
                        win_idx=win.index,
                        ws=win.start_us,
                        we=win.end_us,
                        cx=float(cx * sensor.width),
                        cy=float(cy * sensor.height),
                        w=max(float(bw * sensor.width), 1.0),
                        h=max(float(bh * sensor.height), 1.0),
                        score=float(score),
                    )
                )
            per_window[win.index] = keep
        wbuf.clear()

    for win in wins:
        wbuf.append(win)
        if len(wbuf) >= batch:
            flush()
    flush()
    return per_window, wins


def _fallback_top1(per_window, sensor, threshold):
    """Safe fallback when no moving path passes the physical guards."""

    out = []
    for peaks in per_window:
        if not peaks:
            continue
        p = max(peaks, key=lambda x: x.score)
        if p.score < threshold:
            continue
        out.append(
            D.Detection(
                p.ws,
                p.we,
                int(round(np.clip(p.cx, 0, sensor.width - 1))),
                int(round(np.clip(p.cy, 0, sensor.height - 1))),
                max(int(round(p.w)), 1),
                max(int(round(p.h)), 1),
                p.score,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--peak-floor", type=float, default=0.03)
    ap.add_argument("--seed-thresh", type=float, default=0.10)
    ap.add_argument("--gate-px", type=float, default=10.0)
    ap.add_argument("--gate-growth-px", type=float, default=2.0)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--min-track-len", type=int, default=8)
    ap.add_argument("--min-displacement-px", type=float, default=8.0)
    ap.add_argument("--max-rms-residual-px", type=float, default=8.0)
    ap.add_argument("--beam-size", type=int, default=256)
    ap.add_argument(
        "--box-px",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="optional training-calibrated fixed box size; Thuraya3: 10 12",
    )
    ap.add_argument(
        "--fallback-thresh",
        type=float,
        default=0.30,
        help="ordinary top-1 threshold if no coherent moving path is found",
    )
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    loaded = [load(path, device) for path in args.models]
    models = [m for m, _ in loaded]
    cfgs = [c for _, c in loaded]
    seqs = args.sequences or _discover(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    beam_cfg = BeamConfig(
        peak_floor=args.peak_floor,
        seed_thresh=args.seed_thresh,
        gate_px=args.gate_px,
        gate_growth_px=args.gate_growth_px,
        max_gap=args.max_gap,
        min_track_len=args.min_track_len,
        min_displacement_px=args.min_displacement_px,
        max_rms_residual_px=args.max_rms_residual_px,
        beam_size=args.beam_size,
    )
    print(
        f"[TBD] models={len(models)} topk={args.topk} floor={args.peak_floor:g} "
        f"gate={args.gate_px:g}px max_gap={args.max_gap} beam={args.beam_size} "
        f"tta={args.tta} [{device}]"
    )

    for i, seq in enumerate(seqs, 1):
        event_path = D.find_event_file(args.data_dir, seq)
        if not event_path:
            print(f"[{i:2d}/{len(seqs)}] {seq}: event file missing; skipped")
            continue
        ev = D.Events.from_npy(event_path)
        sensor = sensor_for_sequence(seq)
        t0 = time.perf_counter()
        per_window, wins = collect_peaks(
            ev,
            seq,
            models,
            cfgs,
            DEFAULT_CONFIG,
            topk=args.topk,
            tta=args.tta,
            device=device,
            batch=args.batch,
            peak_floor=args.peak_floor,
        )
        track = link_peaks(per_window, beam_cfg)
        if track is None:
            detections = _fallback_top1(
                per_window, sensor, args.fallback_thresh
            )
            status = f"fallback top1 ({len(detections)} boxes)"
        else:
            detections = track_to_detections(
                track,
                sensor_width=sensor.width,
                sensor_height=sensor.height,
                box_px=tuple(args.box_px) if args.box_px else None,
            )
            status = (
                f"track obs={len(track.peaks)} out={len(detections)} "
                f"disp={track.displacement_px:.1f}px "
                f"rms={track.rms_residual_px:.2f}px"
            )
        D.write_predictions(
            os.path.join(args.out_dir, seq + D.GT_SUFFIX), detections
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[{i:2d}/{len(seqs)}] {seq[:38]:38s} {status}; "
            f"{1000 * elapsed / max(len(wins), 1):.2f} ms/win"
        )
        del ev


if __name__ == "__main__":
    main()
