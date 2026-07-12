#!/usr/bin/env python3
"""End-to-end latency / throughput benchmark for the CenterNet event detector.

Scoring criterion #3 ("Real-time Performance") names a hard target: **end-to-end
latency under 40 ms** — i.e. the pipeline must keep up with the 40 ms event
windows in a streaming setting.  This tool measures the true cost of turning one
40 ms window of raw events into a box, broken down by stage:

    voxelize (CPU)  ->  model forward (GPU/CPU)  ->  decode (CPU)

and reports it per sensor (DAVIS 346x260, DVX 640x480, EVK4 1280x720), because
latency scales with resolution and event rate.  Two regimes are reported:

  * streaming  (batch=1): the per-window latency a real-time deployment sees.
  * batched    (batch=N): throughput when windows are processed in bulk.

Usage:
    python3 scripts/benchmark_latency.py --device cuda \
        --model models/g192_ctx.pt --data-dir OrbitSight_Dataset/Testing_sets \
        --md-out docs/latency.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.evt_model import voxelize
from orbitsight.evt_centernet import EventCenterNet, decode


def load_model(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    m = EventCenterNet(grid=c["grid"], patch=c["patch"], tbins=c["tbins"],
                       dim=c["dim"], hm_div=c["hm_div"],
                       enc_layers=c.get("enc_layers", 3), variant=c["variant"])
    m.load_state_dict(blob["state_dict"]); m.eval(); m.to(device)
    return m, c


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_sequence(ev, seq, model, c, cfg, device, batch, max_windows, warmup):
    """Return per-stage timing stats (ms/window) for one sequence."""
    sn = sensor_for_sequence(seq)
    ctx = c.get("context", 0)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    if max_windows:
        wins = wins[:max_windows]
    if not wins:
        return None

    def vox(win):
        lo, hi, ws, we = win.lo, win.hi, win.start_us, win.end_us
        if ctx > 0:
            ws, we = ws - ctx * cfg.window_us, we + ctx * cfg.window_us
            lo = int(np.searchsorted(ev.t, ws, "left"))
            hi = int(np.searchsorted(ev.t, we, "left"))
        return voxelize(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], ev.t[lo:hi],
                        ws, we, sn.width, sn.height, c["grid"], c["tbins"])

    # ---- warmup (first CUDA kernels + allocator are slow, not representative)
    for w in wins[:warmup]:
        x = torch.from_numpy(vox(w)[None]).float().to(device)
        hm, wh, off = model(x); _sync(device)

    t_vox = t_fwd = t_dec = 0.0
    n = 0
    buf, wbuf = [], []

    def run_batch():
        nonlocal t_vox, t_fwd, t_dec, n
        if not wbuf:
            return
        t0 = time.perf_counter()
        arr = np.stack([vox(w) for w in wbuf])
        x = torch.from_numpy(arr).float().to(device); _sync(device)
        t1 = time.perf_counter()
        hm, wh, off = model(x); _sync(device)
        t2 = time.perf_counter()
        hm, wh, off = hm.cpu(), wh.cpu(), off.cpu()
        decode(hm, wh, off, topk=1); _sync(device)
        t3 = time.perf_counter()
        t_vox += (t1 - t0); t_fwd += (t2 - t1); t_dec += (t3 - t2)
        n += len(wbuf); wbuf.clear()

    for w in wins:
        wbuf.append(w)
        if len(wbuf) >= batch:
            run_batch()
    run_batch()

    ev_total = int(ev.t.size)
    per = 1000.0 / max(n, 1)
    return {
        "sensor": sn.name, "grid": c["grid"], "context": ctx,
        "windows": n, "events": ev_total,
        "vox_ms": t_vox * per, "fwd_ms": t_fwd * per, "dec_ms": t_dec * per,
        "total_ms": (t_vox + t_fwd + t_dec) * per,
        "throughput_win_s": n / max(t_vox + t_fwd + t_dec, 1e-9),
    }


def fmt_table(rows, batch):
    real = "REAL-TIME" if all(r["total_ms"] < 40 for r in rows) else "see rows"
    lines = [
        f"Batch size = {batch}  (streaming latency uses batch=1)",
        "",
        "| Sensor | Grid | Ctx | Windows |   Vox |   Fwd |   Dec | **Total** | <40ms? | win/s |",
        "|--------|------|-----|---------|-------|-------|-------|-----------|--------|-------|",
    ]
    for r in rows:
        ok = "✅" if r["total_ms"] < 40 else "⚠️"
        lines.append(
            f"| {r['sensor']:6s} | {r['grid']:4d} | {r['context']:3d} | "
            f"{r['windows']:7d} | {r['vox_ms']:5.2f} | {r['fwd_ms']:5.2f} | "
            f"{r['dec_ms']:5.2f} | **{r['total_ms']:6.2f}** | {ok} | "
            f"{r['throughput_win_s']:6.0f} |")
    lines.append("")
    lines.append(f"Real-time target (<40 ms end-to-end per window): **{real}**")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="models/g192_ctx.pt")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--batch", type=int, default=1,
                    help="1 = streaming latency; larger = batched throughput")
    ap.add_argument("--max-windows", type=int, default=400,
                    help="cap windows/sequence for a quick, stable measurement")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--md-out", default=None, help="write a markdown table here")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    model, c = load_model(args.model, device)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[INFO] {os.path.basename(args.model)}  grid={c['grid']} "
          f"ctx={c.get('context',0)} tbins={c['tbins']}  device={device} ({gpu})")

    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    rows = []
    for seq in seqs:
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        r = bench_sequence(ev, seq, model, c, cfg, device,
                           args.batch, args.max_windows, args.warmup)
        del ev
        if r is None:
            continue
        r["sequence"] = seq
        rows.append(r)
        print(f"  {seq[:38]:38s} total={r['total_ms']:6.2f} ms/win "
              f"(vox {r['vox_ms']:.2f} + fwd {r['fwd_ms']:.2f} + dec {r['dec_ms']:.2f})")

    table = fmt_table(rows, args.batch)
    print("\n" + table)

    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "w") as f:
            f.write(f"# End-to-end latency ({os.path.basename(args.model)}, "
                    f"{gpu})\n\n" + table + "\n")
        print(f"\n[INFO] markdown -> {args.md_out}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"model": args.model, "device": device, "gpu": gpu,
                       "batch": args.batch, "rows": rows}, f, indent=2)
        print(f"[INFO] json -> {args.json_out}")


if __name__ == "__main__":
    main()
