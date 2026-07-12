#!/usr/bin/env python3
"""Generate result figures for the OrbitSight submission (criterion #4).

Produces publication-quality PNGs under docs/figures/ from the MEASURED results
(frozen evaluate.py + benchmark_latency.py):

  1. map_progression.png   — mAP over the development roadmap, per lever
  2. per_sequence_ap.png    — final per-sequence AP (best routed model)
  3. metrics_summary.png    — precision / recall / F1 / mAP + TP/FP/FN
  4. latency_breakdown.png  — end-to-end ms/window per sensor vs 40 ms budget
  5. temporal_gain.png      — Thuraya3 / DAVIS before vs after temporal context

All numbers are hard-coded from confirmed runs and cited in ROADMAP.md; edit the
RESULTS block below when the temporal ensemble lands.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# palette
NAVY, BLUE, CYAN = "#0d1b2a", "#1b6ca8", "#2ec4b6"
AMBER, RED, GREY = "#f4a261", "#e63946", "#6c757d"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

# --------------------------------------------------------------------------- #
#  MEASURED RESULTS  (edit here when the ensemble finishes)
# --------------------------------------------------------------------------- #
ROADMAP = [
    ("Broken\nglobal-reg", 0.016),
    ("Classical\npipeline", 0.249),
    ("CenterNet\nheatmap", 0.289),
    ("Per-sensor\nrouter", 0.315),
    ("Event\naugment", 0.398),
    ("Grid-192\n+box calib", 0.454),
    ("Ensemble\n+stack", 0.554),
    ("Temporal\ncontext", 0.660),
    ("+ TTA", 0.675),
]
ENSEMBLE_PROJECTED = 0.70

# final routed model, per-sequence AP@0.5 (mAP 0.660 run; TTA lifts overall 0.675)
PER_SEQ = [
    ("EVK4\nmag7.3", 0.896, "EVK4"),
    ("DAVIS\nSAOCOM1B", 0.729, "DAVIS"),
    ("DVX\nStars3", 0.545, "DVX"),
    ("DVX\nThuraya3", 0.469, "DVX"),
]
# overall metrics at the confirmed best (router_ctta, mAP 0.675)
SUMMARY = {"mAP": 0.675, "Precision": 0.575, "Recall": 0.761, "F1": 0.655}
COUNTS = {"TP": 5385, "FP": 3976, "FN": 1691}

# end-to-end latency, ms/window (benchmark_latency.py, grid-128, CPU streaming)
LATENCY = [  # sensor, vox, fwd, dec
    ("EVK4\n1280x720", 1.34, 13.89, 0.19),
    ("DAVIS\n346x260", 0.72, 16.46, 0.21),
    ("DVX\n640x480", 0.51, 13.79, 0.16),
]
BUDGET_MS = 40.0

# temporal-context effect (AP@0.5, before vs after ±3-window context)
TEMPORAL = [("DVX Thuraya3", 0.233, 0.469), ("DAVIS SAOCOM1B", 0.617, 0.729)]

# --------------------------------------------------------------------------- #
#  Canonical source of truth: override the inline defaults from results.json
#  (so the repo stores ONE authoritative results file the figures render from).
# --------------------------------------------------------------------------- #
_RESULTS_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results.json")
if os.path.exists(_RESULTS_JSON):
    import json
    try:
        _r = json.load(open(_RESULTS_JSON))
        ROADMAP = [(x[0], x[1]) for x in _r["roadmap"]]
        ENSEMBLE_PROJECTED = _r["ensemble_projected"]
        PER_SEQ = [(x[0], x[1], x[2]) for x in _r["per_sequence"]]
        _o = _r["overall"]
        SUMMARY = {"mAP": _o["mAP"], "Precision": _o["precision"],
                   "Recall": _o["recall"], "F1": _o["f1"]}
        COUNTS = {"TP": _o["tp"], "FP": _o["fp"], "FN": _o["fn"]}
        LATENCY = [(x[0], x[1], x[2], x[3]) for x in _r["latency_ms"]]
        BUDGET_MS = _r["budget_ms"]
        TEMPORAL = [(x[0], x[1], x[2]) for x in _r["temporal_gain"]]
        print(f"[fig] results loaded from results.json")
    except Exception as e:
        print(f"[fig] WARN could not read results.json ({e}); using inline defaults")


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# 1 -------------------------------------------------------------------------- #
def fig_progression():
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(ROADMAP)))
    ys = [v for _, v in ROADMAP]
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.4, ms=7, zorder=3)
    ax.fill_between(xs, ys, color=BLUE, alpha=0.08)
    for x, (lab, v) in zip(xs, ROADMAP):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=NAVY)
    # projected ensemble
    ax.plot([xs[-1], xs[-1] + 1], [ys[-1], ENSEMBLE_PROJECTED], "--o",
            color=CYAN, lw=2.2, ms=7, zorder=3)
    ax.annotate(f"~{ENSEMBLE_PROJECTED:.2f}\n(ensemble,\nin progress)",
                (xs[-1] + 1, ENSEMBLE_PROJECTED), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=CYAN)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(xs + [xs[-1] + 1])
    ax.set_xticklabels([l for l, _ in ROADMAP] + ["Temporal\nensemble"], fontsize=8.5)
    ax.set_ylabel("mAP @ IoU 0.5")
    ax.set_ylim(0, 0.8)
    ax.set_title("Detection accuracy across the development roadmap",
                 fontweight="bold", color=NAVY)
    save(fig, "map_progression.png")


# 2 -------------------------------------------------------------------------- #
def fig_per_sequence():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = {"EVK4": BLUE, "DAVIS": CYAN, "DVX": AMBER}
    labs = [l for l, _, _ in PER_SEQ]
    vals = [v for _, v, _ in PER_SEQ]
    cols = [cmap[s] for _, _, s in PER_SEQ]
    bars = ax.bar(labs, vals, color=cols, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.axhline(SUMMARY["mAP"], color=RED, ls="--", lw=1.6,
               label=f"mAP {SUMMARY['mAP']:.3f}")
    ax.set_ylabel("AP @ IoU 0.5")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.set_title("Per-sequence AP — final routed model", fontweight="bold", color=NAVY)
    save(fig, "per_sequence_ap.png")


# 3 -------------------------------------------------------------------------- #
def fig_metrics_summary():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3),
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    ks = list(SUMMARY.keys()); vs = [SUMMARY[k] for k in ks]
    cols = [RED, BLUE, CYAN, AMBER]
    bars = a1.bar(ks, vs, color=cols, width=0.6, zorder=3)
    for b, v in zip(bars, vs):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold", color=NAVY)
    a1.set_ylim(0, 1.0); a1.set_ylabel("score")
    a1.set_title("Overall detection metrics", fontweight="bold", color=NAVY)

    # TP/FP/FN donut
    lab = list(COUNTS.keys()); val = [COUNTS[k] for k in lab]
    a2.pie(val, labels=[f"{k}\n{v}" for k, v in COUNTS.items()],
           colors=[CYAN, AMBER, GREY], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor="white"),
           textprops=dict(fontsize=9))
    a2.set_title("Detections (IoU≥0.5)", fontweight="bold", color=NAVY)
    save(fig, "metrics_summary.png")


# 4 -------------------------------------------------------------------------- #
def fig_latency():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labs = [l for l, *_ in LATENCY]
    vox = [v for _, v, _, _ in LATENCY]
    fwd = [v for _, _, v, _ in LATENCY]
    dec = [v for _, _, _, v in LATENCY]
    ax.bar(labs, vox, color=CYAN, label="voxelize", zorder=3)
    ax.bar(labs, fwd, bottom=vox, color=BLUE, label="model forward", zorder=3)
    ax.bar(labs, dec, bottom=[a + b for a, b in zip(vox, fwd)],
           color=AMBER, label="decode", zorder=3)
    for i, (v, f, d) in enumerate(zip(vox, fwd, dec)):
        tot = v + f + d
        ax.text(i, tot + 0.8, f"{tot:.1f} ms", ha="center", fontweight="bold",
                color=NAVY, fontsize=10)
    ax.axhline(BUDGET_MS, color=RED, ls="--", lw=1.8, label="40 ms real-time budget")
    ax.set_ylabel("end-to-end latency (ms / window)")
    ax.set_ylim(0, 45)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Real-time latency vs 40 ms budget (CPU, streaming)",
                 fontweight="bold", color=NAVY)
    save(fig, "latency_breakdown.png")


# 5 -------------------------------------------------------------------------- #
def fig_temporal_gain():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labs = [l for l, _, _ in TEMPORAL]
    before = [b for _, b, _ in TEMPORAL]
    after = [a for _, _, a in TEMPORAL]
    x = range(len(labs)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], before, w, color=GREY,
                label="single 40 ms window", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], after, w, color=CYAN,
                label="±3-window temporal context", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", fontsize=9,
                    fontweight="bold", color=NAVY)
    ax.set_xticks(list(x)); ax.set_xticklabels(labs)
    ax.set_ylabel("AP @ IoU 0.5"); ax.set_ylim(0, 0.85)
    ax.legend(frameon=False)
    ax.set_title("Temporal context — the dim-object recall lever",
                 fontweight="bold", color=NAVY)
    save(fig, "temporal_gain.png")


if __name__ == "__main__":
    fig_progression()
    fig_per_sequence()
    fig_metrics_summary()
    fig_latency()
    fig_temporal_gain()
    print(f"\nAll figures -> {OUT}")
