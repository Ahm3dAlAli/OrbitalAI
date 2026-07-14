#!/usr/bin/env python3
"""Generate the session's ablation/frontier figures (docs/figures/).

All values are the measured results from the DVX-lever, coast-age, and SSM studies.
  1. dvx_levers.png     — which levers moved Stars3/Thuraya3
  2. coast_age_hist.png — Thuraya3 FP coast-age histogram (frontier proof)
  3. ssm_vs_attn.png    — attention vs state-space backbone (accuracy + speed)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, CYAN, AMBER, RED, GREY = "#0d1b2a", "#1b6ca8", "#2ec4b6", "#f4a261", "#e63946", "#6c757d"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {name}")


# 1 — DVX lever ablation --------------------------------------------------- #
def fig_dvx_levers():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # Stars3: which lever helped
    s_lab = ["baseline\n(g192)", "grid-256", "grid-256\n+hard-neg"]
    s_val = [0.545, 0.613, 0.651]
    a1.bar(s_lab, s_val, color=[GREY, BLUE, CYAN], width=0.6, zorder=3)
    for i, v in enumerate(s_val):
        a1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold", color=NAVY)
    a1.set_ylim(0, 0.75); a1.set_ylabel("AP @ IoU 0.5")
    a1.set_title("Stars3 — resolution + hard-negative mining", fontweight="bold", color=NAVY)
    # Thuraya3: levers that failed vs coasting that worked
    t_lab = ["dim-aug", "reweight", "ctx±5", "g256", "stack", "baseline", "+coast"]
    t_val = [0.454, 0.462, 0.435, 0.469, 0.457, 0.469, 0.506]
    cols = [RED, RED, RED, RED, RED, GREY, CYAN]
    a2.bar(t_lab, t_val, color=cols, width=0.7, zorder=3)
    for i, v in enumerate(t_val):
        a2.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=NAVY)
    a2.axhline(0.469, color=GREY, ls="--", lw=1)
    a2.set_ylim(0, 0.58); a2.set_ylabel("AP @ IoU 0.5")
    a2.set_title("Thuraya3 — 6 levers flat/worse, only coasting helps", fontweight="bold", color=NAVY)
    a2.tick_params(axis="x", labelsize=8.5)
    save(fig, "dvx_levers.png")


# 2 — coast-age FP histogram ---------------------------------------------- #
def fig_coast_age():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = ["0\n(real det.)", "1-2", "3-5", "6-10", "11-20", "21+"]
    counts = [161, 69, 31, 14, 10, 0]
    cols = [BLUE] + [AMBER] * 5
    bars = ax.bar(labels, counts, color=cols, width=0.7, zorder=3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 3, str(c), ha="center", fontweight="bold", color=NAVY)
    ax.set_xlabel("coast-age (windows since real above-threshold evidence)")
    ax.set_ylabel("false-positive boxes")
    ax.set_title("Thuraya3 FP coast-age — no over-coast tail (frontier proof)",
                 fontweight="bold", color=NAVY)
    ax.text(0.98, 0.9, "56% are age-0 real detections\n8% at age ≥ 6\n→ no coast-cap knee",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f5f9fc", ec="#dbe4ee"))
    save(fig, "coast_age_hist.png")


# 3 — SSM vs attention ----------------------------------------------------- #
def fig_ssm():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    seqs = ["EVK4", "DAVIS", "Stars3", "Thuraya3"]
    attn = [0.859, 0.729, 0.545, 0.469]
    ssm = [0.768, 0.619, 0.441, 0.421]
    x = np.arange(len(seqs)); w = 0.38
    a1.bar(x - w / 2, attn, w, color=BLUE, label="attention (0.651)", zorder=3)
    a1.bar(x + w / 2, ssm, w, color=CYAN, label="state-space SSM (0.562)", zorder=3)
    a1.set_xticks(x); a1.set_xticklabels(seqs); a1.set_ylim(0, 1.0)
    a1.set_ylabel("AP @ IoU 0.5"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("Attention vs state-space encoder (grid-192, ctx=3)", fontweight="bold", color=NAVY)
    # speed
    a2.bar(["attention", "SSM"], [160, 94], color=[BLUE, CYAN], width=0.6, zorder=3)
    for i, v in enumerate([160, 94]):
        a2.text(i, v + 3, f"{v}s", ha="center", fontweight="bold", color=NAVY)
    a2.set_ylabel("s / epoch (grid-192)")
    a2.set_title("~1.8× faster", fontweight="bold", color=NAVY)
    save(fig, "ssm_vs_attn.png")


if __name__ == "__main__":
    fig_dvx_levers()
    fig_coast_age()
    fig_ssm()
    print(f"\nAll ablation figures -> {OUT}")
