#!/usr/bin/env python3
"""Architecture / pipeline diagram for the proposal (docs/figures/architecture.png)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures", "architecture.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NAVY, BLUE, CYAN, AMBER, GREY = "#0d1b2a", "#1b6ca8", "#2ec4b6", "#f4a261", "#e9eef4"
fig, ax = plt.subplots(figsize=(11, 3.5), dpi=130)
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")


def box(x, y, w, h, title, sub, fc, tc="#0d1b2a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2",
                                linewidth=1.4, edgecolor=BLUE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=tc, zorder=3)
    ax.text(x + w / 2, y + 2.6, sub, ha="center", va="bottom", fontsize=7.3,
            color="#33465c", zorder=3)


def arrow(x0, x1, y=17):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=13, linewidth=1.6, color=NAVY, zorder=1))


# main stages
stages = [
    (1,  "Raw events", "async x,y,p,t\nmicrosecond", GREY),
    (18, "40 ms window\n+ temporal halo", "±3 windows\n(~280 ms track)", GREY),
    (37, "Sparse voxel\ngrid", "[t·2 pol × G × G]\noccupied cells", "#e6f7f4"),
    (56, "Sparse-attention\nencoder", "EvT-SSA + LinaEvT\nmasked/linear attn", "#e6f7f4"),
    (75, "CenterNet head", "heatmap + offset\n+ size → box", "#e6f7f4"),
]
w, h = 15, 15
for i, (x, t, s, fc) in enumerate(stages):
    box(x, 12, w, h, t, s, fc)
    if i < len(stages) - 1:
        arrow(x + w, stages[i + 1][0])

# router band under the head
ax.add_patch(FancyBboxPatch((37, 1), 53, 8, boxstyle="round,pad=0.5,rounding_size=2",
             linewidth=1.3, edgecolor=AMBER, facecolor="#fdf1e5", zorder=2))
ax.text(63.5, 6.6, "Per-sensor router", ha="center", fontsize=8.6, fontweight="bold",
        color="#9a5b1e")
ax.text(63.5, 3.0, "EVK4 → cross-grid ensemble   ·   DAVIS / DVX → temporal model + TTA",
        ha="center", fontsize=7.4, color="#7a4a18")
ax.add_patch(FancyArrowPatch((82.5, 12), (82.5, 9), arrowstyle="-|>",
             mutation_scale=11, linewidth=1.4, color=AMBER, zorder=1))

# output
box(91.5, 12, 7.6, 15, "Box +\nconf.", "one per\nwindow", "#e6f7f4")
arrow(90, 91.5)

ax.text(50, 31.5, "OrbitSight pipeline — raw NVS stream to routed RSO detection",
        ha="center", fontsize=11, fontweight="bold", color=NAVY)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"[arch] {OUT}")
