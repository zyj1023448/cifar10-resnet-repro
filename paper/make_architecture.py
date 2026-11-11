"""Draw the block diagrams used in Figure 1 of the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "fig_architecture.png"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
W = 1.80
CX = 1.05


def box(ax, x, y, w, h, text, fc="#e8f0fb", ec="#2b5d9e", fs=6.8):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                       fc=fc, ec=ec, lw=0.9)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, style="-|>", rad=0.0, color="#333333", lw=0.9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=7,
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw))


def panel_plain(ax):
    ax.set_title("(a) Plain block", fontsize=8.5, pad=6)
    ys = [3.55, 2.70, 1.85, 1.00]
    labels = ["Input $x$", "Conv 3×3 + BN + ReLU", "Conv 3×3 + BN + ReLU", "Output"]
    for y, lab in zip(ys, labels):
        box(ax, CX, y, W, 0.52, lab, fc="#f2f2f2" if y in (3.55, 1.00) else "#e8f0fb")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (CX, a - 0.26), (CX, b + 0.26))
    ax.text(CX, 0.38, "no identity path", ha="center", fontsize=6.3, color="#777777", style="italic")


def panel_res(ax):
    ax.set_title("(b) Residual block", fontsize=8.5, pad=6)
    ys = [3.55, 2.70, 1.85, 1.00]
    labels = ["Input $x$", "Conv 3×3 + BN + ReLU", "Conv 3×3 + BN", "Output"]
    for y, lab in zip(ys, labels):
        box(ax, CX, y, W, 0.52, lab, fc="#f2f2f2" if y in (3.55, 1.00) else "#e8f0fb")
    arrow(ax, (CX, ys[0] - 0.26), (CX, ys[1] + 0.26))
    arrow(ax, (CX, ys[1] - 0.26), (CX, ys[2] + 0.26))
    cy = 1.43
    ax.add_patch(plt.Circle((CX, cy), 0.12, fc="white", ec="#333333", lw=0.9))
    ax.text(CX, cy, "+", ha="center", va="center", fontsize=8)
    arrow(ax, (CX, ys[2] - 0.26), (CX, cy + 0.13))
    arrow(ax, (CX, cy - 0.13), (CX, ys[3] + 0.26))
    # skip connection routed on the right
    xr = CX + W / 2 + 0.28
    arrow(ax, (CX + W / 2, ys[0]), (xr, ys[0]), style="-", color="#d62728", lw=1.0)
    arrow(ax, (xr, ys[0]), (xr, cy), style="-", color="#d62728", lw=1.0)
    arrow(ax, (xr, cy), (CX + 0.13, cy), color="#d62728", lw=1.0)
    ax.text(xr + 0.07, (ys[0] + cy) / 2, "identity", rotation=90, fontsize=6.2, color="#d62728",
            va="center")
    ax.text(CX, 0.38, "$F(x)+x$", ha="center", fontsize=6.3, color="#777777", style="italic")


def panel_se(ax):
    ax.set_title("(c) SE module", fontsize=8.5, pad=6)
    ys = [3.55, 2.70, 1.85, 1.00]
    labels = ["Input $U$ ($C×H×W$)", "Global average pool", "FC $C/r$ → ReLU → FC $C$", "Sigmoid → $s$"]
    for y, lab in zip(ys, labels):
        box(ax, CX, y, W, 0.52, lab, fc="#fdecec", ec="#b23a3a")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (CX, a - 0.26), (CX, b + 0.26))
    cx, cy = CX + W / 2 + 0.42, 1.55
    ax.add_patch(plt.Circle((cx, cy), 0.12, fc="white", ec="#333333", lw=0.9))
    ax.text(cx, cy, "×", ha="center", va="center", fontsize=7)
    arrow(ax, (CX + W / 2, ys[3]), (cx - 0.12, cy), color="#b23a3a", lw=0.9)
    arrow(ax, (CX + W / 2, ys[0]), (cx, ys[0]), style="-", color="#d62728", lw=0.9)
    arrow(ax, (cx, ys[0]), (cx, cy + 0.13), style="-", color="#d62728", lw=0.9)
    ax.text(cx + 0.09, (ys[0] + cy) / 2, "recalibrate", rotation=90, fontsize=6.1, color="#d62728",
            va="center")
    ax.text(cx, 0.52, "Output $U·s$", ha="center", fontsize=6.8)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
for ax, fn in zip(axes, [panel_plain, panel_res, panel_se]):
    ax.set_xlim(0, 2.62)
    ax.set_ylim(0, 3.95)
    ax.axis("off")
    fn(ax)
fig.tight_layout(w_pad=0.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
