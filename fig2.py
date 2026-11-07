"""Figure 2: three analytical bias terms.

Source: fig2.tex
A: Fisher information J(p) = 1/(p(1-p)) (blue), and a constant resource line (red).
B: Repulsion term d/dp[1/J(p)^2] = 2p(1-p)(1-2p).
C: Regression term phi(T*p)/Phi(T*p) - phi(T*(1-p))/Phi(T*(1-p)) with T=8.

Usage:
    python fig2.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(SCRIPT_DIR, "lib", "plot_style.mplstyle"))

T = 8.0

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2))

# Panel A: J(p) = 1/(p(1-p))
ax = axes[0]
p = np.linspace(0.02, 0.98, 400)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.plot(p, 1.0 / (p * (1 - p)), color="blue", linewidth=1.3)
ax.axhline(20, color="red", linewidth=1.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 60)
ax.set_xticks([0, 1])
ax.set_yticks([0])
ax.set_xlabel(r"$p$")
ax.text(-0.08, 1.05, "A", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="left")

# Panel B: 2p(1-p)(1-2p)
ax = axes[1]
p = np.linspace(0.0, 1.0, 400)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.plot(p, 2 * p * (1 - p) * (1 - 2 * p), linewidth=1.3, color="blue")
ax.set_xlim(0, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0])
ax.set_xlabel(r"$p$")
ax.text(-0.08, 1.05, "B", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="left")

# Panel C: phi(T*p)/Phi(T*p) - phi(T*(1-p))/Phi(T*(1-p))
ax = axes[2]
p = np.linspace(0.0, 1.0, 400)
eps = 1e-12
reg = (norm.pdf(T * p) / (norm.cdf(T * p) + eps)
       - norm.pdf(T * (1 - p)) / (norm.cdf(T * (1 - p)) + eps))
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.plot(p, reg, color="red", linewidth=1.3)
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0])
ax.set_xlabel(r"$p$")
ax.text(-0.08, 1.05, "C", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="left")

for ax in axes:
    ax.spines["left"].set_position(("outward", 2))
    ax.spines["bottom"].set_position(("outward", 2))

fig.tight_layout()

out_dir = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig2.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
