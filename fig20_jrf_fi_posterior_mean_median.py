"""Fig 20: Grouped JRF FreeP+FreeE resources for posterior mean and posterior median.

Both panels use analytical full fits, fold=0, grid=200, reg=1.0 and subjects 1--86.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fig17_jrf_fi_grid_reg import load_resources
from util_plot import BAND_ALPHA, LINE_KW, _median_and_iqr

ROOT = Path(__file__).resolve().parent


def main():
    plt.style.use(ROOT / "lib" / "plot_style.mplstyle")
    conditions = (2, 1)
    samples = [load_resources(200, "1.0", posterior_p=p) for p in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.6), sharex=True, sharey=True)
    grid = torch.arange(200, dtype=samples[0].dtype) / 200
    for ax, p, curves in zip(axes, conditions, samples):
        median, lo, hi = _median_and_iqr(curves)
        band = ax.fill_between(grid, lo, hi, color="C0", alpha=BAND_ALPHA)
        line, = ax.plot(grid, median, color="C0", **LINE_KW)
        title = "Posterior mean\n(default setting)" if p == 2 else "Posterior median\n "
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        print(f"P={p}: N={len(curves)}, max upper IQR={hi.max().item():.3f}")
    axes[0].set_ylim(bottom=0)
    axes[0].set_yticks([0, 50, 100])
    axes[0].set_ylabel(r"$\sqrt{J}$")
    fig.supxlabel("Stimulus (p)", fontsize=9, y=0.02)
    fig.legend((line, band), ("Median", "IQR"), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1), frameon=False, borderaxespad=0.2)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.19, top=0.70, wspace=0.25)
    output = ROOT / "figures" / Path(__file__).stem
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
