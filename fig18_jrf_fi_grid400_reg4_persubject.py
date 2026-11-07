"""Fig 18: Per-subject JRF FreeP+FreeE resources, grid 400 and regularization 4.0.

Uses analytical P=2/fold-0 full fits for all 86 subjects. Layout follows
fig9_jrf_fi_bayesian_blo_persubject.py; only this model is plotted.
Run: python fig18_jrf_fi_grid400_reg4_persubject.py
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch

from fig17_jrf_fi_grid_reg import load_resources

ROOT = Path(__file__).resolve().parent


def main(grid_size=400, reg="4.0", output_stem=None):
    plt.style.use(ROOT / "lib" / "plot_style.mplstyle")
    resources = load_resources(grid_size, reg)
    ymax = max(800, 100 * math.ceil(resources.max().item() * 1.05 / 100))
    grid = torch.arange(grid_size, dtype=resources.dtype) / grid_size
    ncols = 10
    nrows = (len(resources) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, curve in enumerate(resources):
        ax = axes[i]
        ax.plot(grid.numpy(), curve.numpy(), lw=2.5, color="blue", alpha=0.7)
        ax.set_title(f"S{i + 1}", fontsize=8, pad=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, ymax])
        # Match the reference's sparse tick labels without changing shared formatters.
        show_ticks = i == (nrows - 1) * ncols
        ax.tick_params(bottom=show_ticks, left=show_ticks,
                       labelbottom=show_ticks, labelleft=show_ticks,
                       labelsize=10, width=0.8)
        for spine in ax.spines.values():
            spine.set_color("gray")
            spine.set_alpha(0.8)
    for ax in axes[len(resources):]:
        ax.axis("off")
    fig.supxlabel("Stimulus (p)", fontsize=18)
    fig.supylabel(r"$\sqrt{J}$", fontsize=18)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.065, top=0.97,
                        wspace=0.2, hspace=0.4)
    output = ROOT / "figures" / (output_stem or Path(__file__).stem)
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {len(resources)} subjects; maximum resources={resources.max().item():.3f}")
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
