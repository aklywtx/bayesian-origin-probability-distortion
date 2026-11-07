"""Grouped Adaptation FreeP+FreeE resources for posterior mean and posterior median.

Both panels use analytical full fits, fold=0, grid=200, reg=1.0 and 26 subjects (1--5 and 19--39).
Run: python fig28_adaptation_fi_posterior_mean_median.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from util import parse_log_file
from util_plot import BAND_ALPHA, LINE_KW, _median_and_iqr

ROOT = Path(__file__).resolve().parent


def load_resources(posterior_p):
    """Load all 26 validated full-fit resource curves in subject order."""
    curves = []
    for sid in list(range(1, 6)) + list(range(19, 40)):
        path = ROOT / "logs" / "Ours" / "CROSSVALID" / (
            f"adaptation_analytical_freeP_freeE_full.py_DistributionSubject{sid}_{posterior_p}_0_1.0_200.txt"
        )
        params = parse_log_file(path, ["sigma_logit", "volume"])
        if not params or any(params.get(key) is None for key in ("sigma_logit", "volume")):
            raise ValueError(f"Missing resource parameters: {path}")
        if params["volume"].shape != (200,) or params["sigma_logit"].numel() != 1:
            raise ValueError(f"Unexpected parameter dimensions: {path}")
        curve = 200 * torch.softmax(params["volume"], dim=0) / torch.sigmoid(params["sigma_logit"]).sqrt()
        if not torch.isfinite(curve).all():
            raise ValueError(f"Nonfinite resources: {path}")
        curves.append(curve)
    return torch.stack(curves)


def main():
    plt.style.use(ROOT / "lib" / "plot_style.mplstyle")
    conditions = (2, 1)
    samples = [load_resources(p) for p in conditions]
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
