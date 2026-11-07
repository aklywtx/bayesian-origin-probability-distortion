"""Fig17: JRF FreeP+FreeE resources for six grid/regularization settings.

Run: python fig17_jrf_fi_grid_reg.py
Uses analytical P=2, fold=0 full fits for subjects 1--86. The grid-200,
reg-1.0 baseline is labeled as the default setting. Curves show median and IQR, as in
fig18_dmr_fi.py, on each condition's original stimulus grid.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))
from util import parse_log_file
from util_plot import BAND_ALPHA, LINE_KW, _median_and_iqr

CONDITIONS = ((200, "0.1"), (200, "1.0"), (200, "4.0"), (400, "0.1"), (400, "1.0"), (400, "4.0"))
LOG_DIR = ROOT / "logs" / "Zhang" / "CROSSVALID"


def load_resources(grid_size, reg, likelihood="analytical", posterior_p=2):
    """Require all 86 finite resource curves with the expected grid length."""
    curves = []
    for sid in range(1, 87):
        path = LOG_DIR / f"jrf_{likelihood}_freeP_freeE_full.py_Subject{sid}_{posterior_p}_0_{reg}_{grid_size}.txt"
        params = parse_log_file(path, ["sigma_logit", "volume"])
        if not params or any(params[key] is None for key in ("sigma_logit", "volume")):
            raise ValueError(f"Missing resource parameters: {path}")
        volume = params["volume"]
        if volume.shape != (grid_size,) or params["sigma_logit"].numel() != 1:
            raise ValueError(f"Unexpected parameter dimensions: {path}")
        curve = grid_size * torch.softmax(volume, dim=0) / torch.sigmoid(params["sigma_logit"]).sqrt()
        if not torch.isfinite(curve).all():
            raise ValueError(f"Nonfinite resources: {path}")
        curves.append(curve)
    return torch.stack(curves)


def main():
    plt.style.use(ROOT / "lib" / "plot_style.mplstyle")
    samples = [load_resources(size, reg) for size, reg in CONDITIONS]
    fig, axes = plt.subplots(1, 6, figsize=(9, 2.2), sharex=True, sharey=True)
    handles = None
    for ax, (size, reg), curves in zip(axes, CONDITIONS, samples):
        grid = torch.arange(size, dtype=curves.dtype) / size
        median, lo, hi = _median_and_iqr(curves)
        band = ax.fill_between(grid, lo, hi, alpha=BAND_ALPHA, color="C0", label="IQR")
        line, = ax.plot(grid, median, color="C0", **LINE_KW, label="Median")
        handles = (line, band)
        title = f"Grid {size}\nreg {reg}"
        title += "\n(default setting)" if (size, reg) == (200, "1.0") else "\n "
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        print(f"Grid {size}, reg {reg}: N={len(curves)}, max upper IQR={hi.max().item():.3f}")
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(r"$\sqrt{J}$")
    fig.supxlabel("Stimulus (p)", fontsize=9, y=0.02)
    fig.legend(handles, ("Median", "IQR"), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=False, borderaxespad=0.2)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.60, wspace=0.2)
    output = ROOT / "figures" / Path(__file__).stem
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
